#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the llm-d dashboard as our own container image, push it to Artifact
# Registry, and run it *in-cluster* next to the llm-d gateway (so the live
# gateway loop and the Pane-2 A/B bench talk to cluster-local Services, not
# cloudtop port-forwards).
#
# Usage:
#   scripts/86_deploy_dashboard_image.sh [--port-forward] [--no-build]
#
#   --port-forward   after rollout, keep a kubectl port-forward alive on
#                    localhost:${DASHBOARD_PORT:-8085} -> svc/llmd-live-dashboard
#   --no-build       skip docker build/push, just (re)deploy the manifest
#
# Knobs (env): DASHBOARD_AR_REPO (default llmd-tpu), DASHBOARD_IMAGE_TAG (default
# git short sha, also tagged :latest), BASELINE_URLS, POD_METRICS_HOSTS, AB_CONCURRENCY.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_bin docker gcloud kubectl envsubst
get_credentials

PORT_FORWARD=0
DO_BUILD=1
for a in "$@"; do
  case "$a" in
    --port-forward) PORT_FORWARD=1 ;;
    --no-build)     DO_BUILD=0 ;;
    *) die "unknown flag: $a (use --port-forward / --no-build)" ;;
  esac
done

# The dashboard image lives in its own AR repo (env.sh's AR_REPO defaults to the
# vLLM-TPU serving repo, which may not exist in this project).
DASHBOARD_AR_REPO="${DASHBOARD_AR_REPO:-llmd-tpu}"
DASHBOARD_IMAGE_TAG="${DASHBOARD_IMAGE_TAG:-$(git -C "${HARNESS_ROOT}" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}"
IMAGE_REPO="${AR_HOST}/${PROJECT_ID}/${DASHBOARD_AR_REPO}/llmd-dashboard"
export DASHBOARD_IMAGE="${IMAGE_REPO}:${DASHBOARD_IMAGE_TAG}"

if ! gcloud artifacts repositories describe "${DASHBOARD_AR_REPO}" \
      --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  die "Artifact Registry repo ${DASHBOARD_AR_REPO} not found in ${PROJECT_ID}/${REGION}. Create it or set DASHBOARD_AR_REPO."
fi

DASHBOARD_DIR="${DASHBOARD_DIR:-$(cd "${HARNESS_ROOT}/.." && pwd)/dashboard}"
if [[ ! -d "${DASHBOARD_DIR}" ]]; then
  DASHBOARD_DIR="${HARNESS_ROOT}/dashboard"
fi

if (( DO_BUILD )); then
  log "Configuring docker auth for ${AR_HOST}"
  gcloud auth configure-docker "${AR_HOST}" --quiet >/dev/null

  log "Building ${DASHBOARD_IMAGE} (linux/amd64) from ${DASHBOARD_DIR}"
  docker build --platform linux/amd64 -t "${DASHBOARD_IMAGE}" -t "${IMAGE_REPO}:latest" "${DASHBOARD_DIR}"

  log "Pushing ${DASHBOARD_IMAGE} and :latest"
  docker push "${DASHBOARD_IMAGE}"
  docker push "${IMAGE_REPO}:latest"
fi

# In-cluster A/B targets. "Without llm-d" = the vLLM Service directly (kube-proxy
# round-robin, bypassing gateway+EPP); KV hit-rate scraped from the same Service.
#
# The baseline MUST be the same pool the "with" side routes to, otherwise Pane 2
# compares capacity, not routing. Measured 2026-10-07: adding the 2x GB200 pods to
# a TPU-only "with" side made "without llm-d" win on every metric (824 vs 731 tok/s).
#
# AB_POOL picks that pool for BOTH sides (B2, Oct 7 sync): KV-cache-aware routing can
# only show a delta over >= 2 endpoints. The TPU pool is a single pod (TP=2 DP=4 on one
# tpu7x host), so the default is the 2-pod GB200 pool:
#   AB_POOL=gpu  -> baseline = llmd-gpu-service, "with" = gateway + x-llm-d-workload: one-shot
#   AB_POOL=tpu  -> baseline = llmd-tpu-service, "with" = gateway + x-llm-d-workload: multi-turn
# Both services are always scraped for the stack view's per-pod KV hit rates.
GPU_NAMESPACE="${GPU_NAMESPACE:-${NAMESPACE}-gpu}"
_tpu_svc="llmd-tpu-service.${NAMESPACE}.svc.cluster.local:8000"
_gpu_svc="llmd-gpu-service.${GPU_NAMESPACE}.svc.cluster.local:8000"
# Headless twins for the per-pod /metrics scrape (KV hit): DNS of a headless Service = all pod IPs.
export GPU_NAMESPACE
envsubst '${NAMESPACE} ${GPU_NAMESPACE}' < "${HARNESS_ROOT}/manifests/dashboard/metrics-headless.yaml.tmpl" | kubectl apply -f -
_tpu_pods="llmd-tpu-pods.${NAMESPACE}.svc.cluster.local:8000"
_gpu_pods="llmd-gpu-pods.${GPU_NAMESPACE}.svc.cluster.local:8000"
_metrics="${_tpu_pods}"
if kubectl -n "${GPU_NAMESPACE}" get svc llmd-gpu-service >/dev/null 2>&1; then
  _metrics="${_metrics},${_gpu_pods}"
else
  log "WARN: llmd-gpu-service not found in ${GPU_NAMESPACE}; forcing AB_POOL=tpu"
  AB_POOL="tpu"
fi
export AB_POOL="${AB_POOL:-gpu}"
case "${AB_POOL}" in
  gpu) _baseline="http://${_gpu_svc}"; _ab_kv="${_gpu_pods}"; _ab_wl="one-shot";  _ab_obj="oneshot-fast" ;;
  tpu) _baseline="http://${_tpu_svc}"; _ab_kv="${_tpu_pods}"; _ab_wl="multi-turn"; _ab_obj="agentic-multi-turn" ;;
  *) die "AB_POOL must be gpu or tpu (got '${AB_POOL}')" ;;
esac
export BASELINE_URLS="${BASELINE_URLS:-${_baseline}}"
export POD_METRICS_HOSTS="${POD_METRICS_HOSTS:-${_metrics}}"
export AB_KV_HOSTS="${AB_KV_HOSTS:-${_ab_kv}}"
export AB_WORKLOAD="${AB_WORKLOAD:-${_ab_wl}}"
export AB_OBJECTIVE="${AB_OBJECTIVE:-${_ab_obj}}"
# Oct 9: 0 = no synthetic A/B sessions; the Hermes agents (odd LIVE_GATEWAY_WORKERS) drive the A/B pool themselves.
export AB_CONCURRENCY="${AB_CONCURRENCY:-0}"
log "A/B pool: ${AB_POOL} (baseline=${BASELINE_URLS}, workload=${AB_WORKLOAD}, objective=${AB_OBJECTIVE})"

# Real hardware labels (0.1 in the Oct 6 review): read from the nodes the decode pods actually run on,
# so the UI never says "G4" when the pool is GB200 (or vice-versa). Override with HW_TPU / HW_GPU.
_pretty_accel() {   # gke accelerator label -> display name
  case "$1" in
    tpu7x|tpu-v7x*) echo "TPU v7x" ;;
    tpu-v6e*|tpu6e*) echo "TPU v6e Trillium" ;;
    tpu-v5p*) echo "TPU v5p" ;;
    tpu-v5-lite*|tpu-v5e*) echo "TPU v5e" ;;
    nvidia-gb200*) echo "NVIDIA GB200" ;;
    nvidia-b200*) echo "NVIDIA B200" ;;
    nvidia-h200*) echo "NVIDIA H200" ;;
    nvidia-h100*) echo "NVIDIA H100" ;;
    nvidia-rtx-pro-6000*|nvidia-rtx-6000*) echo "NVIDIA RTX PRO 6000 (G4)" ;;
    nvidia-l4*) echo "NVIDIA L4" ;;
    "") echo "" ;;
    *) echo "$1" ;;
  esac
}
_node_label() {   # <ns> <deployment> <label-key> -> label value of the first running pod's node
  local ns="$1" dep="$2" key="$3" node
  node="$(kubectl -n "${ns}" get pods -l "app=${dep}" --field-selector=status.phase=Running \
          -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)"
  [[ -z "${node}" ]] && node="$(kubectl -n "${ns}" get pods -o jsonpath="{.items[?(@.metadata.labels.app==\"${dep}\")].spec.nodeName}" 2>/dev/null | awk '{print $1}')"
  [[ -z "${node}" ]] && { echo ""; return; }
  kubectl get node "${node}" -o jsonpath="{.metadata.labels['${key}']}" 2>/dev/null || true
}
_ready() { kubectl -n "$1" get deploy "$2" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0; }
TPU_DEPLOY="${TPU_DEPLOY:-llmd-tpu-decode}"
GPU_DEPLOY="${GPU_DEPLOY:-llmd-gpu-decode}"
_tpu_lbl="$(_node_label "${NAMESPACE}" "${TPU_DEPLOY}" 'cloud\.google\.com/gke-tpu-accelerator')"
_gpu_lbl="$(_node_label "${GPU_NAMESPACE}" "${GPU_DEPLOY}" 'cloud\.google\.com/gke-accelerator')"
# Oct 8 decision: the UI says just "TPU" and "GPU" (heterogeneous accelerator pool); the detected chip
# (e.g. "TPU v7x", "NVIDIA GB200") is logged here for the record. HW_TPU / HW_GPU still override.
_tpu_detected="$(_pretty_accel "${_tpu_lbl}")"; _gpu_detected="$(_pretty_accel "${_gpu_lbl}")"
export HW_TPU="${HW_TPU:-TPU}"
export HW_GPU="${HW_GPU:-GPU}"
# Oct 9: even workers -> agents' pool via the gateway, odd workers -> A/B pool following the Routing toggle.
export LIVE_GATEWAY_WORKERS="${LIVE_GATEWAY_WORKERS:-48}"   # Oct 9 19:30Z: 24 agents per pool
# Agent pacing (s of think time per turn). 0 = closed loop (default since Oct 9 19:15Z, user): the next turn starts when the
# reply lands, so the faster routing mode also completes more turns/s and throughput moves with TTFT. 0.8 = equal rate per
# pool (throughput pinned identical, only latency differs). Runtime switch: Settings > Agent pacing (POST api/agents).
export AGENT_THINK_S="${AGENT_THINK_S:-0}"
# A/B workload shape (Oct 9 19:20Z): 4 turns (= GPU replicas), ~6k-token workspace prefix, 32 output tokens, 16 agents per pool.
export AB_TURNS="${AB_TURNS:-4}" AB_PREFIX_PARAS="${AB_PREFIX_PARAS:-160}" AB_MAX_TOKENS="${AB_MAX_TOKENS:-32}"   # 160 paras ≈ 9.5k tok: measured +76% tok/s, -37% TTFT p50, -54% p95 (19:30Z)
export ENABLE_KV_TIERING="${ENABLE_KV_TIERING:-0}"
export ENABLE_KV_OFFLOAD="${ENABLE_KV_OFFLOAD:-0}"
export ENABLE_KV_STORAGE_TIER="${ENABLE_KV_STORAGE_TIER:-0}"
if [[ "${AB_POOL}" == "tpu" ]]; then
  export KV_TIER_ENABLED="${KV_TIER_ENABLED:-${ENABLE_KV_OFFLOAD}}"
else
  export KV_TIER_ENABLED="${KV_TIER_ENABLED:-${ENABLE_KV_TIERING}}"
fi
if [[ "${ENABLE_KV_TIERING}" == "1" ]]; then
  export KV_TIER_HBM_CAP_BYTES="${KV_TIER_HBM_CAP_BYTES:-8589934592}"
else
  export KV_TIER_HBM_CAP_BYTES="${KV_TIER_HBM_CAP_BYTES:-}"
fi
export KV_TIER_CPU_BYTES="${KV_TIER_CPU_BYTES:-68719476736}"
# Oct 14: host pool size per node pool for the Memory panel (both pools can run a host tier). Empty = unknown ("—");
# the A/B pool falls back to KV_TIER_CPU_BYTES. GPU pool default = the kvtier template default (64 GiB per replica).
export KV_TIER_CPU_BYTES_GPU="${KV_TIER_CPU_BYTES_GPU:-68719476736}"
export KV_TIER_CPU_BYTES_TPU="${KV_TIER_CPU_BYTES_TPU:-}"
_pvc_bytes() {   # <ns> <pvc> -> capacity in bytes (binary and decimal suffixes), empty if the PVC is missing
  local cap
  cap="$(kubectl -n "$1" get pvc "$2" -o jsonpath='{.status.capacity.storage}' 2>/dev/null || true)"
  [[ -z "${cap}" ]] && { echo ""; return; }
  echo "${cap}" | awk '{ n=$0; u=""; if (match(n, /[A-Za-z]+$/)) { u=substr(n, RSTART); n=substr(n, 1, RSTART-1) }
    m["Ki"]=2^10; m["Mi"]=2^20; m["Gi"]=2^30; m["Ti"]=2^40; m["Pi"]=2^50; m["k"]=1e3; m["K"]=1e3; m["M"]=1e6; m["G"]=1e9; m["T"]=1e12; m["P"]=1e15; m[""]=1
    printf "%.0f\n", n * m[u] }'
}
# The Tier 3 capacity shown in the UI is the storage PVC's size when it exists (the GPU pool and the scale-out
# replicas share one PVC); 1 TiB otherwise.
export KV_TIER_STORAGE_BYTES="${KV_TIER_STORAGE_BYTES:-$(_pvc_bytes "${GPU_NAMESPACE}" "${KV_STORAGE_PVC:-llm-d-kv-cache-storage}")}"
export KV_TIER_STORAGE_BYTES="${KV_TIER_STORAGE_BYTES:-1099511627776}"
# Tier 3 panel: the scale-out replicas from scripts/62_deploy_kv_storage_replicas.sh. The dashboard scrapes and
# drives them through their headless Service; with no hosts the Tier 3 button stays disabled.
export ENABLE_KV_STORAGE_REPLICAS="${ENABLE_KV_STORAGE_REPLICAS:-0}"
_st_pods="llmd-gpu-storage-pods.${GPU_NAMESPACE}.svc.cluster.local:8000"
if [[ "${ENABLE_KV_STORAGE_REPLICAS}" == "1" ]]; then
  if kubectl -n "${GPU_NAMESPACE}" get svc llmd-gpu-storage-pods >/dev/null 2>&1; then
    export AB_KV_STORAGE_HOSTS="${AB_KV_STORAGE_HOSTS:-${_st_pods}}"
  else
    warn "ENABLE_KV_STORAGE_REPLICAS=1 but svc/llmd-gpu-storage-pods is missing in ${GPU_NAMESPACE}; run scripts/62_deploy_kv_storage_replicas.sh first. The Tier 3 panel stays off."
  fi
fi
export AB_KV_STORAGE_HOSTS="${AB_KV_STORAGE_HOSTS:-}" AB_KV_STORAGE_URLS="${AB_KV_STORAGE_URLS:-}"
export KV_STORAGE_WORKERS="${KV_STORAGE_WORKERS:-6}" KV_STORAGE_RING="${KV_STORAGE_RING:-5}" KV_STORAGE_THINK_S="${KV_STORAGE_THINK_S:-0.5}"
export KV_STORAGE_REPLICA_HBM_CAP_BYTES="${KV_STORAGE_REPLICA_HBM_CAP_BYTES:-4294967296}"
export KV_STORAGE_REPLICA_CPU_BYTES="${KV_STORAGE_REPLICA_CPU_BYTES:-4294967296}"
[[ -n "${AB_KV_STORAGE_HOSTS}" ]] && log "Tier 3 replicas: ${AB_KV_STORAGE_HOSTS} (ring ${KV_STORAGE_WORKERS}x${KV_STORAGE_RING} sessions)"
export HW_TPU_PODS="${HW_TPU_PODS:-$(_ready "${NAMESPACE}" "${TPU_DEPLOY}")}"; export HW_TPU_PODS="${HW_TPU_PODS:-0}"
export HW_GPU_PODS="${HW_GPU_PODS:-$(_ready "${GPU_NAMESPACE}" "${GPU_DEPLOY}")}"; export HW_GPU_PODS="${HW_GPU_PODS:-0}"
# Oct 8: the Accelerators row names the chip, the TPU topology and the chip count. Chip names come from the detected
# node labels above; topology from cloud.google.com/gke-tpu-topology; chip counts from the topology product (TPU) and
# the decode pods' nvidia.com/gpu limits (GPU). HW_* overrides win; empty = the UI shows the pool name and 8 glyphs.
_topo_chips() { local t="${1//x/ }" n=1 d; [[ -z "$1" ]] && { echo ""; return; }; for d in ${t}; do n=$(( n * d )); done; echo "${n}"; }
_tpu_limit_sum() {   # <ns> <deployment> -> sum of google.com/tpu limits over running pods
  kubectl -n "$1" get pods -l "app=$2" --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.resources.limits.google\.com/tpu}{"\n"}{end}{end}' 2>/dev/null \
    | awk 'NF{s+=$1} END{print s+0}'
}
_gpu_limit_sum() {   # <ns> <deployment> -> sum of nvidia.com/gpu limits over running pods
  kubectl -n "$1" get pods -l "app=$2" --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.resources.limits.nvidia\.com/gpu}{"\n"}{end}{end}' 2>/dev/null \
    | awk 'NF{s+=$1} END{print s+0}'
}
export HW_TPU_CHIP="${HW_TPU_CHIP:-TorchTPU}"
export HW_TPU_DETAIL="${HW_TPU_DETAIL:-vllm-torchtpu}"
export HW_TPU_TOPOLOGY="${HW_TPU_TOPOLOGY:-}"
_tpu_sum="$(_tpu_limit_sum "${NAMESPACE}" "${TPU_DEPLOY}")"
export HW_TPU_CHIPS="${HW_TPU_CHIPS:-$( (( _tpu_sum > 0 )) && echo "${_tpu_sum}" || _topo_chips "${HW_TPU_TOPOLOGY}" )}"; export HW_TPU_CHIPS="${HW_TPU_CHIPS:-0}"
export HW_GPU_CHIP="${HW_GPU_CHIP:-Cloud GPU}"
export HW_GPU_DETAIL="${HW_GPU_DETAIL:-vLLM / SGLang}"
export HW_GPU_CHIPS="${HW_GPU_CHIPS:-$(_gpu_limit_sum "${GPU_NAMESPACE}" "${GPU_DEPLOY}")}"; export HW_GPU_CHIPS="${HW_GPU_CHIPS:-0}"
log "Hardware labels for the UI: TPU='${HW_TPU}' x${HW_TPU_PODS} (chip '${HW_TPU_CHIP:-?}', topology '${HW_TPU_TOPOLOGY:-?}', ${HW_TPU_CHIPS} chips), GPU='${HW_GPU}' x${HW_GPU_PODS} (chip '${HW_GPU_CHIP:-?}', ${HW_GPU_CHIPS} chips), live agent workers=${LIVE_GATEWAY_WORKERS}"

log "Deploying to ${NAMESPACE}: image=${DASHBOARD_IMAGE} baseline=${BASELINE_URLS} metrics=${POD_METRICS_HOSTS}"
kubectl get ns "${NAMESPACE}" >/dev/null 2>&1 || kubectl create ns "${NAMESPACE}"
kubectl -n "${NAMESPACE}" delete configmap llmd-dashboard-files --ignore-not-found
envsubst '${NAMESPACE} ${DASHBOARD_IMAGE} ${SERVED_MODEL_NAME} ${BASELINE_URLS} ${POD_METRICS_HOSTS} ${AB_CONCURRENCY} ${AB_POOL} ${AB_KV_HOSTS} ${AB_WORKLOAD} ${AB_OBJECTIVE} ${HW_TPU} ${HW_GPU} ${HW_TPU_PODS} ${HW_GPU_PODS} ${HW_TPU_CHIP} ${HW_TPU_DETAIL} ${HW_TPU_TOPOLOGY} ${HW_TPU_CHIPS} ${HW_GPU_CHIP} ${HW_GPU_DETAIL} ${HW_GPU_CHIPS} ${LIVE_GATEWAY_WORKERS} ${AGENT_THINK_S} ${AB_TURNS} ${AB_PREFIX_PARAS} ${AB_MAX_TOKENS} ${ENABLE_KV_TIERING} ${ENABLE_KV_OFFLOAD} ${KV_TIER_ENABLED} ${KV_TIER_HBM_CAP_BYTES} ${KV_TIER_CPU_BYTES} ${KV_TIER_CPU_BYTES_GPU} ${KV_TIER_CPU_BYTES_TPU} ${ENABLE_KV_STORAGE_TIER} ${KV_TIER_STORAGE_BYTES} ${AB_KV_STORAGE_HOSTS} ${AB_KV_STORAGE_URLS} ${KV_STORAGE_WORKERS} ${KV_STORAGE_RING} ${KV_STORAGE_THINK_S} ${KV_STORAGE_REPLICA_HBM_CAP_BYTES} ${KV_STORAGE_REPLICA_CPU_BYTES}' \
  < "${HARNESS_ROOT}/manifests/dashboard/deployment.yaml.tmpl" \
  | kubectl apply -f -
kubectl -n "${NAMESPACE}" rollout status deploy/llmd-live-dashboard --timeout=120s
log "Deployed ${DASHBOARD_IMAGE}"
if systemctl --user is-active --quiet port-forward.service 2>/dev/null; then
  systemctl --user restart port-forward.service && log "Restarted port-forward.service on :8085"
fi

# Team endpoint: an Internal LoadBalancer (VPC-reachable, no public IP). Disable with DASHBOARD_ILB=0.
if [[ "${DASHBOARD_ILB:-1}" == "1" ]]; then
  envsubst '${NAMESPACE}' < "${HARNESS_ROOT}/manifests/dashboard/service-ilb.yaml.tmpl" | kubectl apply -f -
  _ilb_ip=""
  for _ in $(seq 1 30); do
    _ilb_ip="$(kubectl -n "${NAMESPACE}" get svc llmd-live-dashboard-ilb -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)"
    [[ -n "${_ilb_ip}" ]] && break
    sleep 5
  done
  if [[ -n "${_ilb_ip}" ]]; then
    log "Internal LB ready: http://${_ilb_ip}/  (reachable from the VPC; add ?view=stack to open on the stack view)"
  else
    warn "Internal LB has no IP yet: kubectl -n ${NAMESPACE} get svc llmd-live-dashboard-ilb -w"
  fi
fi

DASHBOARD_PORT="${DASHBOARD_PORT:-8085}"
# Bind 0.0.0.0 so remote workstation proxies targeting <host>:8085 can reach it.
DASHBOARD_BIND="${DASHBOARD_BIND:-0.0.0.0}"
if (( PORT_FORWARD )); then
  log "Port-forwarding svc/llmd-live-dashboard -> http://${DASHBOARD_BIND}:${DASHBOARD_PORT}/  (Ctrl-C to stop)"
  while true; do
    kubectl -n "${NAMESPACE}" port-forward --address "${DASHBOARD_BIND}" svc/llmd-live-dashboard "${DASHBOARD_PORT}:80" || true
    sleep 1
  done
else
  log "Open it locally with:"
  echo "    kubectl -n ${NAMESPACE} port-forward --address ${DASHBOARD_BIND} svc/llmd-live-dashboard ${DASHBOARD_PORT}:80"
  echo "    # then browse http://localhost:${DASHBOARD_PORT}/"
fi
