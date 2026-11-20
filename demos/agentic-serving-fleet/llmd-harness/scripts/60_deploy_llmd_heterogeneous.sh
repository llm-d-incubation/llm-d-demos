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

# 60_deploy_llmd_heterogeneous.sh — Deploy the heterogeneous llm-d stack for
# google/gemma-4-26B-A4B-it:
#   1. GAIE CRDs + llm-d Router CRDs (InferencePool, InferenceObjective)
#   2. Optimized TPU v7x Multi-Turn Pool (`llmd-tpu-decode` + `llmd-tpu-epp`
#      with Flow Control + Prefix Cache Scorer + CPU DRAM KV Offload)
#   3. GPU One-Shot Pool (`llmd-gpu-decode` + `llmd-gpu-epp` with
#      low-latency stateless load-aware routing)
#   4. Unified Envoy Header-Routing Gateway (`llm-d-gateway`) that
#      splits `x-llm-d-workload: multi-turn` -> TPU EPP and `one-shot` -> GPU EPP.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

need_bin kubectl helm envsubst

if [[ ! -f "${LLMD_REPO}/guides/env.sh" ]]; then
  die "Upstream llm-d checkout not found at LLMD_REPO=${LLMD_REPO}. Run: git clone https://github.com/llm-d/llm-d.git \"${LLMD_REPO}\""
fi

export REPO_ROOT="${LLMD_REPO}"
# shellcheck source=/dev/null
source "${LLMD_REPO}/guides/env.sh"

get_credentials

# ---------- 1. CRDs & Namespace ----------------------------------------------
log "Installing Gateway API Inference Extension & llm-d Router CRDs"
run kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml"
run kubectl apply -f "https://github.com/llm-d/llm-d-router/${ROUTER_RELEASE_URL}/manifests.yaml" || true

export GPU_NAMESPACE="${GPU_NAMESPACE:-${NAMESPACE}-gpu}"

log "Ensuring namespaces ${NAMESPACE} (TPU + Gateway) and ${GPU_NAMESPACE} (GPU One-Shot) and HuggingFace secrets"
for ns in "${NAMESPACE}" "${GPU_NAMESPACE}"; do
  kubectl create namespace "${ns}" --dry-run=client -o yaml | kubectl apply -f -
  if [[ -n "${HF_TOKEN:-}" ]]; then
    kubectl create secret generic llm-d-hf-token \
      --from-literal="HF_TOKEN=${HF_TOKEN}" \
      --namespace "${ns}" --dry-run=client -o yaml | kubectl apply -f -
  else
    for src_ns in llm-d-agentic-serving llm-d-singlehost llm-d-pd-disaggregation; do
      if kubectl get secret llm-d-hf-token -n "${src_ns}" >/dev/null 2>&1; then
        existing_tok="$(kubectl get secret llm-d-hf-token -n "${src_ns}" -o jsonpath='{.data.HF_TOKEN}' | base64 -d)"
        if [[ -n "${existing_tok}" ]]; then
          kubectl create secret generic llm-d-hf-token \
            --from-literal="HF_TOKEN=${existing_tok}" \
            --namespace "${ns}" --dry-run=client -o yaml | kubectl apply -f -
          break
        fi
      fi
    done
  fi
  if kubectl get secret ar-torchtpu-pull -n llm-d-singlehost >/dev/null 2>&1; then
    kubectl get secret ar-torchtpu-pull -n llm-d-singlehost -o json \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); d["metadata"]={"name":"ar-torchtpu-pull","namespace":"'"${ns}"'"}; print(json.dumps(d))' \
      | kubectl apply -f -
  fi
done

# The storage KV tier needs its RWX PVC before anything is deployed (see manifests/storage/README.md).
if [[ "${ENABLE_KV_STORAGE_TIER:-0}" == "1" ]]; then
  [[ "${ENABLE_KV_TIERING:-0}" == "1" || "${ENABLE_KV_OFFLOAD:-0}" == "1" ]] || die "ENABLE_KV_STORAGE_TIER=1 requires ENABLE_KV_TIERING=1 or ENABLE_KV_OFFLOAD=1"
  if [[ "${ENABLE_KV_TIERING:-0}" == "1" ]]; then
    phase="$(kubectl get pvc "${KV_STORAGE_PVC}" -n "${GPU_NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    [[ "${phase}" == "Bound" ]] || die "ENABLE_KV_STORAGE_TIER=1 with ENABLE_KV_TIERING=1 but PVC ${KV_STORAGE_PVC} in ${GPU_NAMESPACE} is not Bound (got '${phase:-missing}')"
  fi
  if [[ "${ENABLE_KV_OFFLOAD:-0}" == "1" ]]; then
    phase="$(kubectl get pvc "${KV_STORAGE_PVC}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    [[ "${phase}" == "Bound" ]] || die "ENABLE_KV_STORAGE_TIER=1 with ENABLE_KV_OFFLOAD=1 but PVC ${KV_STORAGE_PVC} in ${NAMESPACE} is not Bound (got '${phase:-missing}')"
  fi
fi


# ---------- 2. Deploy TPU Multi-Turn EPP Router (with Flow Control) ----------
log "Deploying TPU Multi-Turn EPP Router (llmd-tpu-epp in ${NAMESPACE}) with Flow Control + Prefix Cache Scoring"
run helm upgrade --install "llmd-tpu" "${ROUTER_STANDALONE_CHART}" \
  -f "${LLMD_REPO}/guides/recipes/router/base.values.yaml" \
  -f "${LLMD_REPO}/guides/flow-control/router/flow-control.values.yaml" \
  --set 'router.modelServers.matchLabels.llm-d\.ai/guide=null' \
  --set router.modelServers.matchLabels.app="llmd-tpu-decode" \
  -n "${NAMESPACE}" --version "${ROUTER_CHART_VERSION}"

# ---------- 3. Deploy GPU One-Shot EPP Router --------------------------------
log "Deploying GPU One-Shot EPP Router (llmd-gpu-epp in ${GPU_NAMESPACE}) with Load-Aware Fast Dispatch"
run helm upgrade --install "llmd-gpu" "${ROUTER_STANDALONE_CHART}" \
  -f "${LLMD_REPO}/guides/recipes/router/base.values.yaml" \
  -f "${LLMD_REPO}/guides/optimized-baseline/router/optimized-baseline.values.yaml" \
  --set 'router.modelServers.matchLabels.llm-d\.ai/guide=null' \
  --set router.modelServers.matchLabels.app="llmd-gpu-decode" \
  -n "${GPU_NAMESPACE}" --version "${ROUTER_CHART_VERSION}"

# ---------- 4. Deploy TPU v7x Multi-Turn Model Server Pool -------------------
export ENABLE_KV_OFFLOAD="${ENABLE_KV_OFFLOAD:-0}"
export TPU_KV_STORAGE_VOLUME_MOUNT_LINE="" TPU_KV_STORAGE_VOLUME_LINE=""
if [[ "${ENABLE_KV_OFFLOAD}" == "1" ]]; then
  TPU_HBM_CAP_ARG=""
  if [[ -n "${KV_TIER_HBM_CAP_BYTES:-}" ]]; then
    TPU_HBM_CAP_ARG="--kv-cache-memory-bytes=${KV_TIER_HBM_CAP_BYTES} "
  fi
  TPU_KV_EXTRA="{\"spec_name\":\"TPUCPUOffloadingSpec\",\"spec_module_path\":\"vllm_torchtpu.offload.cpu_tpu\",\"cpu_bytes_to_use\":${KV_TIER_CPU_BYTES:-68719476736},\"block_size\":256}"
  if [[ "${ENABLE_KV_STORAGE_TIER:-0}" == "1" ]]; then
    TPU_KV_STORAGE_ROOT_DIR="${TPU_KV_STORAGE_ROOT_DIR:-${KV_STORAGE_MOUNT_PATH}/kv-cache/vllm-tpu}"
    TPU_KV_EXTRA="{\"spec_name\":\"TPUCPUOffloadingSpec\",\"spec_module_path\":\"vllm_torchtpu.offload.cpu_tpu\",\"cpu_bytes_to_use\":${KV_TIER_CPU_BYTES:-68719476736},\"block_size\":256,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"${TPU_KV_STORAGE_ROOT_DIR}\",\"n_read_threads\":${KV_STORAGE_IO_THREADS:-16},\"n_write_threads\":${KV_STORAGE_IO_THREADS:-16}}]}"
    export TPU_KV_STORAGE_VOLUME_MOUNT_LINE="- {name: kv-storage, mountPath: ${KV_STORAGE_MOUNT_PATH}}"
    export TPU_KV_STORAGE_VOLUME_LINE="- {name: kv-storage, persistentVolumeClaim: {claimName: ${KV_STORAGE_PVC}}}"
    log "TPU KV storage tier: ENABLED (PVC ${KV_STORAGE_PVC} at ${KV_STORAGE_MOUNT_PATH}, root ${TPU_KV_STORAGE_ROOT_DIR})"
  fi
  export KV_TRANSFER_ARG="${TPU_HBM_CAP_ARG}--kv-offloading-backend native --kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":${TPU_KV_EXTRA}}'"
  log "TPU KV Offloading: ENABLED (host pool ${KV_TIER_CPU_BYTES:-68719476736} bytes/engine, HBM KV cap ${KV_TIER_HBM_CAP_BYTES:-none})"
else
  export KV_TRANSFER_ARG=""
fi

TPU_TEMPLATE="${HARNESS_ROOT}/manifests/modelserver/tpu-torchtpu/tpu-pool.yaml.tmpl"
if [[ "${TPU_BACKEND:-torchtpu}" == "torchax" ]]; then
  TPU_TEMPLATE="${HARNESS_ROOT}/manifests/modelserver/tpu-torchax/tpu-pool.yaml.tmpl"
else
  QUANT_PATCH_DIR="${HARNESS_ROOT}/manifests/modelserver/tpu-torchtpu/quant_patch"
  if [[ -d "${QUANT_PATCH_DIR}" ]]; then
    log "Applying ConfigMap torchtpu-quant-patch (compressed-tensors FP8 + TPU L2/L3 KV offload) in ${NAMESPACE}"
    kubectl create configmap torchtpu-quant-patch \
      --from-file="${QUANT_PATCH_DIR}/__init__.py" \
      --from-file="${QUANT_PATCH_DIR}/compressed_tensors.py" \
      --from-file="${QUANT_PATCH_DIR}/fp8.py" \
      --from-file="${QUANT_PATCH_DIR}/cpu_tpu.py" \
      -n "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
  fi
fi

log "Deploying TPU Multi-Turn Model Server Pool (${MODEL_ID} [${TPU_BACKEND:-torchtpu}] on ${TPU_REPLICAS} x tpu7x-${TPU_TOPOLOGY})"
envsubst '${TPU_REPLICAS} ${TPU_SERVING_IMAGE} ${TORCHAX_SERVING_IMAGE} ${MODEL_ID} ${SERVED_MODEL_NAME} ${MAX_MODEL_LEN} ${TPU_TENSOR_PARALLEL_SIZE} ${TPU_DATA_PARALLEL_SIZE} ${KV_TRANSFER_ARG} ${TPU_CHIPS_PER_NODE} ${TPU_TOPOLOGY} ${ENABLE_KV_OFFLOAD} ${TPU_KV_STORAGE_VOLUME_MOUNT_LINE} ${TPU_KV_STORAGE_VOLUME_LINE}' \
  < "${TPU_TEMPLATE}" \
  | kubectl apply -n "${NAMESPACE}" -f -

# ---------- 5. Deploy GPU One-Shot Model Server Pool -------------------------
if nodepool_exists "${GPU_NODEPOOL}"; then
  detected_accel="$(gc container node-pools describe "${GPU_NODEPOOL}" --cluster="${CLUSTER_NAME}" --location="${CLUSTER_LOCATION}" --format="value(config.accelerators[0].acceleratorType)" 2>/dev/null || true)"
  detected_count="$(gc container node-pools describe "${GPU_NODEPOOL}" --cluster="${CLUSTER_NAME}" --location="${CLUSTER_LOCATION}" --format="value(config.accelerators[0].acceleratorCount)" 2>/dev/null || true)"
  if [[ -n "${detected_accel}" ]]; then
    export GPU_ACCELERATOR_TYPE="${detected_accel}"
    case "${detected_accel}" in
      nvidia-b200)
        export GPU_TENSOR_PARALLEL_SIZE="2"
        export GPU_REPLICAS="4"
        ;;
      nvidia-gb200)
        # Oct 9 (option B): 4 x TP=1 on one a4x-highgpu-4g. Gemma 4 26B-A4B FP8 (~27 GB) fits one GB200 easily, and
        # 4 replicas make round-robin miss the session prefix on 2 of 3 turns (vs 1 of 3 with 2 replicas), which is
        # what makes the KV-aware router visible in throughput as well as TTFT.
        export GPU_TENSOR_PARALLEL_SIZE="${GPU_TENSOR_PARALLEL_SIZE_GB200:-1}"
        export GPU_REPLICAS="${GPU_REPLICAS_GB200:-4}"
        ;;
      nvidia-h100-80gb)
        export GPU_TENSOR_PARALLEL_SIZE="${detected_count:-1}"
        ;;
      nvidia-rtx-pro-6000)
        export GPU_TENSOR_PARALLEL_SIZE="${detected_count:-2}"
        ;;
    esac
  fi
fi

log "Deploying GPU One-Shot Model Server Pool (${MODEL_ID} on ${GPU_REPLICAS} x ${GPU_ACCELERATOR_TYPE} [TP=${GPU_TENSOR_PARALLEL_SIZE}] in ${GPU_NAMESPACE})"
GPU_TEMPLATE="${HARNESS_ROOT}/manifests/modelserver/gpu-vllm/gpu-pool.yaml.tmpl"
GPU_ENVSUBST='${GPU_REPLICAS} ${GPU_SERVING_IMAGE} ${MODEL_ID} ${SERVED_MODEL_NAME} ${GPU_TENSOR_PARALLEL_SIZE} ${GPU_ACCELERATOR_TYPE}'
# KV cache tiering on the GPU pool (HBM -> host DRAM via vLLM's OffloadingConnector). Opt-in; with
# ENABLE_KV_TIERING=0 the plain template renders exactly as before. See README "KV cache tiering".
if [[ "${ENABLE_KV_TIERING:-0}" == "1" ]]; then
  GPU_TEMPLATE="${HARNESS_ROOT}/manifests/modelserver/gpu-vllm/gpu-pool-kvtier.yaml.tmpl"
  export KV_TIER_HBM_CAP_LINE=""
  if [[ -n "${KV_TIER_HBM_CAP_BYTES:-}" ]]; then
    export KV_TIER_HBM_CAP_LINE="- \"--kv-cache-memory-bytes=${KV_TIER_HBM_CAP_BYTES}\""
  fi
  # Host DRAM only: CPUOffloadingSpec (vLLM default for OffloadingConnector).
  export KV_TRANSFER_CONFIG_JSON="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${KV_TIER_CPU_BYTES}}}"
  export KV_STORAGE_VOLUME_MOUNT_LINE="" KV_STORAGE_VOLUME_LINE=""
  if [[ "${ENABLE_KV_STORAGE_TIER:-0}" == "1" ]]; then
    # Host DRAM + shared storage: TieringOffloadingSpec with an fs secondary tier on an RWX PVC
    # (Managed Lustre). Chunks of KV_STORAGE_BLOCKS_PER_CHUNK GPU blocks keep the file count low, which
    # is what makes a network filesystem viable here. See README "Shared-storage KV tier".
    export KV_TRANSFER_CONFIG_JSON="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":${KV_TIER_CPU_BYTES},\"blocks_per_chunk\":${KV_STORAGE_BLOCKS_PER_CHUNK},\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"${KV_STORAGE_ROOT_DIR}\",\"n_read_threads\":${KV_STORAGE_IO_THREADS},\"n_write_threads\":${KV_STORAGE_IO_THREADS}}]}}"
    export KV_STORAGE_VOLUME_MOUNT_LINE="- {name: kv-storage, mountPath: ${KV_STORAGE_MOUNT_PATH}}"
    export KV_STORAGE_VOLUME_LINE="- {name: kv-storage, persistentVolumeClaim: {claimName: ${KV_STORAGE_PVC}}}"
    log "GPU KV storage tier: ENABLED (PVC ${KV_STORAGE_PVC} at ${KV_STORAGE_MOUNT_PATH}, root ${KV_STORAGE_ROOT_DIR}, ${KV_STORAGE_BLOCKS_PER_CHUNK} blocks/chunk)"
  fi
  GPU_ENVSUBST+=' ${KV_TIER_CPU_BYTES} ${KV_TIER_HBM_CAP_LINE} ${GPU_POD_MEMORY} ${GPU_DSHM_SIZE} ${KV_TRANSFER_CONFIG_JSON} ${KV_STORAGE_VOLUME_MOUNT_LINE} ${KV_STORAGE_VOLUME_LINE}'
  log "GPU KV tiering: ENABLED (host pool ${KV_TIER_CPU_BYTES} bytes/replica, HBM KV cap ${KV_TIER_HBM_CAP_BYTES:-none}, pod memory ${GPU_POD_MEMORY})"
fi
envsubst "${GPU_ENVSUBST}" \
  < "${GPU_TEMPLATE}" \
  | kubectl apply -n "${GPU_NAMESPACE}" -f -

# ---------- 6. Deploy Unified Header-Routing Gateway & Flow Objectives -------
export TPU_BACKEND_HOST="llmd-tpu-epp.${NAMESPACE}.svc.cluster.local"
export TPU_BACKEND_PORT="80"
export GPU_BACKEND_HOST="llmd-gpu-epp.${GPU_NAMESPACE}.svc.cluster.local"
export GPU_BACKEND_PORT="80"

log "Deploying Unified llm-d Benchmark Gateway (x-llm-d-workload header router) + Flow Control InferenceObjectives"
envsubst '${NAMESPACE} ${GPU_NAMESPACE} ${TPU_BACKEND_HOST} ${TPU_BACKEND_PORT} ${GPU_BACKEND_HOST} ${GPU_BACKEND_PORT}' \
  < "${HARNESS_ROOT}/manifests/routing/gateway-and-objectives.yaml.tmpl" \
  | kubectl apply -f -

log "Waiting for Unified Gateway to become Ready"
run kubectl rollout status deployment/llm-d-gateway -n "${NAMESPACE}" --timeout=180s

ok "Heterogeneous llm-d stack (TPU Multi-Turn + GPU One-Shot + Header/Flow Router) deployed in ${NAMESPACE}"
