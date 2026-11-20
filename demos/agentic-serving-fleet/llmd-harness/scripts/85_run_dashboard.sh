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

# 85_run_dashboard.sh — Launch the live llm-d Control Center UI
# connected to `llm-d-gateway` on `llmd-demo-cluster`.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

need_bin kubectl python3

get_credentials

DASHBOARD_PORT="${DASHBOARD_PORT:-8085}"
GATEWAY_LOCAL_PORT="${GATEWAY_LOCAL_PORT:-18080}"
TPU_POD1_LOCAL_PORT="${TPU_POD1_LOCAL_PORT:-18081}"
TPU_POD2_LOCAL_PORT="${TPU_POD2_LOCAL_PORT:-18082}"
GPU_POD1_LOCAL_PORT="${GPU_POD1_LOCAL_PORT:-18083}"
GPU_NAMESPACE="${GPU_NAMESPACE:-${NAMESPACE}-gpu}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    pkill -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

pf_loop() {
  local ns="$1" target="$2" ports="$3" logfile="$4"
  while true; do
    kubectl port-forward -n "${ns}" "${target}" "${ports}" >>"${logfile}" 2>&1 || true
    sleep 1
  done
}

log "Starting kubectl port-forward to service/llm-d-gateway (${NAMESPACE}) on localhost:${GATEWAY_LOCAL_PORT}"
pf_loop "${NAMESPACE}" "svc/llm-d-gateway" "${GATEWAY_LOCAL_PORT}:80" "/tmp/llmd-gw-pf.log" &
PIDS+=($!)

read -r -a TPU_PODS <<< "$(kubectl get pods -n "${NAMESPACE}" -l app=llmd-tpu-decode --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)"
if [[ -n "${TPU_PODS[0]:-}" ]]; then
  log "Port-forwarding TPU Pod 1 (${TPU_PODS[0]}) /metrics to localhost:${TPU_POD1_LOCAL_PORT}"
  pf_loop "${NAMESPACE}" "pod/${TPU_PODS[0]}" "${TPU_POD1_LOCAL_PORT}:8000" "/tmp/llmd-tpu1-pf.log" &
  PIDS+=($!)
  export TPU_POD1_HOST="127.0.0.1:${TPU_POD1_LOCAL_PORT}"
fi
if [[ -n "${TPU_PODS[1]:-}" ]]; then
  log "Port-forwarding TPU Pod 2 (${TPU_PODS[1]}) /metrics to localhost:${TPU_POD2_LOCAL_PORT}"
  pf_loop "${NAMESPACE}" "pod/${TPU_PODS[1]}" "${TPU_POD2_LOCAL_PORT}:8000" "/tmp/llmd-tpu2-pf.log" &
  PIDS+=($!)
  export TPU_POD2_HOST="127.0.0.1:${TPU_POD2_LOCAL_PORT}"
fi

GPU_POD="$(kubectl get pods -n "${GPU_NAMESPACE}" -l app=llmd-gpu-decode --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -n "${GPU_POD}" ]]; then
  log "Port-forwarding GPU Pod (${GPU_POD}) /metrics to localhost:${GPU_POD1_LOCAL_PORT}"
  pf_loop "${GPU_NAMESPACE}" "pod/${GPU_POD}" "${GPU_POD1_LOCAL_PORT}:8000" "/tmp/llmd-gpu1-pf.log" &
  PIDS+=($!)
  export GPU_POD1_HOST="127.0.0.1:${GPU_POD1_LOCAL_PORT}"
fi

sleep 2

# "Without llm-d" baseline for the live A/B: client round-robin straight at the
# TPU pods (bypassing the gateway/EPP). Override with BASELINE_URLS=a,b.
if [[ -z "${BASELINE_URLS:-}" ]]; then
  _bl=()
  [[ -n "${TPU_POD1_HOST:-}" ]] && _bl+=("http://${TPU_POD1_HOST}")
  [[ -n "${TPU_POD2_HOST:-}" ]] && _bl+=("http://${TPU_POD2_HOST}")
  BASELINE_URLS="$(IFS=,; echo "${_bl[*]:-}")"
fi
export BASELINE_URLS
log "Live A/B baseline targets: ${BASELINE_URLS:-<none>} (concurrency ${AB_CONCURRENCY:-12})"

DASHBOARD_DIR="${DASHBOARD_DIR:-$(cd "${HARNESS_ROOT}/.." && pwd)/dashboard}"
if [[ ! -d "${DASHBOARD_DIR}" ]]; then
  DASHBOARD_DIR="${HARNESS_ROOT}/dashboard"
fi
log "Launching llm-d Live Control Center on http://localhost:${DASHBOARD_PORT}"
python3 "${DASHBOARD_DIR}/server.py" \
  --port="${DASHBOARD_PORT}" \
  --gateway-url="http://127.0.0.1:${GATEWAY_LOCAL_PORT}" \
  --model="${SERVED_MODEL_NAME}"



