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

# 80_run_benchmark.sh — Spawn 1,000s of Hermes Agents (multi-turn + one-shot)
# inside the cluster against `llm-d-gateway` and collect per-pool
# (TPU Multi-Turn vs. GPU One-Shot) latency, QPS, and prefix-cache telemetry.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

need_bin kubectl

get_credentials
mkdir -p "${RESULTS_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
CACHE_TAG="cached"
EXTRA_FLAG=""
if [[ "${NO_PREFIX_CACHE:-0}" == "1" ]]; then
  CACHE_TAG="nocache"
  EXTRA_FLAG=", \"--no-prefix-cache\""
fi
POD_NAME="hermes-agent-swarm-${CACHE_TAG}-${TS}"
OUT_JSON="${RESULTS_DIR}/hermes-${NUM_HERMES_AGENTS}agents-${CACHE_TAG}-${TS}.json"

kubectl create configmap hermes-swarm-script \
  --from-file=hermes_swarm.py="${HARNESS_ROOT}/agents/hermes_swarm.py" \
  --namespace "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

TPU_POD_IPS="$(kubectl get pods -n "${NAMESPACE}" -l app=llmd-tpu-decode -o jsonpath='{range .items[*]}{.status.podIP}{" "}{end}' 2>/dev/null || true)"
METRICS_URLS=""
for ip in ${TPU_POD_IPS}; do
  if [[ -n "${ip}" ]]; then
    METRICS_URLS="${METRICS_URLS:+${METRICS_URLS},}http://${ip}:8000/metrics"
  fi
done

log "Launching ${NUM_HERMES_AGENTS} Hermes Agents (${CONCURRENT_ACTIVE_AGENTS} concurrent active Actors, mode=${CACHE_TAG})"
dim "Multi-Turn Ratio: ${MULTI_TURN_RATIO} (${TURNS_PER_SESSION} turns/session -> TPU v7x Pool)"
dim "One-Shot Ratio  : $(awk "BEGIN {printf \"%.2f\", 1.0 - ${MULTI_TURN_RATIO}}") (1 turn -> GPU Pool)"
dim "Prometheus URLs : ${METRICS_URLS:-none}"

kubectl delete pod "${POD_NAME}" -n "${NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true

kubectl run "${POD_NAME}" \
  --restart=Never \
  --image=python:3.12-slim \
  --namespace="${NAMESPACE}" \
  --overrides="{
    \"spec\": {
      \"containers\": [{
        \"name\": \"hermes-swarm\",
        \"image\": \"python:3.12-slim\",
        \"command\": [
          \"python3\", \"/opt/hermes/hermes_swarm.py\",
          \"--endpoint=http://llm-d-gateway.${NAMESPACE}.svc.cluster.local/v1/chat/completions\",
          \"--model=${SERVED_MODEL_NAME}\",
          \"--num-agents=${NUM_HERMES_AGENTS}\",
          \"--concurrency=${CONCURRENT_ACTIVE_AGENTS}\",
          \"--multi-turn-ratio=${MULTI_TURN_RATIO}\",
          \"--turns=${TURNS_PER_SESSION}\",
          \"--cohorts=${SHARED_WORKSPACE_COHORTS}\",
          \"--prefix-tokens=${FROZEN_PREFIX_TOKENS}\",
          \"--metrics-endpoints=${METRICS_URLS}\",
          \"--output-json=/tmp/summary.json\"${EXTRA_FLAG}
        ],
        \"volumeMounts\": [{
          \"name\": \"hermes-runner\",
          \"mountPath\": \"/opt/hermes\"
        }]
      }],
      \"volumes\": [{
        \"name\": \"hermes-runner\",
        \"configMap\": {\"name\": \"hermes-swarm-script\"}
      }]
    }
  }"

log "Waiting for Hermes swarm pod ${POD_NAME} to start..."
kubectl wait --for=condition=Ready "pod/${POD_NAME}" -n "${NAMESPACE}" --timeout=120s || true

log "Streaming live benchmark telemetry (TPU Multi-Turn vs. GPU One-Shot):"
kubectl logs -f "${POD_NAME}" -n "${NAMESPACE}" | tee "/tmp/${POD_NAME}.log" || true

awk '/^================ FINAL BENCHMARK TELEMETRY ================/{flag=1;next}/^Saved structured summary/{flag=0}flag' "/tmp/${POD_NAME}.log" > "${OUT_JSON}" || true
kubectl delete pod "${POD_NAME}" -n "${NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true

if [[ -s "${OUT_JSON}" ]]; then
  ok "Saved cluster benchmark telemetry to ${OUT_JSON}"
fi
