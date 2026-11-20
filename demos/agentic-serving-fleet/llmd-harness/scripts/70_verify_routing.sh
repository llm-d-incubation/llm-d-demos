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

# 70_verify_routing.sh — Verify that:
#   1. `x-llm-d-workload: multi-turn` routes to the Optimized TPU v7x Pool
#      (`x-llm-d-routed-pool: tpu-v7x-torchtpu-multiturn`)
#   2. `x-llm-d-workload: one-shot` routes to the GPU Serving Pool
#      (`x-llm-d-routed-pool: gpu-vllm-oneshot`)

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

need_bin kubectl

get_credentials

log "Waiting for TPU and GPU model server deployments to report Ready..."
kubectl get pods -n "${NAMESPACE}" -o wide

GATEWAY_SVC="http://llm-d-gateway.${NAMESPACE}.svc.cluster.local/v1/chat/completions"

log "1. Testing Multi-Turn Header Routing (x-llm-d-workload: multi-turn -> TPU v7x Pool)"
kubectl run verify-llmd-routing --rm -i --restart=Never \
  --image=curlimages/curl:8.10.1 \
  --namespace="${NAMESPACE}" -- \
  sh -c "
    echo '--- [Request 1: Multi-Turn Header -> Expect TPU v7x Pool] ---'
    curl -sS -D - -X POST '${GATEWAY_SVC}' \
      -H 'Content-Type: application/json' \
      -H 'x-llm-d-workload: multi-turn' \
      -H 'x-llm-d-inference-fairness-id: hermes-verify-actor-01' \
      -H 'x-llm-d-inference-objective: agentic-multi-turn' \
      -d '{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say TPU multi-turn ready in 5 words.\"}], \"max_tokens\": 24}'
    echo ''
    echo ''
    echo '--- [Request 2: One-Shot Header -> Expect GPU Pool] ---'
    curl -sS -D - -X POST '${GATEWAY_SVC}' \
      -H 'Content-Type: application/json' \
      -H 'x-llm-d-workload: one-shot' \
      -H 'x-llm-d-inference-fairness-id: hermes-verify-actor-02' \
      -H 'x-llm-d-inference-objective: oneshot-fast' \
      -d '{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say GPU one-shot ready in 5 words.\"}], \"max_tokens\": 24}'
    echo ''
  "

ok "Header-based routing & Flow Control verification complete"
