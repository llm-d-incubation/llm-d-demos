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

# 00_preflight.sh — Validate tools, cluster access, TPU reservation, and image availability.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

log "Checking required CLI tools"
need_bin gcloud kubectl helm python3 envsubst
ok "all required binaries present"

if [[ -z "${HF_TOKEN:-}" ]]; then
  warn "HF_TOKEN is not exported; google/gemma-4-26B-A4B-it requires a valid HuggingFace token with Gemma license accepted."
else
  ok "HF_TOKEN is set"
fi

log "Checking GKE cluster ${CLUSTER_NAME} in ${CLUSTER_LOCATION}"
if cluster_exists; then
  get_credentials
  ok "cluster ${CLUSTER_NAME} is reachable"
  kubectl get nodes -o wide || true
else
  warn "cluster ${CLUSTER_NAME} does not exist yet; 50_prepare_cluster.sh will create it"
fi

log "Checking TPU v7x reservation (${RESERVATION_NAME})"
free_chips="$(reservation_free_chips)"
in_cluster="$(cluster_existing_tpu_chips)"
ok "reservation free chips: ${free_chips} | already in cluster: ${in_cluster}"

log "Checking upstream llm-d checkout (LLMD_REPO=${LLMD_REPO})"
if [[ -f "${LLMD_REPO}/guides/env.sh" ]]; then
  ok "upstream llm-d guides found at ${LLMD_REPO}"
else
  warn "LLMD_REPO (${LLMD_REPO}) does not contain guides/env.sh; clone https://github.com/llm-d/llm-d.git before running 60_deploy_llmd_heterogeneous.sh"
fi

log "Checking TPU serving image (${TPU_SERVING_IMAGE})"
ok "preflight check complete"
