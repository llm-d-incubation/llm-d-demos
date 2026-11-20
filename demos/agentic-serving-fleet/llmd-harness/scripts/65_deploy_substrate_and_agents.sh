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

# 65_deploy_substrate_and_agents.sh — Stage the Hermes Agent Runner ConfigMap
# and (if GKE Agent Substrate CRDs are present on the cluster) deploy the
# Substrate WorkerPool (`hermes-sandbox-workerpool`) and ActorTemplate
# (`hermes-agent-actor`).

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

need_bin kubectl envsubst

get_credentials

log "Staging Hermes Agent runner script (agents/hermes_swarm.py) into ConfigMap/hermes-swarm-script"
kubectl create configmap hermes-swarm-script \
  --from-file=hermes_swarm.py="${HARNESS_ROOT}/agents/hermes_swarm.py" \
  --namespace "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -
ok "ConfigMap hermes-swarm-script updated"

if ! kubectl api-resources | grep -q "substrate.agent-substrate.io"; then
  log "Installing GKE Agent Substrate CRDs (workerpools & actortemplates.substrate.agent-substrate.io)"
  run kubectl apply -f "${HARNESS_ROOT}/manifests/substrate/crds.yaml"
fi

log "Deploying GKE Agent Substrate WorkerPool + ActorTemplate (hermes-agent-actor)"
envsubst '${NAMESPACE} ${SERVED_MODEL_NAME} ${TURNS_PER_SESSION} ${MULTI_TURN_RATIO}' \
  < "${HARNESS_ROOT}/manifests/substrate/workerpool-and-actortemplate.yaml.tmpl" \
  | kubectl apply -n "${NAMESPACE}" -f -
ok "GKE Agent Substrate WorkerPool & ActorTemplate deployed"

