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

# 90_teardown.sh — Clean up the benchmark demo namespace (`${NAMESPACE}`) and
# optionally scale down or delete node pools to free TPU reservation chips.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

need_bin gcloud kubectl helm

get_credentials

export GPU_NAMESPACE="${GPU_NAMESPACE:-${NAMESPACE}-gpu}"

log "Deleting demo namespaces ${NAMESPACE} and ${GPU_NAMESPACE} (removes TPU/GPU deployments, EPPs, Dashboard, and Gateway)"
run kubectl delete namespace "${NAMESPACE}" "${GPU_NAMESPACE}" --ignore-not-found
run kubectl delete deployment llm-d-gateway -n default --ignore-not-found || true

if [[ "${DELETE_NODEPOOLS:-0}" == "1" ]]; then
  for pool in "${TPU_NODEPOOL}" "${GPU_NODEPOOL}" "${SUBSTRATE_NODEPOOL}"; do
    if nodepool_exists "${pool}"; then
      log "Deleting node pool ${pool} in ${CLUSTER_NAME}"
      run gc container node-pools delete "${pool}" \
        --cluster="${CLUSTER_NAME}" \
        --location="${CLUSTER_LOCATION}" \
        --quiet || true
    fi
  done
elif [[ "${SCALE_TO_ZERO:-1}" == "1" ]]; then
  for pool in "${TPU_NODEPOOL}" "${GPU_NODEPOOL}" "${SUBSTRATE_NODEPOOL}"; do
    if nodepool_exists "${pool}"; then
      existing_nodes="$(kubectl get nodes -l "cloud.google.com/gke-nodepool=${pool}" --no-headers 2>/dev/null | wc -l | tr -d ' ' || echo 0)"
      if (( existing_nodes > 0 )); then
        log "Scaling node pool ${pool} down to 0 nodes (currently ${existing_nodes})"
        run gc container clusters resize "${CLUSTER_NAME}" \
          --node-pool="${pool}" \
          --location="${CLUSTER_LOCATION}" \
          --num-nodes=0 \
          --quiet || true
      else
        ok "Node pool ${pool} is already at 0 nodes"
      fi
    fi
  done
  ok "All demo node pools (${TPU_NODEPOOL}, ${GPU_NODEPOOL}, ${SUBSTRATE_NODEPOOL}) scaled to 0; $(reservation_free_chips) TPU v7x chips now free in reservation ${RESERVATION_NAME}"
else
  dim "Node pools left at current size (SCALE_TO_ZERO=0)."
fi

ok "Teardown complete"
