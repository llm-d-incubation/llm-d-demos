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

# 62_deploy_kv_storage_replicas.sh — Deploy the scale-out replicas behind the
# dashboard's "Tier 3 · Lustre" panel: KV_STORAGE_REPLICAS copies of the GPU
# model server (llmd-gpu-storage in ${GPU_NAMESPACE}) with the host-DRAM
# tier and the shared-storage tier on, outside the InferencePool. The dashboard
# drives a small ring of sessions across them; every returning turn lands on
# the other replica and is served from the KV it finds on Lustre.
#
# Usage:
#   scripts/62_deploy_kv_storage_replicas.sh            # deploy / update
#   scripts/62_deploy_kv_storage_replicas.sh --down     # scale to 0 (keeps the Deployment and Services)
#
# Needs the RWX PVC ${KV_STORAGE_PVC} Bound in ${GPU_NAMESPACE} (manifests/storage/README.md)
# and free GPUs: one per replica, on the same node pool as the GPU pool.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_bin kubectl envsubst
get_credentials

export GPU_NAMESPACE="${GPU_NAMESPACE:-${NAMESPACE}-gpu}"

if [[ "${1:-}" == "--down" ]]; then
  run kubectl -n "${GPU_NAMESPACE}" scale deploy/llmd-gpu-storage --replicas=0
  ok "llmd-gpu-storage scaled to 0 in ${GPU_NAMESPACE}"
  exit 0
fi

phase="$(kubectl get pvc "${KV_STORAGE_PVC}" -n "${GPU_NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
[[ "${phase}" == "Bound" ]] || die "PVC ${KV_STORAGE_PVC} in ${GPU_NAMESPACE} is not Bound (got '${phase:-missing}'); see manifests/storage/README.md"

if nodepool_exists "${GPU_NODEPOOL}"; then
  detected_accel="$(gc container node-pools describe "${GPU_NODEPOOL}" --cluster="${CLUSTER_NAME}" --location="${CLUSTER_LOCATION}" --format="value(config.accelerators[0].acceleratorType)" 2>/dev/null || true)"
  [[ -n "${detected_accel}" ]] && export GPU_ACCELERATOR_TYPE="${detected_accel}"
fi

export KV_STORAGE_TRANSFER_CONFIG_JSON="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":${KV_STORAGE_REPLICA_CPU_BYTES},\"blocks_per_chunk\":${KV_STORAGE_BLOCKS_PER_CHUNK},\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"${KV_STORAGE_REPLICA_ROOT_DIR}\",\"n_read_threads\":${KV_STORAGE_IO_THREADS},\"n_write_threads\":${KV_STORAGE_IO_THREADS}}]}}"

log "Deploying ${KV_STORAGE_REPLICAS} scale-out replica(s) (${MODEL_ID} on ${GPU_ACCELERATOR_TYPE}, HBM cap ${KV_STORAGE_REPLICA_HBM_CAP_BYTES}, host ${KV_STORAGE_REPLICA_CPU_BYTES}, storage ${KV_STORAGE_REPLICA_ROOT_DIR}) in ${GPU_NAMESPACE}"
envsubst '${KV_STORAGE_REPLICAS} ${GPU_SERVING_IMAGE} ${MODEL_ID} ${SERVED_MODEL_NAME} ${GPU_ACCELERATOR_TYPE} ${KV_STORAGE_REPLICA_HBM_CAP_BYTES} ${KV_STORAGE_REPLICA_MEMORY} ${KV_STORAGE_REPLICA_DSHM_SIZE} ${KV_STORAGE_TRANSFER_CONFIG_JSON} ${KV_STORAGE_MOUNT_PATH} ${KV_STORAGE_PVC}' \
  < "${HARNESS_ROOT}/manifests/modelserver/gpu-vllm/gpu-storage-replicas.yaml.tmpl" \
  | kubectl apply -n "${GPU_NAMESPACE}" -f -

run kubectl -n "${GPU_NAMESPACE}" rollout status deploy/llmd-gpu-storage --timeout=900s
ok "llmd-gpu-storage ready in ${GPU_NAMESPACE}; point the dashboard at it with scripts/86_deploy_dashboard_image.sh (AB_KV_STORAGE_URLS / AB_KV_STORAGE_HOSTS are set from it when ENABLE_KV_STORAGE_REPLICAS=1)"
