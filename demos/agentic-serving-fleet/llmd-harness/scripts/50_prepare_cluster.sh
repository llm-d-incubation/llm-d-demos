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

# 50_prepare_cluster.sh — Ensure existing cluster (${CLUSTER_NAME}) has:
#   1. Non-E2 CPU worker pool for GKE Agent Substrate & Hermes agents (${SUBSTRATE_NODEPOOL})
#   2. TPU v7x node pool (${TPU_NODEPOOL}) for multi-turn agentic serving
#   3. GPU node pool (${GPU_NODEPOOL}) for one-shot serving
#
# Idempotent: skips any node pool that already exists with sufficient capacity.

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

need_bin gcloud kubectl

if ! cluster_exists; then
  log "Cluster ${CLUSTER_NAME} not found; creating GKE Standard cluster"
  run gc container clusters create "${CLUSTER_NAME}" \
    --location="${CLUSTER_LOCATION}" \
    --node-locations="${ZONE}" \
    --release-channel="regular" \
    --machine-type="${SUBSTRATE_MACHINE_TYPE}" \
    --num-nodes="${SUBSTRATE_NUM_NODES}" \
    --enable-dataplane-v2 \
    --enable-ip-alias \
    --workload-pool="${PROJECT_ID}.svc.id.goog" \
    --addons=GcsFuseCsiDriver \
    --enable-image-streaming \
    --quiet
else
  ok "Reusing existing cluster ${CLUSTER_NAME}"
fi

get_credentials

# ---------- 1. GKE Agent Substrate / Hermes CPU Node Pool --------------------
existing_cpu="$(kubectl get nodes --no-headers 2>/dev/null | grep -E 'cpu-pool|substrate' | wc -l | tr -d ' ' || echo 0)"
if (( existing_cpu >= 2 )); then
  ok "Cluster already has ${existing_cpu} Ready CPU worker nodes (cpu-pool / ${SUBSTRATE_NODEPOOL})"
elif nodepool_exists "${SUBSTRATE_NODEPOOL}"; then
  existing_sub="$(kubectl get nodes -l "cloud.google.com/gke-nodepool=${SUBSTRATE_NODEPOOL}" --no-headers 2>/dev/null | wc -l | tr -d ' ' || echo 0)"
  if (( existing_sub >= SUBSTRATE_NUM_NODES )); then
    ok "Substrate/Hermes CPU node pool ${SUBSTRATE_NODEPOOL} already has ${existing_sub} nodes ready"
  else
    log "Scaling up existing Substrate/Hermes CPU node pool ${SUBSTRATE_NODEPOOL} from ${existing_sub} to ${SUBSTRATE_NUM_NODES} nodes"
    run gc container clusters resize "${CLUSTER_NAME}" \
      --node-pool="${SUBSTRATE_NODEPOOL}" \
      --location="${CLUSTER_LOCATION}" \
      --num-nodes="${SUBSTRATE_NUM_NODES}" \
      --quiet
  fi
else
  log "Creating Substrate/Hermes CPU node pool ${SUBSTRATE_NODEPOOL} (${SUBSTRATE_NUM_NODES} x ${SUBSTRATE_MACHINE_TYPE})"
  run gc container node-pools create "${SUBSTRATE_NODEPOOL}" \
    --cluster="${CLUSTER_NAME}" \
    --location="${CLUSTER_LOCATION}" \
    --node-locations="${ZONE}" \
    --machine-type="${SUBSTRATE_MACHINE_TYPE}" \
    --num-nodes="${SUBSTRATE_NUM_NODES}" \
    --node-labels="llm-d.ai/role=agent-substrate" \
    --quiet || warn "Could not create ${SUBSTRATE_NODEPOOL}; falling back to existing CPU nodes"
fi

# ---------- 2. TPU v7x Node Pool (Multi-Turn Agentic Stack) ------------------
CHIPS_PER_SLICE="$(topology_chips)"
TOTAL_TPU_CHIPS=$(( CHIPS_PER_SLICE * TPU_NUM_SLICES ))
# Count only nodes in the reserved pool; a legacy spot pool shares the same TPU labels.
reserved_tpu_nodes="$(kubectl get nodes -l "cloud.google.com/gke-nodepool=${TPU_NODEPOOL}" --no-headers 2>/dev/null | wc -l | tr -d ' ' || echo 0)"
already_tpu=$(( reserved_tpu_nodes * TPU_CHIPS_PER_NODE ))

if (( already_tpu >= TOTAL_TPU_CHIPS )); then
  ok "Reserved TPU pool ${TPU_NODEPOOL} already has ${already_tpu} TPU v7x chips (>= ${TOTAL_TPU_CHIPS} requested)"
elif nodepool_exists "${TPU_NODEPOOL}"; then
  log "Resizing existing TPU node pool ${TPU_NODEPOOL} to ${TPU_NUM_SLICES} nodes (${TOTAL_TPU_CHIPS} chips on ${RESERVATION_NAME})"
  run gc container clusters resize "${CLUSTER_NAME}" \
    --node-pool="${TPU_NODEPOOL}" \
    --location="${CLUSTER_LOCATION}" \
    --num-nodes="${TPU_NUM_SLICES}" \
    --quiet
else
  log "Creating TPU v7x node pool ${TPU_NODEPOOL} (${TPU_NUM_SLICES} x ${TPU_MACHINE_TYPE} on ${RESERVATION_NAME})"
  run gc container node-pools create "${TPU_NODEPOOL}" \
    --cluster="${CLUSTER_NAME}" \
    --location="${CLUSTER_LOCATION}" \
    --node-locations="${ZONE}" \
    --machine-type="${TPU_MACHINE_TYPE}" \
    --num-nodes="${TPU_NUM_SLICES}" \
    --reservation-affinity=specific \
    --reservation="${RESERVATION_PATH}" \
    --disk-type=hyperdisk-balanced \
    --disk-size=300 \
    --quiet
fi

# Retire any legacy Spot TPU pool once reserved capacity exists (never demo on preemptible chips).
if nodepool_exists "${TPU_NODEPOOL}-spot" && nodepool_exists "${TPU_NODEPOOL}"; then
  log "Deleting legacy Spot TPU node pool ${TPU_NODEPOOL}-spot (reserved pool ${TPU_NODEPOOL} is in place)"
  run gc container node-pools delete "${TPU_NODEPOOL}-spot" \
    --cluster="${CLUSTER_NAME}" --location="${CLUSTER_LOCATION}" --quiet \
    || warn "Could not delete ${TPU_NODEPOOL}-spot; remove it manually"
fi

# ---------- 3. GPU Node Pool (One-Shot Stack: B200 -> H100 -> G4) ------------
provision_gpu_pool_tier() {
  local tier="$1"
  local mtype="" accel="" acount="" zones="" dtype="hyperdisk-balanced"
  case "${tier}" in
    b200)
      mtype="a4-highgpu-8g"
      accel="nvidia-b200"
      acount="8"
      zones="${REGION}-b"
      dtype="hyperdisk-balanced"
      ;;
    h100)
      mtype="a3-highgpu-1g"
      accel="nvidia-h100-80gb"
      acount="1"
      zones="${REGION}-c,${REGION}-b,${REGION}-a"
      dtype="pd-balanced"
      ;;
    g4)
      mtype="g4-standard-96"
      accel="nvidia-rtx-pro-6000"
      acount="2"
      # Prefer the TPU zone (same-zone as Ironwood pool); fall back region-wide.
      zones="${GPU_ZONES:-${ZONE}}"
      dtype="hyperdisk-balanced"
      ;;
    *)
      return 1
      ;;
  esac

  local capacity_flags=()
  if [[ "${GPU_USE_SPOT:-0}" == "1" ]]; then
    capacity_flags+=(--spot)
  fi
  if [[ -n "${GPU_RESERVATION_PATH:-}" ]]; then
    capacity_flags+=(--reservation-affinity=specific --reservation="${GPU_RESERVATION_PATH}")
  fi

  local capacity_desc="on-demand"
  [[ "${GPU_USE_SPOT:-0}" == "1" ]] && capacity_desc="spot"
  [[ -n "${GPU_RESERVATION_PATH:-}" ]] && capacity_desc="reservation ${GPU_RESERVATION_PATH}"

  log "Provisioning GPU node pool ${GPU_NODEPOOL} using tier '${tier}' (${GPU_NUM_NODES} x ${mtype}, ${acount}x ${accel} in ${zones}, ${capacity_desc})"
  if gc container node-pools create "${GPU_NODEPOOL}" \
    --cluster="${CLUSTER_NAME}" \
    --location="${CLUSTER_LOCATION}" \
    --node-locations="${zones}" \
    --machine-type="${mtype}" \
    --accelerator="type=${accel},count=${acount},gpu-driver-version=latest" \
    "${capacity_flags[@]}" \
    --num-nodes="${GPU_NUM_NODES}" \
    --disk-type="${dtype}" \
    --disk-size=200 \
    --quiet; then
    export GPU_MACHINE_TYPE="${mtype}"
    export GPU_ACCELERATOR_TYPE="${accel}"
    export GPU_COUNT_PER_NODE="${acount}"
    ok "Created ${GPU_NODEPOOL} on tier '${tier}' (${mtype} · ${acount}x ${accel} · ${GPU_NUM_NODES} node(s) · ${capacity_desc})"
    return 0
  fi
  # G4: retry region-wide if the preferred zone had no on-demand capacity.
  if [[ "${tier}" == "g4" && "${zones}" == "${ZONE}" ]]; then
    warn "No ${mtype} capacity in ${ZONE}; retrying across ${REGION}-b,${REGION}-c,${REGION}-f"
    GPU_ZONES="${REGION}-b,${REGION}-c,${REGION}-f" provision_gpu_pool_tier g4 && return 0
  fi
  warn "Tier '${tier}' (${mtype} / ${accel}) unavailable or quota exceeded; trying next tier in B200 -> H100 -> G4..."
  return 1
}

if nodepool_exists "${GPU_NODEPOOL}"; then
  cur_accel="$(gc container node-pools describe "${GPU_NODEPOOL}" --cluster="${CLUSTER_NAME}" --location="${CLUSTER_LOCATION}" --format="value(config.accelerators[0].acceleratorType)" 2>/dev/null || true)"
  cur_spot="$(gc container node-pools describe "${GPU_NODEPOOL}" --cluster="${CLUSTER_NAME}" --location="${CLUSTER_LOCATION}" --format="value(config.spot)" 2>/dev/null || true)"
  if [[ "${cur_accel}" == "nvidia-l4" ]] || [[ "${GPU_TIER_PREFERENCE:-g4}" == "g4" && "${cur_accel}" != "nvidia-rtx-pro-6000" ]]; then
    log "Replacing GPU node pool ${GPU_NODEPOOL} (${cur_accel}) with preferred tier (${GPU_TIER_PREFERENCE:-g4})..."
    run gc container node-pools delete "${GPU_NODEPOOL}" --cluster="${CLUSTER_NAME}" --location="${CLUSTER_LOCATION}" --quiet
  elif [[ "${cur_spot}" == "True" && "${GPU_USE_SPOT:-0}" != "1" ]]; then
    log "Replacing Spot GPU node pool ${GPU_NODEPOOL} (${cur_accel}) with on-demand capacity..."
    run gc container node-pools delete "${GPU_NODEPOOL}" --cluster="${CLUSTER_NAME}" --location="${CLUSTER_LOCATION}" --quiet
  else
    ok "GPU node pool ${GPU_NODEPOOL} (${cur_accel}, spot=${cur_spot:-False}) already configured"
  fi
fi

if ! nodepool_exists "${GPU_NODEPOOL}"; then
  IFS=',' read -r -a gpu_tiers <<< "${GPU_TIER_PREFERENCE:-b200,h100,g4}"
  for tier in "${gpu_tiers[@]}"; do
    if provision_gpu_pool_tier "${tier}"; then
      break
    fi
  done
fi

log "Current cluster node summary:"
kubectl get nodes -L cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-accelerator,llm-d.ai/role
ok "Heterogeneous cluster capacity (TPU + GPU [B200 -> H100 -> G4] + Agent Substrate CPU) ready"
