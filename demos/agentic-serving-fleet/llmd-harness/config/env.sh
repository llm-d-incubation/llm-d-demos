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

# =============================================================================
# llm-d Benchmark & Deployment Harness — Central Configuration
#
# Scenario:
#   - 1,000s of Hermes agents running on GKE Agent Substrate
#     and/or K8s async swarm
#   - Each agent sends either multi-turn or one-shot requests with custom
#     headers (x-llm-d-workload, x-llm-d-inference-fairness-id,
#     x-llm-d-inference-objective) to a single llm-d Gateway endpoint
#   - llm-d Flow + Header Routing splits traffic between:
#       1. TPU v7x Ironwood Serving Stack (multi-turn, prefix-cache-aware
#          + Priority Flow Control)
#       2. NVIDIA G4 GPU Serving Stack (one-shot, low-latency dispatch)
#   - Model: google/gemma-4-26B-A4B-it (26B total / 4B active MoE)
# =============================================================================

# ---------- Optional Local Overrides (.env.local, git-ignored) ---------------
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_ENV_DIR}/.env.local" ]]; then
  # shellcheck disable=SC1091
  source "${_ENV_DIR}/.env.local"
fi

# ---------- Google Cloud & Cluster -------------------------------------------
export PROJECT_ID="${PROJECT_ID:-your-gcp-project-id}"
export PROJECT_NUMBER="${PROJECT_NUMBER:-}"
export REGION="${REGION:-us-central1}"
export ZONE="${ZONE:-us-central1-c}"

# Regional or zonal cluster location (defaults to REGION).
export CLUSTER_NAME="${CLUSTER_NAME:-llmd-demo-cluster}"
export CLUSTER_LOCATION="${CLUSTER_LOCATION:-${REGION}}"
export NAMESPACE="${NAMESPACE:-llm-d-serving}"
export GUIDE_NAME="${GUIDE_NAME:-agentic-flow}"

# NOTE: GKE `--reservation` does not accept the zonal Compute URI form
# (projects/.../zones/.../reservations/...). Use projects/{project}/reservations/{name};
# the zone is implied by --node-locations.
export RESERVATION_NAME="${RESERVATION_NAME:-}"
export RESERVATION_PATH="${RESERVATION_PATH:+${RESERVATION_PATH}}${RESERVATION_NAME:+projects/${PROJECT_ID}/reservations/${RESERVATION_NAME}}"

# ---------- Paths & Upstream Repos -------------------------------------------
HARNESS_ROOT="${HARNESS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HARNESS_ROOT
# Local checkout of upstream https://github.com/llm-d/llm-d (used for guides/env.sh & router values)
export LLMD_REPO="${LLMD_REPO:-${HOME}/llm-d}"
export LLMD_REF="${LLMD_REF:-main}"
export RESULTS_DIR="${RESULTS_DIR:-${HARNESS_ROOT}/results}"

# ---------- Model Configuration (Gemma 4 26B-A4B MoE FP8-Dynamic) ------------
export MODEL_ID="${MODEL_ID:-RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export HF_TOKEN="${HF_TOKEN:-}"

# ---------- 1. TPU v7x Ironwood Stack (Multi-Turn Agentic Pool) --------------
export TPU_BACKEND="${TPU_BACKEND:-torchtpu}"
export AR_REPO="${AR_REPO:-llmd-tpu}"
export AR_HOST="${AR_HOST:-${REGION}-docker.pkg.dev}"
export AR_BASE="${AR_BASE:-${AR_HOST}/${PROJECT_ID}/${AR_REPO}}"
export IMAGE_TAG="${IMAGE_TAG:-v0.32.0}"
export SGLANG_IMAGE_TAG="${SGLANG_IMAGE_TAG:-v0.5.22}"
export TPU_SERVING_IMAGE="${TPU_SERVING_IMAGE:-${AR_BASE}/vllm-torchtpu:${IMAGE_TAG}}"
export SGLANG_TPU_SERVING_IMAGE="${SGLANG_TPU_SERVING_IMAGE:-${AR_BASE}/sglang-torchtpu:${SGLANG_IMAGE_TAG}}"
export TORCHAX_SERVING_IMAGE="${TORCHAX_SERVING_IMAGE:-docker.io/vllm/vllm-tpu:v0.27.0}"

export TPU_MACHINE_TYPE="${TPU_MACHINE_TYPE:-tpu7x-standard-4t}"
export TPU_CHIPS_PER_NODE="${TPU_CHIPS_PER_NODE:-4}"
export TPU_TOPOLOGY="${TPU_TOPOLOGY:-2x2x1}"
export TPU_NODEPOOL="${TPU_NODEPOOL:-tpu7x-${TPU_TOPOLOGY}-pool}"
export TPU_NUM_SLICES="${TPU_NUM_SLICES:-2}"
export TPU_REPLICAS="${TPU_REPLICAS:-${TPU_NUM_SLICES}}"
export TPU_TENSOR_PARALLEL_SIZE="${TPU_TENSOR_PARALLEL_SIZE:-2}"
export TPU_DATA_PARALLEL_SIZE="${TPU_DATA_PARALLEL_SIZE:-4}"
export ENABLE_KV_OFFLOAD="${ENABLE_KV_OFFLOAD:-0}"

# ---------- 2. GPU Serving Stack (NVIDIA G4 Blackwell Server Edition) --------
# Standardized on Ironwood (TPU v7x) + G4 (g4-standard-96: 2x nvidia-rtx-pro-6000 96GB GDDR7 = 192GB VRAM)
export GPU_TIER_PREFERENCE="${GPU_TIER_PREFERENCE:-g4}"
export GPU_NODEPOOL="${GPU_NODEPOOL:-gpu-oneshot-pool}"
export GPU_MACHINE_TYPE="${GPU_MACHINE_TYPE:-g4-standard-96}"
export GPU_ACCELERATOR_TYPE="${GPU_ACCELERATOR_TYPE:-nvidia-rtx-pro-6000}"
export GPU_COUNT_PER_NODE="${GPU_COUNT_PER_NODE:-2}"
export GPU_NUM_NODES="${GPU_NUM_NODES:-1}"
export GPU_REPLICAS="${GPU_REPLICAS:-1}"
export GPU_TENSOR_PARALLEL_SIZE="${GPU_TENSOR_PARALLEL_SIZE:-2}"
export GPU_SERVING_IMAGE="${GPU_SERVING_IMAGE:-vllm/vllm-openai:latest}"
# Never run the benchmark GPU pool on Spot (preemption mid-demo). On-demand by default.
export GPU_USE_SPOT="${GPU_USE_SPOT:-0}"
# Optional: a G4 reservation (projects/{project}/reservations/{name}). Empty => on-demand.
# NOTE: no G4 reservation exists in ${PROJECT_ID} as of 2026-10-02.
export GPU_RESERVATION_PATH="${GPU_RESERVATION_PATH:-}"

# ---------- 2b. KV cache tiering (HBM -> host DRAM, GPU and TPU pools) -------
# ENABLE_KV_TIERING=1 (GPU pool) deploys gpu-pool-kvtier.yaml.tmpl instead of gpu-pool.yaml.tmpl,
# and ENABLE_KV_OFFLOAD=1 (TPU pool) enables TPUCPUOffloadingSpec on llmd-tpu-decode: vLLM's
# native OffloadingConnector keeps evicted KV blocks in a host-RAM pool and restores them on the
# next turn instead of re-prefilling. Nothing else in the stack (EPP, gateway, CRDs) changes.
# Sizing: for Gemma 4 keep KV_TIER_CPU_BYTES at least 3x the total HBM KV budget of the replica
# (all TP ranks), because the host copy of a short prefix is larger than its HBM footprint.
# KV_TIER_HBM_CAP_BYTES (per chip/GPU) is a demo-only cap so the host tier is exercised on large cards;
# leave empty for no cap. See README "KV cache tiering" for measured results and gotchas.
export ENABLE_KV_TIERING="${ENABLE_KV_TIERING:-0}"
export KV_TIER_CPU_BYTES="${KV_TIER_CPU_BYTES:-68719476736}"   # 64 GiB host pool per replica/engine
export KV_TIER_HBM_CAP_BYTES="${KV_TIER_HBM_CAP_BYTES:-}"      # e.g. 8589934592 (8 GiB per chip/GPU)
export GPU_POD_MEMORY="${GPU_POD_MEMORY:-160Gi}"               # >= host pool + weights + runtime
export GPU_DSHM_SIZE="${GPU_DSHM_SIZE:-96Gi}"                  # /dev/shm holds the host pool mmap

# ---------- 2c. Optional shared-storage KV tier (HBM -> host DRAM -> filesystem) ----------
# ENABLE_KV_STORAGE_TIER=1 (requires ENABLE_KV_TIERING=1 and/or ENABLE_KV_OFFLOAD=1) adds a
# filesystem tier on an RWX PVC (Managed Lustre) below the host pool, so a new or restarted
# replica can restore KV written by the others. Keep the storage in the cluster's region.
# TPU and GPU use separate default subdirectories under KV_STORAGE_MOUNT_PATH because TPU uses
# 256-token Pallas-layout blocks while GPU uses 64-token CUDA-layout blocks.
export ENABLE_KV_STORAGE_TIER="${ENABLE_KV_STORAGE_TIER:-0}"
export KV_STORAGE_PVC="${KV_STORAGE_PVC:-llm-d-kv-cache-storage}"   # RWX PVC in ${NAMESPACE} / ${GPU_NAMESPACE}
export KV_STORAGE_MOUNT_PATH="${KV_STORAGE_MOUNT_PATH:-/mnt/kv-storage}"
export KV_STORAGE_ROOT_DIR="${KV_STORAGE_ROOT_DIR:-${KV_STORAGE_MOUNT_PATH}/kv-cache/vllm}"
export TPU_KV_STORAGE_ROOT_DIR="${TPU_KV_STORAGE_ROOT_DIR:-${KV_STORAGE_MOUNT_PATH}/kv-cache/vllm-tpu}"
export KV_STORAGE_BLOCKS_PER_CHUNK="${KV_STORAGE_BLOCKS_PER_CHUNK:-16}"   # GPU blocks per file (1k tokens at 64-token blocks)
export KV_STORAGE_IO_THREADS="${KV_STORAGE_IO_THREADS:-16}"

# ---------- 2d. Scale-out replicas for the storage tier (dashboard Tier 3 panel) ----------
# ENABLE_KV_STORAGE_REPLICAS=1 points the dashboard at llmd-gpu-storage, a separate set of
# GPU replicas (scripts/62_deploy_kv_storage_replicas.sh) with host DRAM + shared storage on,
# outside the InferencePool. The dashboard runs a small ring of sessions across them so each
# returning turn is served from KV another replica wrote to Lustre. The live GPU pool and the
# routing A/B are not touched. The small host pool is deliberate: it keeps the working set on
# the storage tier so the panel shows storage reads, not host-DRAM hits.
export ENABLE_KV_STORAGE_REPLICAS="${ENABLE_KV_STORAGE_REPLICAS:-0}"
export KV_STORAGE_REPLICAS="${KV_STORAGE_REPLICAS:-2}"
export KV_STORAGE_REPLICA_HBM_CAP_BYTES="${KV_STORAGE_REPLICA_HBM_CAP_BYTES:-4294967296}"   # 4 GiB HBM KV per GPU (~9 sessions)
export KV_STORAGE_REPLICA_CPU_BYTES="${KV_STORAGE_REPLICA_CPU_BYTES:-4294967296}"           # 4 GiB host pool per replica (~9 sessions)
export KV_STORAGE_REPLICA_MEMORY="${KV_STORAGE_REPLICA_MEMORY:-64Gi}"
export KV_STORAGE_REPLICA_DSHM_SIZE="${KV_STORAGE_REPLICA_DSHM_SIZE:-16Gi}"
export KV_STORAGE_REPLICA_ROOT_DIR="${KV_STORAGE_REPLICA_ROOT_DIR:-${KV_STORAGE_MOUNT_PATH}/kv-cache/vllm-scaleout}"
export KV_STORAGE_WORKERS="${KV_STORAGE_WORKERS:-6}"       # dashboard ring workers driving the replicas
export KV_STORAGE_RING="${KV_STORAGE_RING:-5}"             # sessions per worker; the set is fixed, so storage use plateaus

# ---------- 3. GKE Agent Substrate & Hermes Agent Swarm ----------------------
export SUBSTRATE_NODEPOOL="${SUBSTRATE_NODEPOOL:-substrate-worker-pool}"
export SUBSTRATE_MACHINE_TYPE="${SUBSTRATE_MACHINE_TYPE:-n2-standard-16}"
export SUBSTRATE_NUM_NODES="${SUBSTRATE_NUM_NODES:-4}"

# Traffic generation parameters (1,000 Hermes agents)
export AGENT_EXECUTION_MODE="${AGENT_EXECUTION_MODE:-both}" # "substrate", "swarm", or "both"
export NUM_HERMES_AGENTS="${NUM_HERMES_AGENTS:-1000}"
export CONCURRENT_ACTIVE_AGENTS="${CONCURRENT_ACTIVE_AGENTS:-100}"
export MULTI_TURN_RATIO="${MULTI_TURN_RATIO:-0.6}" # 60% multi-turn (TPU), 40% one-shot (GPU)
export TURNS_PER_SESSION="${TURNS_PER_SESSION:-5}"
export FROZEN_PREFIX_TOKENS="${FROZEN_PREFIX_TOKENS:-2048}"
export SHARED_WORKSPACE_COHORTS="${SHARED_WORKSPACE_COHORTS:-20}"

# Custom HTTP headers used by Hermes agents & llm-d routing / Flow Control
export HEADER_WORKLOAD_TYPE="x-llm-d-workload"
export HEADER_FAIRNESS_ID="x-llm-d-inference-fairness-id"
export HEADER_OBJECTIVE="x-llm-d-inference-objective"
