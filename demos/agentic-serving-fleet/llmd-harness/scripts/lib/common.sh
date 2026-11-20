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

# Shared helper library for llmd-harness scripts.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../config/env.sh
source "${_SCRIPT_DIR}/../../config/env.sh"
export KUBECONFIG="${KUBECONFIG:-${HARNESS_ROOT}/.kubeconfig}"

# ---------- Logging -----------------------------------------------------------
_c_reset=$'\033[0m'; _c_blue=$'\033[1;34m'; _c_green=$'\033[1;32m'
_c_yellow=$'\033[1;33m'; _c_red=$'\033[1;31m'; _c_dim=$'\033[2m'

log()   { printf '%s==>%s %s\n' "${_c_blue}"   "${_c_reset}" "$*"; }
ok()    { printf '%s  ok%s %s\n' "${_c_green}"  "${_c_reset}" "$*"; }
warn()  { printf '%swarn%s %s\n' "${_c_yellow}" "${_c_reset}" "$*" >&2; }
die()   { printf '%sFAIL%s %s\n' "${_c_red}"    "${_c_reset}" "$*" >&2; exit 1; }
dim()   { printf '%s     %s%s\n' "${_c_dim}" "$*" "${_c_reset}"; }

run() {
  printf '%s  $ %s%s\n' "${_c_dim}" "$*" "${_c_reset}"
  [[ "${DRY_RUN:-0}" == "1" ]] && return 0
  "$@"
}

# ---------- Guards ------------------------------------------------------------
need_bin() {
  for b in "$@"; do
    command -v "$b" >/dev/null 2>&1 || die "required binary not found: $b"
  done
}

need_env() {
  for v in "$@"; do
    [[ -n "${!v:-}" ]] || die "required variable is empty: $v (export it or set in config/env.sh)"
  done
}

# ---------- GCloud / Kubectl --------------------------------------------------
gc()  { gcloud --project="${PROJECT_ID}" "$@"; }
kc()  { kubectl --namespace="${NAMESPACE}" "$@"; }

cluster_exists() {
  gc container clusters describe "${CLUSTER_NAME}" \
    --location="${CLUSTER_LOCATION}" >/dev/null 2>&1
}

nodepool_exists() {
  gc container node-pools describe "$1" \
    --cluster="${CLUSTER_NAME}" --location="${CLUSTER_LOCATION}" >/dev/null 2>&1
}

get_credentials() {
  run gc container clusters get-credentials "${CLUSTER_NAME}" \
    --location="${CLUSTER_LOCATION}"
}

topology_chips() {
  local t="${1:-${TPU_TOPOLOGY}}"
  awk -F'x' '{print $1*$2*$3}' <<<"$t"
}

reservation_free_chips() {
  local reserved in_use
  reserved="$(gc compute reservations describe "${RESERVATION_NAME}" --zone="${ZONE}" \
    --format='value(aggregateReservation.reservedResources[0].accelerator.acceleratorCount)' 2>/dev/null || echo 0)"
  in_use="$(gc compute reservations describe "${RESERVATION_NAME}" --zone="${ZONE}" \
    --format='value(aggregateReservation.inUseResources[0].accelerator.acceleratorCount)' 2>/dev/null || echo 0)"
  reserved="${reserved:-0}"; in_use="${in_use:-0}"
  echo $(( reserved - in_use ))
}

cluster_existing_tpu_chips() {
  local topo="${1:-${TPU_TOPOLOGY}}"
  cluster_exists || { echo 0; return; }
  gc container clusters get-credentials "${CLUSTER_NAME}" --location="${CLUSTER_LOCATION}" >/dev/null 2>&1 || { echo 0; return; }
  local nodes
  nodes="$(kubectl get nodes -l "cloud.google.com/gke-tpu-accelerator=tpu7x,cloud.google.com/gke-tpu-topology=${topo}" --no-headers 2>/dev/null | wc -l || echo 0)"
  echo $(( nodes * TPU_CHIPS_PER_NODE ))
}
