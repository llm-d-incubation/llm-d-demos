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
#
# One-liner for the team: open the llm-d dashboard from any machine with gcloud + kubectl.
# Prints the Internal LB URL (VPC-reachable) and keeps a local port-forward alive as the fallback.
#
#   scripts/87_open_dashboard.sh [--no-forward]      # PORT=8085 to change the local port
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
need_bin gcloud kubectl
get_credentials

PORT="${PORT:-8085}"
FORWARD=1
[[ "${1:-}" == "--no-forward" ]] && FORWARD=0

ilb="$(kubectl -n "${NAMESPACE}" get svc llmd-live-dashboard-ilb -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)"
if [[ -n "${ilb}" ]]; then
  log "Internal LB (from the VPC / cloudtop):  http://${ilb}/?view=stack"
else
  warn "No Internal LB yet (run scripts/86_deploy_dashboard_image.sh --no-build, or use the port-forward below)"
fi
kubectl -n "${NAMESPACE}" get deploy llmd-live-dashboard -o jsonpath='image={.spec.template.spec.containers[0].image} ready={.status.readyReplicas}/{.spec.replicas}{"\n"}'

if (( FORWARD )); then
  log "Port-forward fallback: http://localhost:${PORT}/?view=stack  (Ctrl-C to stop; auto-reconnects)"
  while true; do
    kubectl -n "${NAMESPACE}" port-forward --address 0.0.0.0 svc/llmd-live-dashboard "${PORT}:80" || true
    sleep 1
  done
fi
