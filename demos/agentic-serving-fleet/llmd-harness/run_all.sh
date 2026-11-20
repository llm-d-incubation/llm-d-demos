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

# Master runner for the llm-d Deployment & Benchmark Harness (`google/gemma-4-26B-A4B-it`).
#
# Usage:
#   export HF_TOKEN=<your_hf_token>
#   ./run_all.sh                 # run all stages (00 -> 50 -> 60 -> 65 -> 70 -> 80)
#   ./run_all.sh 60 70 80        # run specific stages

set -euo pipefail
HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_STAGES=(00 50 60 65 70 80)
STAGES=("${@:-${DEFAULT_STAGES[@]}}")

for s in "${STAGES[@]}"; do
  script="$(find "${HARNESS_ROOT}/scripts" -maxdepth 1 -name "${s}_*.sh" | head -n 1)"
  if [[ -z "${script}" ]]; then
    echo "Unknown stage: ${s}" >&2
    exit 1
  fi
  echo ""
  echo "============================================================================="
  echo "  Running Stage ${s}: $(basename "${script}")"
  echo "============================================================================="
  "${script}"
done
