#!/bin/sh
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

# Writes Hermes' config from the template's env, starts the gateway and warms it
# up (see warmup.py).
# Runs once per golden snapshot: every agent is restored from that snapshot,
# so nothing here runs again on wake.
set -eu

: "${API_SERVER_KEY:?API_SERVER_KEY must be set (at least 16 characters)}"
export MODEL_NAME="${MODEL_NAME:-google/gemma-4-12B-it}"
export MODEL_BASE_URL="${MODEL_BASE_URL:?MODEL_BASE_URL must be set, e.g. http://proxy:8091/llm/v1}"
export MAX_TOKENS="${MAX_TOKENS:-50}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"

mkdir -p "$HERMES_HOME"
python3 - <<'EOF'
import os, string
src = open("/opt/hermes/config.yaml.tmpl").read()
dst = os.path.join(os.environ["HERMES_HOME"], "config.yaml")
open(dst, "w").write(string.Template(src).substitute(os.environ))
EOF
[ -f /data/AGENTS.md ] || cp /opt/hermes/AGENTS.md /data/AGENTS.md

# The gateway runs as a child; warmup.py warms it up, answers the template's
# readyz on :8081/ready once warm, and exits if the gateway does.
hermes gateway run &
exec python3 /opt/hermes/warmup.py "$!"
