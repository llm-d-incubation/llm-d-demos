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

# Drive the demo arc against a backend and screenshot every screen with Chrome-in-Docker (host network).
#   shots.sh <base-url> <outdir>
set -u
BASE="${1:-http://127.0.0.1:8765}"; OUT="${2:-/tmp/shots6}"
mkdir -p "$OUT"
shot() { # name view
  timeout 240 docker run --rm --network host --user "$(id -u):$(id -g)" -v "$OUT":/out zenika/alpine-chrome:latest --headless --no-sandbox --disable-gpu \
    --disable-dev-shm-usage --hide-scrollbars --window-size=1920,1080 --virtual-time-budget=9000 \
    --screenshot="/out/$1.png" "$BASE/?view=$2" >/dev/null 2>&1
  echo "$1 -> $(stat -c %s "$OUT/$1.png" 2>/dev/null || echo MISSING) bytes"
}
post() { curl -s -X POST -H 'Content-Type: application/json' -d "$2" "$BASE/$1" >/dev/null; }
state() { curl -s "$BASE/api/state" | python3 -c "import json,sys;d=json.load(sys.stdin);c=d['llmd']['compare'];r=c['routing'];f=lambda m:(None if r.get(m) is None else {k:r[m].get(k) for k in ('tok_s','ttft_p50_ms','cache_hit_pct','n')});print(' state:',d['phase'],'rate',d['traffic'].get('rate'),'routing',c['routing_mode'],'flow',c['flow_mode'],'| without',f('without'),'| with',f('with'),'| with_fc',f('with_fc'))"; }

echo "== 1 open"; post api/strategy '{"routing":"without"}'; post api/strategy '{"flow":"without"}'
shot 1_stack_open stack
echo "== 2 agents"; post api/burst '{}' ; sleep 8; post api/simulate_traffic '{}'; sleep 6
shot 2_agents agents; state
shot 3_stack_live stack
echo "== 4 router without"; sleep 12; shot 4_router_without routing; state
echo "== 5 router with"; post api/strategy '{"routing":"with"}'; sleep 25; shot 5_router_with routing; state
shot 6_stack_router stack
echo "== 7 flow off"; shot 7_flow_off flow
echo "== 8 flow on"; post api/strategy '{"flow":"with"}'; sleep 25; shot 8_flow_on flow; state
echo "== 9 close"; shot 9_stack_close summary
