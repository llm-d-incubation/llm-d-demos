#!/bin/bash
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

set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PORT="${PORT:-8080}"
SIM_PORT="${SIM_PORT:-8086}"
HARNESS="${HARNESS:-hermes}"
APP_MODE="${APP_MODE:-dual}"

echo "Starting llm-d dashboard with APP_MODE=${APP_MODE}, PORT=${PORT}, HARNESS=${HARNESS}..."

cleanup() {
    echo "Caught shutdown signal. Terminating processes..."
    if [ -n "${SIM_PID:-}" ]; then
        kill -TERM "${SIM_PID}" 2>/dev/null || true
    fi
    if [ -n "${SERVER_PID:-}" ]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    exit 0
}

trap cleanup TERM INT

if [ "${APP_MODE}" = "sim" ]; then
    echo "Running standalone simulation server on port ${PORT}..."
    python3 mock_server.py --host=0.0.0.0 --port="${PORT}" --harness="${HARNESS}" "$@" &
    SERVER_PID=$!
elif [ "${APP_MODE}" = "live" ]; then
    echo "Running live server on port ${PORT}..."
    python3 server.py --host=0.0.0.0 --port="${PORT}" --harness="${HARNESS}" "$@" &
    SERVER_PID=$!
else
    # Dual mode (default): run mock_server on internal port and server.py on main port with SIM_PROXY
    echo "Starting background simulation sidecar on 127.0.0.1:${SIM_PORT}..."
    python3 mock_server.py --host=127.0.0.1 --port="${SIM_PORT}" --harness="${HARNESS}" &
    SIM_PID=$!

    export SIM_PROXY="http://127.0.0.1:${SIM_PORT}"
    echo "Starting primary server on 0.0.0.0:${PORT} (proxying /sim/* to ${SIM_PROXY})..."
    python3 server.py --host=0.0.0.0 --port="${PORT}" --harness="${HARNESS}" "$@" &
    SERVER_PID=$!
fi

wait "${SERVER_PID}"
cleanup
