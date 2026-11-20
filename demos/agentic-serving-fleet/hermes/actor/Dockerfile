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

# Hermes Agent (NousResearch, MIT) as the harness inside each Hermes agent.
#
# One Hermes gateway process per Substrate sandbox, exposing its
# OpenAI-compatible API server on :80. Built only from public sources: the
# upstream hermes-agent package, pinned by commit.
#
# Everything the agent must remember lives under /data (sessions in SQLite,
# config). The template snapshots the whole sandbox on pause (memory + rootfs),
# so it survives suspend/resume without a durable volume.
FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/data \
    HERMES_HOME=/data/.hermes \
    API_SERVER_HOST=0.0.0.0 \
    API_SERVER_PORT=80

# Pinned by commit: the golden snapshot is taken from this image, so a moving
# dependency would silently change every restored agent.
# The base package plus aiohttp (via the small "homeassistant" extra) is all the
# API server needs; "[all]" adds integrations we don't use.
# HERMES_NIX_BUILD=1 is upstream's switch for packagers: without it, setup.py
# refuses to build a regular (non-editable) wheel.
ARG HERMES_REF=5ffbea81e995aaa3b6b6b5ba81d1f453a6b88ef8
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && HERMES_NIX_BUILD=1 pip install \
      "hermes-agent[homeassistant] @ git+https://github.com/NousResearch/hermes-agent.git@${HERMES_REF}" \
    && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /usr/local/bin/hermes-entrypoint
COPY config.yaml.tmpl AGENTS.md warmup.py /opt/hermes/
RUN chmod +x /usr/local/bin/hermes-entrypoint

WORKDIR /data
EXPOSE 80 8081
ENTRYPOINT ["/usr/local/bin/hermes-entrypoint"]
