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

# ==============================================================================
# Agentic Serving Fleet Demo
# Local Simulator & Reference-Environment (GKE) Reproduction Makefile
# ==============================================================================

SHELL := /bin/bash
PORT ?= 8090
HARNESS ?= hermes

.PHONY: help simulate rehearse verify verify-oss deploy preflight deploy-cluster deploy-llmd deploy-substrate verify-routing benchmark dashboard-live teardown

help: ## Show all available commands for local simulation and live cluster deployment
	@echo "=============================================================================="
	@echo "  Agentic Serving Fleet Demo (llm-d)"
	@echo "=============================================================================="
	@echo ""
	@echo "Quick Local Simulator (No Cluster Required):"
	@echo "  make simulate          Start the interactive 5-screen dashboard on :$(PORT)"
	@echo "  make verify            Run offline syntax, shell, and license header checks"
	@echo ""
	@echo "Live Reference-Environment Deployment (GKE: TPU v7x + NVIDIA GPUs + Agent Substrate):"
	@echo "  make preflight         Verify gcloud/kubectl/helm, HF_TOKEN, LLMD_REPO, and cluster access"
	@echo "  make deploy-cluster    Provision TPU v7x, NVIDIA GPU, and Agent Substrate CPU node pools"
	@echo "  make deploy-llmd       Deploy llm-d Gateway, GAIE Endpoint Picker, and Gemma 4 vLLM pods"
	@echo "  make deploy-substrate  Deploy Agent Substrate WorkerPool and 1,000 Hermes Agents"
	@echo "  make verify-routing    Verify multi-turn and one-shot header/KV routing"
	@echo "  make benchmark         Run the 1,000-Hermes-Agent A/B benchmark against llm-d"
	@echo "  make dashboard-live    Connect the interactive dashboard to the live cluster"
	@echo "  make teardown          Clean up demo namespaces (set DELETE_NODEPOOLS=1 for pools)"
	@echo ""
	@echo "Configuration:"
	@echo "  Override defaults via environment variables or llmd-harness/config/env.sh:"
	@echo "    export PROJECT_ID=<your-gcp-project-id>"
	@echo "    export CLUSTER_NAME=llmd-demo-cluster"
	@echo "    export LLMD_REPO=\$$HOME/llm-d    # local clone of https://github.com/llm-d/llm-d"
	@echo "    export HF_TOKEN=<your-huggingface-token>"
	@echo "=============================================================================="

simulate rehearse: ## Launch the standalone dashboard simulator on http://localhost:8090
	python3 dashboard/mock_server.py $(PORT) --harness $(HARNESS)

verify verify-oss: ## Run offline Python compile, shell syntax, and Apache-2.0 header checks
	@python3 -m py_compile dashboard/server.py dashboard/mock_server.py dashboard/abbench.py dashboard/logos/embed.py llmd-harness/agents/hermes_swarm.py
	@for sh in $$(find llmd-harness manifests hermes dashboard -type f -name '*.sh' | sort); do bash -n "$$sh"; done
	@python3 -c 'import pathlib; root=pathlib.Path("."); \
	  exts={".py",".sh",".go",".c",".mjs"}; names={"Dockerfile","Makefile"}; \
	  files=[p for p in sorted(root.rglob("*")) if p.is_file() and (p.suffix in exts or p.name in names) and "__pycache__" not in p.parts]; \
	  missing=[str(p) for p in files if "Apache" not in p.read_text(errors="ignore")[:900] and "SPDX-License-Identifier" not in p.read_text(errors="ignore")[:900]]; \
	  assert not missing, f"Missing Apache/SPDX headers: {missing}"; \
	  print(f"OK: {len(files)} demo source files passed syntax and Apache-2.0 license checks.")'

deploy: preflight deploy-cluster deploy-llmd deploy-substrate ## Run full live reference-environment deployment

preflight: ## Run preflight checks (CLI tools, auth, LLMD_REPO, cluster connectivity)
	./llmd-harness/scripts/00_preflight.sh

deploy-cluster: ## Provision GKE node pools (TPU v7x, NVIDIA GPU, Agent Substrate CPU)
	./llmd-harness/scripts/50_prepare_cluster.sh

deploy-llmd: ## Deploy heterogeneous llm-d Gateway, Endpoint Picker (EPP), and vLLM pods
	./llmd-harness/scripts/60_deploy_llmd_heterogeneous.sh

deploy-substrate: ## Deploy Agent Substrate WorkerPool, ActorTemplate, and 1,000 Hermes Agents
	./llmd-harness/scripts/65_deploy_substrate_and_agents.sh

verify-routing: ## Send smoke requests to verify llm-d KV-aware and priority flow routing
	./llmd-harness/scripts/70_verify_routing.sh

benchmark: ## Run the 1,000-Hermes-Agent swarm benchmark against the live cluster
	./llmd-harness/scripts/80_run_benchmark.sh

dashboard-live: ## Connect the interactive dashboard to the live cluster
	./llmd-harness/scripts/85_run_dashboard.sh

teardown: ## Tear down deployed demo workloads
	./llmd-harness/scripts/90_teardown.sh
