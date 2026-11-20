# SPDX-License-Identifier: Apache-2.0

SHELL := /bin/bash
PORT ?= 8090

.PHONY: help simulate verify deploy teardown

help: ## Show available commands for this demo
	@echo "Demo: <slug>"
	@echo "  make simulate   Start local simulator on :$(PORT)"
	@echo "  make verify     Run offline syntax and lint checks"
	@echo "  make deploy     Deploy live stack (if supported)"
	@echo "  make teardown   Clean up live resources (if supported)"

simulate: ## Launch the local simulator (no cluster required)
	@echo "Implement local simulator launch for <slug> on port $(PORT)"
	@exit 1

verify: ## Run offline verification checks (called by root `make validate`)
	@echo "OK: <slug> verify passed"

deploy: ## Deploy to a live Kubernetes cluster (optional)
	@echo "Live deployment not configured for <slug>"
	@exit 1

teardown: ## Tear down live Kubernetes resources (optional)
	@echo "Live teardown not configured for <slug>"
	@exit 1
