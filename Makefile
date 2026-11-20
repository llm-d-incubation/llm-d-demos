# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# llm-d-demos: Root Catalog Dispatcher & Validation Makefile
# ==============================================================================

SHELL := /bin/bash
PYTHON ?= python3
DEMO ?=
SLUG ?=
PORT ?=

.PHONY: help list sim new-demo index check-index validate

help: ## Show available catalog commands
	@echo "=============================================================================="
	@echo "  llm-d-demos: Community Demo Catalog"
	@echo "=============================================================================="
	@echo ""
	@echo "Catalog commands:"
	@echo "  make list                     List all demos in demos/"
	@echo "  make sim DEMO=<slug>          Run a demo's local simulator (no cluster needed)"
	@echo "  make new-demo SLUG=<slug>     Scaffold a new demo from templates/demo-starter/"
	@echo "  make index                    Regenerate README.md catalog table & catalog.json"
	@echo "  make validate                 Validate schemas, hygiene, catalog sync, & verify targets"
	@echo ""
	@echo "Example:"
	@echo "  make sim DEMO=agentic-serving-fleet"
	@echo "=============================================================================="

list: ## List registered demos from demos/*/demo.yaml
	@$(PYTHON) scripts/render_catalog.py --list

sim: ## Run a demo's local simulator (usage: make sim DEMO=<slug> [PORT=8090])
	@if [ -z "$(DEMO)" ]; then \
	  echo "ERROR: specify DEMO=<slug>. Available demos:" >&2; \
	  $(MAKE) --no-print-directory list >&2; \
	  exit 1; \
	fi
	@if [ ! -d "demos/$(DEMO)" ]; then \
	  echo "ERROR: demos/$(DEMO) does not exist." >&2; \
	  exit 1; \
	fi
	@$(MAKE) -C "demos/$(DEMO)" simulate $(if $(PORT),PORT=$(PORT),)

new-demo: ## Scaffold a new demo from templates/demo-starter/ (usage: make new-demo SLUG=<slug>)
	@if [ -z "$(SLUG)" ]; then \
	  echo "ERROR: specify SLUG=<lowercase-kebab-name> (e.g. make new-demo SLUG=pd-disaggregation-live)" >&2; \
	  exit 1; \
	fi
	@if [ -e "demos/$(SLUG)" ]; then \
	  echo "ERROR: demos/$(SLUG) already exists." >&2; \
	  exit 1; \
	fi
	@mkdir -p demos
	@cp -R templates/demo-starter "demos/$(SLUG)"
	@$(PYTHON) -c 'import pathlib; slug = "$(SLUG)"; root = pathlib.Path("demos") / slug; \
	  [p.write_text(p.read_text().replace("<slug>", slug)) for p in root.rglob("*") if p.is_file()]'
	@echo "Scaffolded demos/$(SLUG)/ from templates/demo-starter/."
	@echo "Next: edit demos/$(SLUG)/demo.yaml, README.md, OWNERS, and Makefile, then run 'make index && make validate'."

index: ## Regenerate README.md catalog table and catalog.json from demos/*/demo.yaml
	@$(PYTHON) scripts/render_catalog.py

check-index: ## Verify README.md and catalog.json are up to date
	@$(PYTHON) scripts/render_catalog.py --check

validate: ## Run schema, hygiene, catalog sync, and per-demo verify checks
	@$(PYTHON) scripts/validate.py
