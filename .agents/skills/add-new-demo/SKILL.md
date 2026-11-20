---
name: add-new-demo
description: Step-by-step workflow for scaffolding, implementing, documenting, and registering a new demo under demos/<slug>/ in llm-d-demos.
---

# Add a New Demo to llm-d-demos

Follow this workflow whenever adding a new demo or migrating an external demo into `llm-d-demos`.

## 1. Check Scope First

Confirm the contribution belongs in `llm-d-demos` (interactive app, stage/booth demo, visual scenario, or simulator) rather than `llm-d/llm-d/guides/` or `llm-d/llm-d-benchmark`. See `DEMOS-POLICY.md`.

Pick a short, durable, lowercase kebab-case slug describing the **workload or capability** (for example `pd-disaggregation-live`, `session-routing-playground`, `wva-autoscaling-sim`). Avoid one-time event names (like `kubecon-2026`) or vendor names in the directory slug.

## 2. Scaffold From the Starter Template

Never copy an existing complex demo as a base. Run:

```bash
make new-demo SLUG=<slug>
```

This creates `demos/<slug>/` from `templates/demo-starter/` with `demo.yaml`, `README.md`, `OWNERS`, and `Makefile`.

## 3. Fill in `demos/<slug>/demo.yaml`

Populate every field accurately:

- `schema_version`: `1`
- `id`: must match `<slug>`
- `title` and `summary`: concise, vendor-neutral description of what the user experiences
- `status`: `experimental` for early work, `active` when ready for general use
- `capabilities`: pick only slugs from the controlled taxonomy in `DEMOS-POLICY.md` (if a brand-new capability is needed, update `DEMOS-POLICY.md` and `scripts/validate.py` in the same PR)
- `modes.simulator`: strongly encouraged; set `supported: true`, `command`, `port`, and `requirements`
- `modes.live`: set `supported: true` and fill in `reference_environment` (`platform`, `accelerators`, `engines`, `model`) if live deployment manifests/scripts are included
- `links.guides`: link to the corresponding upstream `https://github.com/llm-d/llm-d/tree/main/guides/...` pages instead of vendoring guide YAML
- `maintainers`: GitHub handles matching `demos/<slug>/OWNERS`

## 4. Build Inside `demos/<slug>/`

- Keep all source code, manifests, Dockerfiles, images, and docs strictly inside `demos/<slug>/`.
- Use only relative paths anchored to the demo directory (for example `$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)` in scripts, or `make -C demos/<slug>`).
- Keep default simulator UI copy vendor-neutral. Clearly label simulated telemetry as simulated.
- Add Apache-2.0 SPDX headers (`SPDX-License-Identifier: Apache-2.0`) to new source files.
- Never hardcode secrets, API keys, kubeconfig contexts, real GCP/AWS/Azure project IDs, personal namespaces, or internal corporate links.
- Wire `make -C demos/<slug> verify` to run fast, offline checks (syntax compile, lint, unit tests) with zero cluster access.

## 5. Write `demos/<slug>/README.md` and Ownership

- Follow the required section order from `templates/demo-starter/README.md`:
  1. Overview
  2. Quickstart (Simulator) (first, if supported)
  3. Live Cluster Deployment & Reference Environment (if supported)
  4. Architecture
  5. Cleanup
- Add demo maintainers to `demos/<slug>/OWNERS` and add a matching path rule in `.github/CODEOWNERS`.

## 6. Regenerate Catalog & Validate

```bash
make index
make validate
```

If `modes.simulator.supported` is `true`, follow `.agents/skills/validate-demos/SKILL.md` to confirm the simulator starts cleanly and serves HTTP 200 on its declared port.

## 7. Commit With DCO Sign-Off

```bash
git commit -s -m "Add <slug> demo"
```
