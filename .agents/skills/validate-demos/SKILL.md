---
name: validate-demos
description: Workflow for validating demo schemas, catalog sync, secret/internal-reference hygiene, and offline simulator smoke tests in llm-d-demos.
---

# Validate Demos in llm-d-demos

Run these checks before opening a PR or finalizing any change to a demo or root tooling.

## 1. Run Full Repository Validation

```bash
make validate
```

This executes `python3 scripts/validate.py`, which checks:

1. Every directory in `demos/` has a valid kebab-case slug matching `demo.yaml` `id`.
2. Required files exist (`demo.yaml`, `README.md`, `OWNERS`, `Makefile`).
3. `demo.yaml` matches Schema v1 and uses only approved `capabilities` slugs.
4. `demo.yaml` `maintainers` align with `demos/<slug>/OWNERS`.
5. `Makefile` exposes required targets (`help`, `verify`, plus `simulate` when `modes.simulator.supported` is `true`).
6. Hygiene scan across non-binary files finds zero secrets, zero internal corporate URLs, and zero disallowed leftover references.
7. `README.md` and `catalog.json` match the output of `python3 scripts/render_catalog.py --check`.
8. Every demo's `make -C demos/<slug> verify` exits 0 offline.

## 2. Check Catalog Freshness

If `validate.py` reports catalog drift:

```bash
make index
git diff README.md catalog.json
```

Inspect the diff to confirm only intended catalog rows changed.

## 3. Smoke-Test Simulator Mode (Offline)

For any demo with `modes.simulator.supported: true` that was added or modified:

1. Start the simulator on a temporary free local port (for example `PORT=18090 make -C demos/<slug> simulate` or the command listed in `demo.yaml`).
2. Confirm the main HTTP endpoint responds `200 OK` within a few seconds and does not crash on load.
3. Stop the background simulator process cleanly before finishing.

Never run `deploy`, `kubectl apply`, or cloud provisioning scripts as part of local validation.
