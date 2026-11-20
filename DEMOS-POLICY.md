# llm-d Demos: Definition and Policy

This document defines what belongs in `llm-d-demos`, how demos relate to upstream `llm-d` guides, and the requirements for adding and maintaining a demo.

## 1. Demos vs. Guides

`llm-d/llm-d/guides/` and `llm-d-demos` serve different purposes:

| Surface | Question it answers | Primary artifact |
| :--- | :--- | :--- |
| **[llm-d/llm-d/guides](https://github.com/llm-d/llm-d/tree/main/guides)** | "How do I deploy llm-d capability X in production?" | Kustomize overlays, Helm values, `guide.yaml`, nightly verification |
| **llm-d-demos (this repo)** | "How do I see, experience, or present llm-d capability X in action?" | Interactive UIs, local simulators, live stage/booth apps, visual workload scenarios |

As stated in [llm-d's Guides Policy](https://github.com/llm-d/llm-d/blob/main/guides/GUIDES-POLICY.md), one-off conference apps, stage demos, and interactive showcases belong here rather than inside `llm-d/llm-d/guides/`.

### What is NOT a demo here

- **A production deployment guide or model recipe:** Contribute to [llm-d/llm-d/guides](https://github.com/llm-d/llm-d/tree/main/guides).
- **A benchmark harness or sweep script without an interactive or storytelling experience:** Contribute to [llm-d/llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark).
- **A vendored fork of upstream `llm-d` guides:** Reference upstream guides by URL or require a local checkout of `llm-d/llm-d` via `LLMD_REPO` instead of copying manifests into this repo.

## 2. Structural Rules

1. **Every demo lives under `demos/<slug>/`.** Never put demo application code, Dockerfiles, or manifests at the repository root.
2. **Every demo is self-contained and stack-agnostic.** Choose the language, frontend, or tooling that best fits the demo (Python, Go, TypeScript, notebooks, Grafana, Kubernetes manifests). Do not force one UI shell across unrelated demos.
3. **Rule of Two for `shared/`.** Do not extract code into `shared/` speculatively. Only promote a utility, component, or asset into `shared/` once at least two demos actively use it.
4. **Vendor-neutral default framing.** Keep the repository root and default simulator views community-neutral (`llm-d`, Kubernetes, generic accelerator/node-pool labels). Cloud, hardware, and engine specifics belong in `modes.live.reference_environment` inside `demo.yaml`, in the demo's Reference Environment section, or behind an explicit optional mode toggle.

## 3. Baseline Requirements for Every Demo

Every directory under `demos/<slug>/` must contain:

1. **`demo.yaml`**: Machine-readable catalog manifest conforming to Schema v1 (validated by `scripts/validate.py`).
2. **`README.md`**: Human-readable guide with these sections:
   - Overview (what `llm-d` capabilities it shows and why)
   - Quickstart (Simulator / local run first, when supported)
   - Live Cluster Deployment & Reference Environment (when live mode is supported)
   - Architecture (how the pieces fit together)
   - Cleanup (tearing down any live resources)
3. **`OWNERS`**: Kubernetes-style owners file naming at least one maintainer accountable for keeping the demo working.
4. **`Makefile`**: Exposes the standard interface targets where applicable:
   - `help`: lists demo commands
   - `verify`: offline syntax, lint, or smoke validation run by root `make validate`
   - `simulate`: required whenever `modes.simulator.supported: true`
   - `deploy` / `teardown`: recommended whenever `modes.live.supported: true`
5. **Hygiene**: Apache-2.0 license headers on source files, zero secrets or API tokens, zero hardcoded personal usernames or real cloud project IDs, and zero internal corporate links.

## 4. Simulator vs. Live Modes

- **Simulator mode (`modes.simulator.supported: true`)** is strongly encouraged. It lets conference speakers, community members, and evaluators run the demo locally in under a minute with no cluster or accelerators. Simulator data must be clearly labeled as simulated in the UI.
- **Live mode (`modes.live.supported: true`)** connects to a real Kubernetes cluster running `llm-d`. Every live demo must declare its **Reference Environment** (platform, accelerators, serving engines, and model) in `demo.yaml` and `README.md`, following the same honesty standard as `llm-d/llm-d/guides/`.

## 5. Controlled Capability Taxonomy

Use values from this list in `demo.yaml` (`capabilities:`) so the catalog stays consistent and maps cleanly to upstream `llm-d` guides:

- `optimized-baseline`
- `precise-prefix-cache-routing`
- `predicted-latency-routing`
- `session-aware-routing`
- `flow-control`
- `turn-priority-fairness`
- `tiered-prefix-cache`
- `p2p-kv-cache-sharing`
- `pd-disaggregation`
- `wide-ep`
- `multimodal-serving`
- `omni-serving`
- `agentic-api`
- `workload-autoscaling`
- `fast-model-actuation`
- `pod-snapshot`
- `multi-model-routing`
- `batch-serving`
- `heterogeneous-accelerators`
- `reinforcement-learning`

If a new demo showcases a genuinely new `llm-d` capability, propose adding the capability slug to both this file and `scripts/validate.py` in the same PR.

## 6. Lifecycle

| Status | Meaning |
| :--- | :--- |
| `experimental` | Early or work-in-progress demo; APIs, visuals, or live scripts may change quickly. |
| `active` | Maintained demo passing CI validation, with active `OWNERS` and up-to-date docs. |
| `archived` | Historical demo no longer actively maintained (e.g., superseded by a newer demo or tied to an old `llm-d` release). Kept for reference or removed after notice. |

Maintainers are expected to keep `make -C demos/<slug> verify` green, keep `demo.yaml` accurate, and update `status: archived` if a demo is no longer maintained.
