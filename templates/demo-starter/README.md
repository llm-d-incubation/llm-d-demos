# <Demo Title>

> **Status:** Experimental | **Capabilities:** `optimized-baseline` | **Modes:** Local Simulator

One-paragraph summary explaining the problem this demo illustrates, what a viewer sees, and which `llm-d` capabilities are highlighted.

## Overview

- **What it demonstrates:** Concise bullet list of the key `llm-d` behaviors shown.
- **Upstream guides:** Links to the corresponding [llm-d/llm-d/guides](https://github.com/llm-d/llm-d/tree/main/guides) pages.

## Quickstart (Simulator)

Run the demo locally with no cluster or accelerators required:

```bash
make -C demos/<slug> simulate
# Open http://localhost:8090
```

Explain what to click or observe in 3 to 5 short steps.

## Live Cluster Deployment & Reference Environment

*(If `modes.live.supported: true`, document prerequisites, declared Reference Environment, and deploy commands here. Otherwise, state that this demo is simulator-only.)*

| Field | Reference Environment |
| :--- | :--- |
| Platform | Kubernetes 1.32+ |
| Accelerators | TBD |
| Serving Engine | vLLM / SGLang |
| Model | `<HuggingFace/Model-Name>` |

```bash
make -C demos/<slug> deploy
```

## Architecture

Include a concise diagram or component list explaining how the workload generator, `llm-d` router/gateway, serving engine, and UI interact.

## Cleanup

Document exact steps to tear down any live cluster resources created by this demo:

```bash
make -C demos/<slug> teardown
```
