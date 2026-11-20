# llm-d Demos

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Community catalog of interactive demos, live-stage applications, and zero-cluster simulators showcasing [llm-d](https://github.com/llm-d/llm-d) capabilities on Kubernetes.

- Looking for **production deployment recipes**? See [llm-d/llm-d/guides](https://github.com/llm-d/llm-d/tree/main/guides).
- Looking for **interactive demos, booth apps, and visual scenarios**? You are in the right place.

---

## Demo Catalog

<!-- catalog:start -->
_No demos registered yet. Run `make new-demo SLUG=<name>` to add the first demo._
<!-- catalog:end -->

---

## Quickstart

List all available demos:

```bash
make list
```

Launch any demo that supports local simulation (no cluster or accelerators required):

```bash
make sim DEMO=<slug>
```

Validate schemas, hygiene checks, and catalog sync:

```bash
make validate
```

---

## Adding a New Demo

Every demo in this repository lives in its own self-contained directory under `demos/<slug>/` with its own stack, `Makefile`, `OWNERS`, `README.md`, and `demo.yaml`. Nothing at the repository root is tied to a single demo.

### For Human Contributors

1. Read [DEMOS-POLICY.md](DEMOS-POLICY.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
2. Scaffold from the starter template:
   ```bash
   make new-demo SLUG=my-demo-name
   ```
3. Build your demo inside `demos/my-demo-name/` (local simulator mode strongly encouraged).
4. Regenerate the catalog and run validation:
   ```bash
   make index
   make validate
   ```
5. Open a PR with DCO-signed commits (`git commit -s`).

### For AI Coding Agents

Start with [AGENTS.md](AGENTS.md) (also linked from [CLAUDE.md](CLAUDE.md)) and follow the repo skills under `.agents/skills/`:

- [`.agents/skills/add-new-demo/SKILL.md`](.agents/skills/add-new-demo/SKILL.md): scaffold, implement, document, and register a new demo
- [`.agents/skills/validate-demos/SKILL.md`](.agents/skills/validate-demos/SKILL.md): verify schemas, hygiene, catalog sync, and offline simulator startup

---

## Ideas for Future Demos

Community contributions are welcome! Areas where interactive demos would complement existing [llm-d guides](https://github.com/llm-d/llm-d/tree/main/guides):

- **Prefill/Decode Disaggregation:** Live TTFT vs. ITL stability under long-prompt bursts comparing aggregated vs. disaggregated pools.
- **Session-Aware & Turn-Priority Routing:** Visualizing multi-turn conversation fairness and prefix affinity under mixed session depths.
- **Workload Variant Autoscaling (WVA):** Interactive scaling demo driven by EPP queue depth and KV pressure across heterogeneous pools.
- **Fast Model Actuation & Pod Snapshots:** Live cold-start and model-swapping comparison.
- **Multimodal & Omni Serving:** Interactive visual/audio request routing across specialized Encode, Prefill, and Decode pools.

---

## Governance & License

- **License:** [Apache License 2.0](LICENSE)
- **Code of Conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Security:** [SECURITY.md](SECURITY.md)
- **Commit Sign-off (DCO):** [PR_SIGNOFF.md](PR_SIGNOFF.md)
