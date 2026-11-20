# Contributing to llm-d-demos

Thank you for contributing to `llm-d-demos`. This repository follows the broader [llm-d Contributing Guidelines](https://github.com/llm-d/llm-d/blob/main/CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## Developer Certificate of Origin (DCO)

Every commit must include a DCO sign-off line (`Signed-off-by: Name <email>`). See [DCO](DCO) and [PR_SIGNOFF.md](PR_SIGNOFF.md) for details.

```bash
git commit -s -m "Add my-demo under demos/my-demo"
```

## Adding a New Demo

Before writing code, read [DEMOS-POLICY.md](DEMOS-POLICY.md) to make sure your contribution belongs in `llm-d-demos` (rather than a production deployment guide in [llm-d/llm-d/guides](https://github.com/llm-d/llm-d/tree/main/guides)).

1. **Scaffold from the starter template:**
   ```bash
   make new-demo SLUG=my-demo-name
   ```
2. **Build your demo inside `demos/my-demo-name/`:**
   - Keep all source code, manifests, scripts, Dockerfiles, and docs inside your demo directory. Never place demo code at the repository root.
   - Include a zero-cluster **simulator or local mode** whenever practical so anyone can experience the demo without reserving accelerators.
   - Keep default UI copy and top-level framing vendor-neutral. Declare cloud, accelerator, engine, and model details under `modes.live.reference_environment` in `demo.yaml` and in your demo's Reference Environment section.
   - Implement the standard `Makefile` targets (`help`, `verify`, plus `simulate` when simulator mode is supported).
3. **Fill in `demo.yaml` and `OWNERS`:**
   - Tag the `llm-d` capabilities your demo showcases using the controlled taxonomy in [DEMOS-POLICY.md](DEMOS-POLICY.md).
   - List at least one active maintainer in both `OWNERS` and `demo.yaml`, and add a matching path rule in `.github/CODEOWNERS`.
4. **Regenerate the catalog and validate:**
   ```bash
   make index
   make validate
   ```
5. **Open a pull request** with DCO-signed commits.

## Updating an Existing Demo

- Keep changes scoped to the relevant `demos/<slug>/` directory.
- If you edit `demo.yaml` (title, summary, capabilities, status, or links), run `make index` to refresh the root `README.md` catalog table and `catalog.json`.
- Run `make validate` before opening your PR.

## Using AI Coding Agents

If you are building or updating a demo with a coding agent, point it at [AGENTS.md](AGENTS.md) and the playbooks in `.agents/skills/`:

- `.agents/skills/add-new-demo/SKILL.md`: scaffold, wire, test, and register a new demo
- `.agents/skills/validate-demos/SKILL.md`: run schema, hygiene, catalog sync, and simulator smoke checks

## Community Channels

- **Slack:** [llm-d Developer Slack](https://llm-d.ai/slack)
- **Main Repository:** [llm-d/llm-d](https://github.com/llm-d/llm-d)
- **Bi-weekly Community Meeting:** [Shared Calendar](https://red.ht/llm-d-public-calendar)
