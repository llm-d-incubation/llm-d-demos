## Summary

<!-- What does this PR add or change? If adding a new demo, summarize what llm-d capabilities it demonstrates. -->

## Type of Change

- [ ] New demo under `demos/<slug>/`
- [ ] Update to an existing demo
- [ ] Repo tooling, templates, or documentation

## Checklist

- [ ] Commits are DCO signed-off (`git commit -s`) per `PR_SIGNOFF.md`
- [ ] All demo code lives under `demos/<slug>/` (no demo code added at the repo root)
- [ ] `demo.yaml`, `README.md`, `OWNERS`, and `Makefile` are present and up to date
- [ ] `make index` and `make validate` pass locally
- [ ] Simulator mode tested locally (if `modes.simulator.supported: true`)
- [ ] No secrets, tokens, hardcoded cloud project IDs, internal URLs, or personal namespaces
- [ ] Default UI and documentation framing stays vendor-neutral (cloud/accelerator specifics scoped under Reference Environment)
