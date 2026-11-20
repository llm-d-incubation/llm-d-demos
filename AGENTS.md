# Working in llm-d-demos

What an AI coding agent or new contributor needs before changing anything in this repository.

`llm-d-demos` is a **multi-demo catalog** for the `llm-d` community. Each demo is a self-contained directory under `demos/<slug>/` with its own `demo.yaml`, `README.md`, `OWNERS`, and `Makefile`. Read [DEMOS-POLICY.md](DEMOS-POLICY.md) for scope and lifecycle rules.

## Repository Layout

```text
├── README.md                 # Auto-rendered catalog table between catalog:start/end markers
├── catalog.json              # Auto-generated machine-readable catalog for tooling and llm-d.ai
├── DEMOS-POLICY.md           # Scope, capability taxonomy, simulator/live rules, lifecycle
├── Makefile                  # Root dispatcher (list, sim, new-demo, index, validate)
├── scripts/
│   ├── render_catalog.py     # Renders README.md catalog table + catalog.json from demos/*/demo.yaml
│   └── validate.py           # Enforces schema, required files, hygiene, catalog sync, and demo verify
├── templates/demo-starter/   # Copy-paste skeleton for every new demo
├── shared/                   # Shared assets only (Rule of Two applies before adding code)
└── demos/<slug>/             # Peer, self-contained demo directories
```

## Skills

Each playbook lives at `.agents/skills/<name>/SKILL.md`. Read the relevant skill before starting work.

| Skill | Use it when |
| :--- | :--- |
| [add-new-demo](.agents/skills/add-new-demo/SKILL.md) | Creating a new demo from scratch or migrating an existing demo into `demos/<slug>/` |
| [validate-demos](.agents/skills/validate-demos/SKILL.md) | Auditing schemas, secret/internal-link hygiene, catalog freshness, or smoke-testing simulators |

`.agents/` is the shared location for agent instructions and skills. `.claude/` is gitignored; Claude Code users can symlink `.agents/skills` into `.claude/skills`.

## Hard Rules

1. **Never put demo code at the repository root.** Every demo lives strictly inside `demos/<slug>/`.
2. **Never hand-edit the generated catalog table or `catalog.json`.** Edit `demos/<slug>/demo.yaml` and run `make index` (`python3 scripts/render_catalog.py`).
3. **Never copy another demo as a starter template.** Always scaffold from `templates/demo-starter/` via `make new-demo SLUG=<slug>`.
4. **Never vendor upstream `llm-d/llm-d` guides.** Reference upstream guides by URL or require a user-provided `LLMD_REPO` checkout.
5. **Never commit secrets, real cloud project IDs, internal corporate URLs, or personal usernames/namespaces.** Use environment variables with safe placeholders (such as `your-gcp-project-id`, `HF_TOKEN`).
6. **Keep default UI and catalog framing vendor-neutral.** Describe cloud, accelerator, engine, and model specifics under `modes.live.reference_environment` in `demo.yaml` and in the demo's Reference Environment section. When listing accelerators in prose, avoid vendor lock-in framing.
7. **Follow the Rule of Two for `shared/`.** Do not create shared UI shells or helper libraries unless at least two demos actively use them.
8. **Use DCO sign-off on all commits.** Every commit must be created with `git commit -s` (`Signed-off-by:`).
9. **Never run destructive cluster commands automatically.** Validation and local smoke tests must run completely offline without touching a live cluster.

## Standard Verification Loop

Before finishing any change, run:

```bash
make index
make validate
```

If you modified a demo that supports simulator mode, also verify that its simulator boots cleanly and responds on its declared port (see `.agents/skills/validate-demos/SKILL.md`).
