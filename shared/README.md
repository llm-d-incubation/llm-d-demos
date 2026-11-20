# Shared Assets & Utilities

This directory follows the **Rule of Two** from [DEMOS-POLICY.md](../DEMOS-POLICY.md):

- **Do not** build a shared demo runtime or frontend shell speculatively. Each demo under `demos/<slug>/` should stay self-contained and free to use whatever stack fits best.
- **Do** place official logos, reusable artwork, or small helper modules here once **two or more demos** actively depend on them.

## Contents

- `assets/`: Official `llm-d` and ecosystem SVG marks available for demos to copy or reference. See `assets/README.md` for attribution and licensing.
