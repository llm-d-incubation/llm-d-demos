#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the root README.md catalog table and catalog.json from demos/*/demo.yaml."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMOS_DIR = REPO_ROOT / "demos"
README_PATH = REPO_ROOT / "README.md"
CATALOG_JSON_PATH = REPO_ROOT / "catalog.json"

START_MARKER = "<!-- catalog:start -->"
END_MARKER = "<!-- catalog:end -->"

STATUS_ORDER = {"active": 0, "experimental": 1, "archived": 2}


def load_demos() -> list[dict[str, Any]]:
    demos: list[dict[str, Any]] = []
    if not DEMOS_DIR.exists():
        return demos
    for entry in sorted(DEMOS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        manifest = entry / "demo.yaml"
        if not manifest.exists():
            continue
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = dict(data)
            data["_path"] = f"demos/{entry.name}"
            demos.append(data)
    demos.sort(
        key=lambda d: (
            STATUS_ORDER.get(str(d.get("status", "")), 99),
            str(d.get("id", "")),
        )
    )
    return demos


def _escape_cell(text: str) -> str:
    return " ".join(text.strip().split()).replace("|", "\\|")


def render_markdown_table(demos: list[dict[str, Any]]) -> str:
    if not demos:
        return (
            "_No demos registered yet. Run `make new-demo SLUG=<name>` to add the first demo._"
        )

    lines = [
        "| Demo | Capabilities | Local Simulator | Live Reference Stack | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for demo in demos:
        slug = demo["id"]
        title = _escape_cell(demo["title"])
        summary = _escape_cell(demo["summary"])
        demo_cell = f"[**{title}**](demos/{slug}/README.md)<br>{summary}"

        caps = demo.get("capabilities") or []
        caps_cell = ", ".join(f"`{c}`" for c in caps)

        sim = (demo.get("modes") or {}).get("simulator") or {}
        if sim.get("supported"):
            sim_cell = f"Yes (`make sim DEMO={slug}`)"
        else:
            sim_cell = "No"

        live = (demo.get("modes") or {}).get("live") or {}
        if live.get("supported"):
            ref = live.get("reference_environment") or {}
            platform = ref.get("platform", "Kubernetes")
            accels = ", ".join(ref.get("accelerators") or [])
            live_cell = _escape_cell(f"{platform} ({accels})" if accels else str(platform))
        else:
            live_cell = "Simulator only"

        status = str(demo.get("status", "experimental")).capitalize()
        lines.append(
            f"| {demo_cell} | {caps_cell} | {sim_cell} | {live_cell} | {status} |"
        )
    return "\n".join(lines)


def render_readme_content(demos: list[dict[str, Any]]) -> str:
    readme = README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"({re.escape(START_MARKER)}\n)(.*?)(\n{re.escape(END_MARKER)})",
        re.DOTALL,
    )
    if not pattern.search(readme):
        raise RuntimeError(
            f"Missing {START_MARKER} / {END_MARKER} markers in {README_PATH}"
        )
    table = render_markdown_table(demos)
    return pattern.sub(rf"\1{table}\3", readme, count=1)


def render_catalog_json(demos: list[dict[str, Any]]) -> str:
    payload = {
        "schema_version": 1,
        "demos": [
            {
                "id": d["id"],
                "path": d["_path"],
                "title": d["title"],
                "summary": " ".join(str(d["summary"]).split()),
                "status": d["status"],
                "capabilities": d.get("capabilities", []),
                "modes": d.get("modes", {}),
                "links": d.get("links", {}),
                "maintainers": d.get("maintainers", []),
            }
            for d in demos
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def check_catalog() -> list[str]:
    errors: list[str] = []
    demos = load_demos()
    expected_readme = render_readme_content(demos)
    actual_readme = README_PATH.read_text(encoding="utf-8")
    if actual_readme != expected_readme:
        errors.append(
            "README.md catalog table is out of date. Run 'make index' to regenerate."
        )
    expected_json = render_catalog_json(demos)
    if not CATALOG_JSON_PATH.exists():
        errors.append("catalog.json is missing. Run 'make index' to generate.")
    elif CATALOG_JSON_PATH.read_text(encoding="utf-8") != expected_json:
        errors.append("catalog.json is out of date. Run 'make index' to regenerate.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify README.md and catalog.json match demos/*/demo.yaml without modifying files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print a concise one-line summary of each registered demo.",
    )
    args = parser.parse_args()

    if args.list:
        demos = load_demos()
        if not demos:
            print("No demos registered yet.")
            return 0
        for d in demos:
            sim = bool(((d.get("modes") or {}).get("simulator") or {}).get("supported"))
            live = bool(((d.get("modes") or {}).get("live") or {}).get("supported"))
            print(
                f"{d['id']:<28} status={d['status']:<12} sim={str(sim):<5} live={str(live):<5} | {d['title']}"
            )
        return 0

    if args.check:
        errors = check_catalog()
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return 1
        print("OK: README.md and catalog.json are up to date.")
        return 0

    demos = load_demos()
    README_PATH.write_text(render_readme_content(demos), encoding="utf-8")
    CATALOG_JSON_PATH.write_text(render_catalog_json(demos), encoding="utf-8")
    print(f"Updated README.md and catalog.json ({len(demos)} demo(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
