#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate llm-d-demos catalog schemas, ownership, hygiene, sync, and demo verify targets."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMOS_DIR = REPO_ROOT / "demos"
TEMPLATE_DIR = REPO_ROOT / "templates" / "demo-starter"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import render_catalog  # noqa: E402

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAKE_TARGET_RE = re.compile(r"^([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)*)\s*:(?!=)")

REQUIRED_ROOT_FILES = [
    "README.md",
    "DEMOS-POLICY.md",
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "LICENSE",
    "DCO",
    "OWNERS",
    "Makefile",
]

REQUIRED_DEMO_FILES = ["demo.yaml", "README.md", "OWNERS", "Makefile"]

ALLOWED_STATUSES = {"active", "experimental", "archived"}

ALLOWED_CAPABILITIES = {
    "optimized-baseline",
    "precise-prefix-cache-routing",
    "predicted-latency-routing",
    "session-aware-routing",
    "flow-control",
    "turn-priority-fairness",
    "tiered-prefix-cache",
    "p2p-kv-cache-sharing",
    "pd-disaggregation",
    "wide-ep",
    "multimodal-serving",
    "omni-serving",
    "agentic-api",
    "workload-autoscaling",
    "fast-model-actuation",
    "pod-snapshot",
    "multi-model-routing",
    "batch-serving",
    "heterogeneous-accelerators",
    "reinforcement-learning",
}

BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".woff",
    ".woff2",
    ".pdf",
    ".mp4",
    ".webm",
    ".tar",
    ".gz",
    ".zip",
}

HYGIENE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("HuggingFace token", re.compile(r"\bhf_[A-Za-z0-9]{15,}\b")),
    ("Google API key", re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("OpenAI-style secret key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Private key block", re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")),
    (
        "Google-internal URL or CLA",
        re.compile(
            r"(?:corp\.google\.com|googlers\.com|g\.co/vulnz|goto\.google\.com|cla\.developers\.google\.com|opensource\.google/conduct)",
            re.IGNORECASE,
        ),
    ),
    (
        "Leftover private source reference",
        re.compile(
            r"(?:ai-on-gke/llm-d-demo|pytorch_conf_demo|llmd-tpu-harness|llmd-tpu-guides)",
            re.IGNORECASE,
        ),
    ),
]


def parse_make_targets(makefile: pathlib.Path) -> set[str]:
    targets: set[str] = set()
    for line in makefile.read_text(encoding="utf-8").splitlines():
        if line.startswith(("\t", "#", ".")):
            continue
        m = MAKE_TARGET_RE.match(line)
        if m:
            for tok in m.group(1).split():
                targets.add(tok)
    return targets


def parse_owners(owners_path: pathlib.Path) -> set[str]:
    data = yaml.safe_load(owners_path.read_text(encoding="utf-8"))
    handles: set[str] = set()
    if isinstance(data, dict):
        for key in ("approvers", "reviewers"):
            vals = data.get(key) or []
            if isinstance(vals, list):
                for item in vals:
                    if isinstance(item, str) and item.strip():
                        handles.add(item.strip())
    return handles


def validate_root(errors: list[str]) -> None:
    for rel in REQUIRED_ROOT_FILES:
        if not (REPO_ROOT / rel).is_file():
            errors.append(f"Missing required root file: {rel}")

    for rel in REQUIRED_DEMO_FILES:
        if not (TEMPLATE_DIR / rel).is_file():
            errors.append(f"Missing required starter template file: templates/demo-starter/{rel}")

    root_owners = parse_owners(REPO_ROOT / "OWNERS") if (REPO_ROOT / "OWNERS").is_file() else set()
    if not root_owners:
        errors.append("Root OWNERS must list at least one approver or reviewer.")


def validate_demo(demo_dir: pathlib.Path, errors: list[str], run_verify: bool) -> None:
    slug = demo_dir.name
    prefix = f"demos/{slug}"

    if not SLUG_RE.match(slug):
        errors.append(f"{prefix}: directory name must be lowercase kebab-case ({SLUG_RE.pattern})")

    for req in REQUIRED_DEMO_FILES:
        if not (demo_dir / req).is_file():
            errors.append(f"{prefix}: missing required file '{req}'")

    manifest_path = demo_dir / "demo.yaml"
    if not manifest_path.is_file():
        return

    try:
        data: Any = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        errors.append(f"{prefix}/demo.yaml: invalid YAML: {exc}")
        return

    if not isinstance(data, dict):
        errors.append(f"{prefix}/demo.yaml: top-level document must be a mapping")
        return

    if data.get("schema_version") != 1:
        errors.append(f"{prefix}/demo.yaml: schema_version must be 1")

    if data.get("id") != slug:
        errors.append(f"{prefix}/demo.yaml: id '{data.get('id')}' must match directory name '{slug}'")

    for field in ("title", "summary"):
        val = data.get(field)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{prefix}/demo.yaml: '{field}' must be a non-empty string")

    status = data.get("status")
    if status not in ALLOWED_STATUSES:
        errors.append(
            f"{prefix}/demo.yaml: status '{status}' must be one of {sorted(ALLOWED_STATUSES)}"
        )

    caps = data.get("capabilities")
    if not isinstance(caps, list) or not caps:
        errors.append(f"{prefix}/demo.yaml: 'capabilities' must be a non-empty list")
    else:
        invalid_caps = [c for c in caps if c not in ALLOWED_CAPABILITIES]
        if invalid_caps:
            errors.append(
                f"{prefix}/demo.yaml: unknown capabilities {invalid_caps}; see DEMOS-POLICY.md"
            )

    modes = data.get("modes")
    sim_supported = False
    if not isinstance(modes, dict):
        errors.append(f"{prefix}/demo.yaml: 'modes' must be a mapping with 'simulator' and 'live'")
    else:
        sim = modes.get("simulator")
        if not isinstance(sim, dict) or not isinstance(sim.get("supported"), bool):
            errors.append(f"{prefix}/demo.yaml: modes.simulator.supported must be a boolean")
        elif sim["supported"]:
            sim_supported = True
            if not isinstance(sim.get("command"), str) or not sim["command"].strip():
                errors.append(f"{prefix}/demo.yaml: modes.simulator.command is required when supported=true")
            port = sim.get("port")
            if not isinstance(port, int) or not (1 <= port <= 65535):
                errors.append(f"{prefix}/demo.yaml: modes.simulator.port must be an integer 1..65535")

        live = modes.get("live")
        if not isinstance(live, dict) or not isinstance(live.get("supported"), bool):
            errors.append(f"{prefix}/demo.yaml: modes.live.supported must be a boolean")
        elif live["supported"]:
            ref = live.get("reference_environment")
            if not isinstance(ref, dict):
                errors.append(
                    f"{prefix}/demo.yaml: modes.live.reference_environment is required when supported=true"
                )
            else:
                for ref_key in ("platform", "accelerators", "engines", "model"):
                    if not ref.get(ref_key):
                        errors.append(
                            f"{prefix}/demo.yaml: modes.live.reference_environment.{ref_key} must be non-empty"
                        )

        if isinstance(sim, dict) and isinstance(live, dict):
            if not sim.get("supported") and not live.get("supported"):
                errors.append(
                    f"{prefix}/demo.yaml: at least one of modes.simulator or modes.live must be supported"
                )

    maintainers = data.get("maintainers")
    if not isinstance(maintainers, list) or not maintainers:
        errors.append(f"{prefix}/demo.yaml: 'maintainers' must be a non-empty list of GitHub handles")
    else:
        owners_path = demo_dir / "OWNERS"
        owners_handles = parse_owners(owners_path) if owners_path.is_file() else set()
        for m in maintainers:
            if not isinstance(m, str) or not m.strip() or m.startswith("@") or m == "your-github-handle":
                errors.append(
                    f"{prefix}/demo.yaml: invalid maintainer '{m}' (use plain GitHub handle without @)"
                )
            elif owners_handles and m not in owners_handles:
                errors.append(
                    f"{prefix}/demo.yaml: maintainer '{m}' must also be listed in {prefix}/OWNERS"
                )

    makefile = demo_dir / "Makefile"
    if makefile.is_file():
        targets = parse_make_targets(makefile)
        required_targets = {"help", "verify"}
        if sim_supported:
            required_targets.add("simulate")
        missing_targets = sorted(required_targets - targets)
        if missing_targets:
            errors.append(f"{prefix}/Makefile: missing required target(s): {', '.join(missing_targets)}")

    if run_verify and makefile.is_file():
        proc = subprocess.run(
            ["make", "-C", str(demo_dir), "verify"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            errors.append(
                f"{prefix}: 'make -C {prefix} verify' failed (exit {proc.returncode}):\n"
                f"{proc.stdout}{proc.stderr}".strip()
            )


def scan_hygiene(errors: list[str]) -> None:
    skip_paths = {
        "scripts/validate.py",
        ".agents/skills/validate-demos/SKILL.md",
    }
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [
            d
            for d in dirs
            if d not in {".git", "__pycache__", "node_modules", ".venv", "venv"}
        ]
        for fn in sorted(files):
            path = pathlib.Path(root) / fn
            rel = str(path.relative_to(REPO_ROOT))
            if rel in skip_paths or path.suffix.lower() in BINARY_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                for label, rx in HYGIENE_PATTERNS:
                    m = rx.search(line)
                    if m:
                        errors.append(f"Hygiene violation ({label}) in {rel}:{lineno} -> '{m.group(0)}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-demo-verify",
        action="store_true",
        help="Skip running per-demo `make -C demos/<slug> verify` targets.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    validate_root(errors)

    demo_dirs: list[pathlib.Path] = []
    if DEMOS_DIR.is_dir():
        demo_dirs = [
            p for p in sorted(DEMOS_DIR.iterdir()) if p.is_dir() and not p.name.startswith(".")
        ]

    for demo_dir in demo_dirs:
        validate_demo(demo_dir, errors, run_verify=not args.skip_demo_verify)

    scan_hygiene(errors)
    errors.extend(render_catalog.check_catalog())

    if errors:
        print(f"VALIDATION FAILED ({len(errors)} issue(s)):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"OK: validated root framework and {len(demo_dirs)} demo(s) with 0 issues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
