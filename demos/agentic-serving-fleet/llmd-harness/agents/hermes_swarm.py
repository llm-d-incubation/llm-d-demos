#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Hermes Agent Fleet & High-Scale Stress Generator for the llm-d Benchmark.

Simulates 1,000s to 10,000s of Hermes Agents (`nousanalysis/research/hermes-agent`)
running across GKE Agent Substrate (AX / Substrate) or as an in-cluster async swarm.

Key architectural properties modeled:
  1. Hermes Frozen Snapshot Pattern (`MEMORY.md` + `USER.md` + `SKILLS`):
     Injected once into a static System Prompt at session start and never
     mutated mid-session, guaranteeing high prefix-cache hit rates on the
     optimized TPU v7x (`vllm-torchtpu`) pool via `llm-d` `prefix-cache-scorer`.
  2. Multi-Turn Agentic Loop (`x-llm-d-workload: multi-turn`):
     Each multi-turn Actor runs N turns (coding / tool execution / debugging),
     accumulating conversation history + tool outputs and setting:
       - `x-llm-d-workload: multi-turn`
       - `x-llm-d-inference-fairness-id: hermes-actor-<id>`
       - `x-llm-d-inference-objective: agentic-multi-turn`
  3. One-Shot Agentic Tasks (`x-llm-d-workload: one-shot`):
     Each one-shot Actor sends a single-turn triage / commit-summary / PR-label
     request routed by `llm-d` to the GPU serving stack with:
       - `x-llm-d-workload: one-shot`
       - `x-llm-d-inference-fairness-id: hermes-actor-<id>`
       - `x-llm-d-inference-objective: oneshot-fast`
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import http.client
import json
import os
import random
import sys
import threading
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

import urllib.request
import uuid

_thread_local = threading.local()


def scrape_vllm_prom(metrics_urls: List[str]) -> Dict[str, float]:
    """Scrapes and sums vLLM Prometheus counters across one or more pod /metrics URLs."""
    totals = {
        "prompt_tokens": 0.0,
        "gen_tokens": 0.0,
        "requests": 0.0,
        "e2e_sum_s": 0.0,
        "e2e_count": 0.0,
        "ttft_sum_s": 0.0,
        "ttft_count": 0.0,
        "queue_sum_s": 0.0,
        "queue_count": 0.0,
        "cache_hits": 0.0,
        "cache_queries": 0.0,
    }
    for url in metrics_urls:
        if not url:
            continue
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or not line.startswith("vllm:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                name = parts[0].split("{", 1)[0]
                try:
                    val = float(parts[-1])
                except ValueError:
                    continue
                if name == "vllm:prompt_tokens_total":
                    totals["prompt_tokens"] += val
                elif name == "vllm:generation_tokens_total":
                    totals["gen_tokens"] += val
                elif name == "vllm:e2e_request_latency_seconds_sum":
                    totals["e2e_sum_s"] += val
                elif name == "vllm:e2e_request_latency_seconds_count":
                    totals["e2e_count"] += val
                    totals["requests"] += val
                elif name == "vllm:time_to_first_token_seconds_sum":
                    totals["ttft_sum_s"] += val
                elif name == "vllm:time_to_first_token_seconds_count":
                    totals["ttft_count"] += val
                elif name == "vllm:request_queue_time_seconds_sum":
                    totals["queue_sum_s"] += val
                elif name == "vllm:request_queue_time_seconds_count":
                    totals["queue_count"] += val
                elif name == "vllm:prefix_cache_hits_total":
                    totals["cache_hits"] += val
                elif name == "vllm:prefix_cache_queries_total":
                    totals["cache_queries"] += val
        except Exception as exc:
            print(f"[warn] failed to scrape {url}: {exc}", flush=True)
    return totals


def compute_prom_delta(before: Dict[str, float], after: Dict[str, float], elapsed_s: float) -> Dict[str, float]:
    d_req = max(0.0, after["requests"] - before["requests"])
    d_ptok = max(0.0, after["prompt_tokens"] - before["prompt_tokens"])
    d_gtok = max(0.0, after["gen_tokens"] - before["gen_tokens"])
    d_ttft_s = max(0.0, after["ttft_sum_s"] - before["ttft_sum_s"])
    d_ttft_n = max(0.0, after["ttft_count"] - before["ttft_count"])
    d_e2e_s = max(0.0, after["e2e_sum_s"] - before["e2e_sum_s"])
    d_e2e_n = max(0.0, after["e2e_count"] - before["e2e_count"])
    d_q_s = max(0.0, after["queue_sum_s"] - before["queue_sum_s"])
    d_q_n = max(0.0, after["queue_count"] - before["queue_count"])
    d_hits = max(0.0, after["cache_hits"] - before["cache_hits"])
    d_queries = max(0.0, after["cache_queries"] - before["cache_queries"])
    return {
        "requests": int(round(d_req)),
        "prompt_tokens": int(round(d_ptok)),
        "generation_tokens": int(round(d_gtok)),
        "cached_tokens": int(round(d_hits)),
        "queried_tokens": int(round(d_queries)),
        "recomputed_tokens": int(round(max(0.0, d_queries - d_hits))),
        "prefix_cache_hit_pct": round((d_hits / d_queries * 100.0), 2) if d_queries > 0 else 0.0,
        "ttft_mean_ms": round((d_ttft_s / d_ttft_n * 1000.0), 2) if d_ttft_n > 0 else 0.0,
        "e2e_mean_ms": round((d_e2e_s / d_e2e_n * 1000.0), 2) if d_e2e_n > 0 else 0.0,
        "queue_mean_ms": round((d_q_s / d_q_n * 1000.0), 2) if d_q_n > 0 else 0.0,
        "prefill_tok_per_s": round(d_ptok / max(0.001, elapsed_s), 1),
        "recomputed_tok_per_s": round(max(0.0, d_queries - d_hits) / max(0.001, elapsed_s), 1),
        "decode_tok_per_s": round(d_gtok / max(0.001, elapsed_s), 1),
    }


def build_frozen_snapshot_system_prompt(
    cohort_id: int, target_tokens: int = 2048, bust_cache_nonce: str = ""
) -> str:
    """Builds a deterministic Hermes Frozen Snapshot system prompt (`MEMORY.md` + `USER.md`).

    Agents within the same workspace cohort (`cohort_id`) share this exact prefix,
    and every multi-turn Actor preserves this exact prefix across all its turns.
    When `bust_cache_nonce` is non-empty, it is prepended at token 0 so every
    PagedAttention block hash is unique (forcing 0% prefix-cache hit rate).
    """
    nonce_prefix = f"[UNCACHED_RUN_NONCE={bust_cache_nonce}]\n" if bust_cache_nonce else ""
    base_header = (
        f"{nonce_prefix}"
        f"You are Hermes Agent (Cohort #{cohort_id:03d}), an autonomous software engineering "
        f"actor running inside a GKE Agent Substrate gVisor sandbox.\n\n"
        f"=== FROZEN SNAPSHOT: MEMORY.md (Immutable Session Snapshot) ===\n"
        f"- Repository: github.com/llm-d/llm-d (Kubernetes-native distributed LLM serving)\n"
        f"- Target Cluster: GKE Standard with TPU v7x (Ironwood) + NVIDIA GPU node pools\n"
        f"- Routing Architecture: Gateway API Inference Extension (GAIE) + Endpoint Picker (EPP)\n"
        f"- Active Model: google/gemma-4-26B-A4B-it (26B Total Params, 4B Active MoE)\n\n"
        f"=== FROZEN SNAPSHOT: USER.md ===\n"
        f"- Role: Staff Infrastructure Engineer\n"
        f"- Style: Concise, production-grade YAML/Go/Python, deterministic verification.\n\n"
        f"=== REGISTERED HERMES SKILLS & REPO CONTEXT ===\n"
    )
    target_chars = max(512, target_tokens * 4)
    block = (
        f"[Skill Cohort-{cohort_id:03d}] Module `epp.flow_control`: enforces per-tenant fairness "
        f"queues (`FlowKey = FairnessID + PriorityBand`) and prefix-cache affinity scoring across "
        f"disaggregated and unified TPU/GPU inference pools. "
    )
    repeats = max(1, (target_chars - len(base_header)) // len(block))
    return base_header + (block * repeats)


ONE_SHOT_PROMPTS = [
    "Summarize in 2 sentences why routing one-shot stateless requests to GPUs prevents KV cache eviction on multi-turn TPU pools.",
    "Generate a conventional git commit title for adding x-llm-d-workload header routing between TPU v7x and GPU InferencePools.",
    "Classify this agent task as ONE-SHOT or MULTI-TURN: 'Format the JSON output of a completed test run into a Markdown table.'",
    "Write a 3-line Kubernetes label selector matching pods with llm-d.ai/workload=one-shot and llm-d.ai/accelerator-vendor=nvidia.",
]

MULTI_TURN_USER_STEPS = [
    "Turn 1: Inspect the Envoy HTTPRoute header matcher for `x-llm-d-workload: multi-turn` and propose a 3-line verification curl command.",
    "Turn 2: Tool output showed `x-llm-d-routed-pool: tpu-v7x-torchtpu-multiturn` and TTFT of 140ms. Now write the Python snippet to attach `x-llm-d-inference-fairness-id` per Actor.",
    "Turn 3: Now add an `InferenceObjective` resource named `agentic-multi-turn` with priority 100 targeting `llmd-tpu-pool`.",
    "Turn 4: Verify how Hermes Agent's frozen `MEMORY.md` snapshot keeps prefix-cache hit rates >90% across turns 1..4 on TPU v7x HBM.",
    "Turn 5: Provide a final 3-bullet readiness sign-off for the benchmark demo.",
]


@dataclass
class RequestRecord:
    actor_id: int
    workload_type: str  # "multi-turn" or "one-shot"
    turn_idx: int
    routed_pool: str
    status_code: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None


@dataclass
class TelemetryStore:
    records: List[RequestRecord] = field(default_factory=list)
    active_multi_turn_actors: int = 0
    active_oneshot_actors: int = 0
    in_flight_tpu: int = 0
    in_flight_gpu: int = 0
    peak_in_flight: int = 0
    start_time: float = field(default_factory=time.time)

    def summary_dict(self) -> Dict:
        elapsed = max(0.001, time.time() - self.start_time)
        out = {
            "elapsed_s": round(elapsed, 2),
            "peak_concurrent_in_flight": self.peak_in_flight,
            "pools": {},
        }
        for wtype in ("multi-turn", "one-shot"):
            sub = [r for r in self.records if r.workload_type == wtype]
            ok_sub = [r for r in sub if r.status_code == 200]
            lats = sorted(r.latency_ms for r in ok_sub)
            in_toks = sum(r.prompt_tokens for r in ok_sub)
            out_toks = sum(r.completion_tokens for r in ok_sub)
            routed_counts: Dict[str, int] = {}
            for r in ok_sub:
                routed_counts[r.routed_pool] = routed_counts.get(r.routed_pool, 0) + 1
            out["pools"][wtype] = {
                "total_requests": len(sub),
                "ok_requests": len(ok_sub),
                "error_requests": len(sub) - len(ok_sub),
                "qps": round(len(ok_sub) / elapsed, 2),
                "input_tok_per_s": round(in_toks / elapsed, 1),
                "output_tok_per_s": round(out_toks / elapsed, 1),
                "total_input_tokens": in_toks,
                "total_output_tokens": out_toks,
                "latency_p50_ms": round(lats[len(lats) // 2], 1) if lats else 0.0,
                "latency_p90_ms": round(lats[int(len(lats) * 0.9)], 1) if lats else 0.0,
                "latency_p99_ms": round(lats[int(len(lats) * 0.99)], 1) if lats else 0.0,
                "routed_backends": routed_counts,
            }
        return out


def _get_conn(host: str, port: int, timeout_s: float) -> http.client.HTTPConnection:
    conn = getattr(_thread_local, "conn", None)
    conn_key = getattr(_thread_local, "conn_key", None)
    if conn is None or conn_key != (host, port):
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        _thread_local.conn = conn
        _thread_local.conn_key = (host, port)
    return conn


def _sync_post_chat(
    endpoint: str,
    model: str,
    messages: List[Dict[str, str]],
    headers: Dict[str, str],
    max_tokens: int,
    timeout_s: float = 180.0,
) -> tuple[int, str, Dict, float, Optional[str]]:
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/v1/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        **headers,
    }
    t0 = time.perf_counter()
    for attempt in range(2):
        try:
            conn = _get_conn(host, port, timeout_s)
            conn.request("POST", path, body=payload, headers=req_headers)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8")
            dt_ms = (time.perf_counter() - t0) * 1000.0
            routed_pool = resp.getheader("x-llm-d-routed-pool") or "unknown-pool"
            if resp.status != 200:
                return resp.status, routed_pool, {}, dt_ms, body[:200]
            return resp.status, routed_pool, json.loads(body), dt_ms, None
        except Exception as exc:
            _thread_local.conn = None
            if attempt == 1:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                return 500, "error", {}, dt_ms, str(exc)
    return 500, "error", {}, 0.0, "exhausted"


async def run_hermes_actor(
    actor_id: int,
    is_multi_turn: bool,
    endpoint: str,
    model: str,
    turns: int,
    cohorts: int,
    prefix_tokens: int,
    max_tokens: int,
    no_prefix_cache: bool,
    semaphore: asyncio.Semaphore,
    store: TelemetryStore,
) -> None:
    fairness_id = f"hermes-actor-{actor_id:05d}"
    if is_multi_turn:
        cohort_id = actor_id % max(1, cohorts)
        sys_prompt = build_frozen_snapshot_system_prompt(cohort_id, prefix_tokens)
        messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]
        headers = {
            "x-llm-d-workload": "multi-turn",
            "x-llm-d-inference-fairness-id": fairness_id,
            "x-llm-d-inference-objective": "agentic-multi-turn",
            "x-hermes-actor-id": fairness_id,
            "x-hermes-cohort-id": f"cohort-{cohort_id:03d}",
        }
        store.active_multi_turn_actors += 1
        try:
            for t in range(turns):
                if no_prefix_cache:
                    messages[0] = {
                        "role": "system",
                        "content": build_frozen_snapshot_system_prompt(
                            cohort_id, prefix_tokens, bust_cache_nonce=f"{actor_id}-{t}-{uuid.uuid4().hex[:12]}"
                        ),
                    }
                step_prompt = MULTI_TURN_USER_STEPS[t % len(MULTI_TURN_USER_STEPS)]
                messages.append({"role": "user", "content": step_prompt})
                async with semaphore:
                    store.in_flight_tpu += 1
                    cur_inf = store.in_flight_tpu + store.in_flight_gpu
                    if cur_inf > store.peak_in_flight:
                        store.peak_in_flight = cur_inf
                    try:
                        status, routed_pool, data, dt_ms, err = await asyncio.to_thread(
                            _sync_post_chat, endpoint, model, list(messages), headers, max_tokens
                        )
                    finally:
                        store.in_flight_tpu -= 1
                usage = data.get("usage") or {}
                p_tok = int(usage.get("prompt_tokens") or (prefix_tokens + t * 180))
                c_tok = int(usage.get("completion_tokens") or max_tokens)
                store.records.append(
                    RequestRecord(
                        actor_id=actor_id,
                        workload_type="multi-turn",
                        turn_idx=t + 1,
                        routed_pool=routed_pool,
                        status_code=status,
                        latency_ms=dt_ms,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        error=err,
                    )
                )
                if status == 200 and data.get("choices"):
                    reply = data["choices"][0].get("message", {}).get("content", "OK")
                    messages.append({"role": "assistant", "content": reply})
                else:
                    break
        finally:
            store.active_multi_turn_actors -= 1
    else:
        prompt = ONE_SHOT_PROMPTS[actor_id % len(ONE_SHOT_PROMPTS)]
        messages = [{"role": "user", "content": prompt}]
        headers = {
            "x-llm-d-workload": "one-shot",
            "x-llm-d-inference-fairness-id": fairness_id,
            "x-llm-d-inference-objective": "oneshot-fast",
            "x-hermes-actor-id": fairness_id,
        }
        store.active_oneshot_actors += 1
        try:
            async with semaphore:
                store.in_flight_gpu += 1
                cur_inf = store.in_flight_tpu + store.in_flight_gpu
                if cur_inf > store.peak_in_flight:
                    store.peak_in_flight = cur_inf
                try:
                    status, routed_pool, data, dt_ms, err = await asyncio.to_thread(
                        _sync_post_chat, endpoint, model, messages, headers, max_tokens
                    )
                finally:
                    store.in_flight_gpu -= 1
            usage = data.get("usage") or {}
            p_tok = int(usage.get("prompt_tokens") or 48)
            c_tok = int(usage.get("completion_tokens") or max_tokens)
            store.records.append(
                RequestRecord(
                    actor_id=actor_id,
                    workload_type="one-shot",
                    turn_idx=1,
                    routed_pool=routed_pool,
                    status_code=status,
                    latency_ms=dt_ms,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    error=err,
                )
            )
        finally:
            store.active_oneshot_actors -= 1


async def live_reporter(store: TelemetryStore, total_agents: int, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(3.0)
        s = store.summary_dict()
        mt = s["pools"]["multi-turn"]
        os_p = s["pools"]["one-shot"]
        print(
            f"[{s['elapsed_s']:>6.1f}s | in-flight TPU={store.in_flight_tpu:>3} GPU={store.in_flight_gpu:>3}] "
            f"TPU Multi-Turn: {mt['ok_requests']:>5}/{mt['total_requests']:<5} OK "
            f"({mt['qps']:>5.1f} req/s, {mt['input_tok_per_s']:>8.0f} in tok/s, {mt['output_tok_per_s']:>6.0f} out tok/s, P50={mt['latency_p50_ms']:>6.0f}ms) | "
            f"GPU One-Shot: {os_p['ok_requests']:>4}/{os_p['total_requests']:<4} OK "
            f"({os_p['qps']:>5.1f} req/s, P50={os_p['latency_p50_ms']:>6.0f}ms)",
            flush=True,
        )


async def main_async(args: argparse.Namespace) -> int:
    loop = asyncio.get_running_loop()
    # Critical: override Python's default 20-32 thread cap so high concurrency actually dispatches in parallel
    loop.set_default_executor(ThreadPoolExecutor(max_workers=max(256, args.concurrency + 64)))

    metrics_urls = [u.strip() for u in args.metrics_endpoints.split(",") if u.strip()]
    prom_before = scrape_vllm_prom(metrics_urls) if metrics_urls else {}

    store = TelemetryStore()
    semaphore = asyncio.Semaphore(args.concurrency)
    stop_event = asyncio.Event()

    print(
        f"==> Spawning {args.num_agents} Hermes Agent Actors "
        f"(concurrency={args.concurrency}, multi_turn_ratio={args.multi_turn_ratio:.0%}, "
        f"turns={args.turns}, prefix_caching={'OFF (busted)' if args.no_prefix_cache else 'ON'}, model={args.model})\n"
        f"    Unified Endpoint: {args.endpoint}",
        flush=True,
    )

    reporter_task = asyncio.create_task(live_reporter(store, args.num_agents, stop_event))

    tasks = []
    for i in range(1, args.num_agents + 1):
        is_mt = (random.random() < args.multi_turn_ratio)
        tasks.append(
            asyncio.create_task(
                run_hermes_actor(
                    actor_id=i,
                    is_multi_turn=is_mt,
                    endpoint=args.endpoint,
                    model=args.model,
                    turns=args.turns,
                    cohorts=args.cohorts,
                    prefix_tokens=args.prefix_tokens,
                    max_tokens=args.max_tokens,
                    no_prefix_cache=args.no_prefix_cache,
                    semaphore=semaphore,
                    store=store,
                )
            )
        )

    await asyncio.gather(*tasks)
    stop_event.set()
    await reporter_task

    final = store.summary_dict()
    final["prefix_caching_enabled"] = not args.no_prefix_cache
    if metrics_urls:
        prom_after = scrape_vllm_prom(metrics_urls)
        final["vllm_prometheus_delta"] = compute_prom_delta(prom_before, prom_after, final["elapsed_s"])

    print("\n================ FINAL BENCHMARK TELEMETRY ================")
    print(json.dumps(final, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        print(f"Saved structured summary to {args.output_json}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hermes Agent Swarm & Substrate Stress Runner")
    p.add_argument("--endpoint", required=True, help="Unified llm-d Gateway /v1/chat/completions URL")
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it", help="Served model name")
    p.add_argument("--num-agents", type=int, default=1000, help="Total Hermes Actors to spawn")
    p.add_argument("--concurrency", type=int, default=256, help="Max concurrent in-flight requests")
    p.add_argument("--multi-turn-ratio", type=float, default=0.6, help="Fraction of multi-turn agents (0..1)")
    p.add_argument("--turns", type=int, default=5, help="Turns per multi-turn Hermes agent session")
    p.add_argument("--cohorts", type=int, default=20, help="Number of shared Frozen-Snapshot prefix cohorts")
    p.add_argument("--prefix-tokens", type=int, default=2048, help="Frozen Snapshot system prompt token length")
    p.add_argument("--max-tokens", type=int, default=64, help="Max completion tokens per request")
    p.add_argument("--no-prefix-cache", action="store_true", help="Bust prefix cache on every request (0% hit baseline)")
    p.add_argument("--metrics-endpoints", default="", help="Comma-separated vLLM /metrics URLs to scrape before/after")
    p.add_argument("--output-json", default="", help="Optional file path to save summary JSON")
    return p.parse_args()


if __name__ == "__main__":
    random.seed(42)
    sys.exit(asyncio.run(main_async(parse_args())))
