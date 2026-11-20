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

"""89_kv_storage_crossreplica.py: warm replica A, then hit replica B cold with the same sessions.

The scale-out / restart case for the shared-storage KV tier (ENABLE_KV_STORAGE_TIER=1): replica A has
served the agents and written their KV through to storage; replica B is a fresh pod (empty HBM and
host pool) that can either restore from storage or recompute.

Per leg, with a fresh session set each time:
  warm   S sessions x 1 turn to A (concurrency C), then wait until A's fs write counter is flat
  seq    the same sessions to B, one at a time
  burst  new sessions warmed on A, then all S sent to B at once (what a new replica sees on scale-out)
Legs: tier (B restores from storage) and notier (kv_transfer_params max_load_tokens=0 -> B recomputes).
Prints one JSON line per leg/phase: TTFT p50/p95/max on B and B's counter deltas (HBM hits, external
hits, bytes read from the fs tier, lookup wait).

Example (port-forward two specific pods, not the Service):
  kubectl -n llm-d-gpu port-forward pod/<warm-pod> 18201:8000 &
  kubectl -n llm-d-gpu port-forward pod/<cold-pod> 18202:8000 &
  python3 scripts/89_kv_storage_crossreplica.py --warm-url http://127.0.0.1:18201 --cold-url http://127.0.0.1:18202 \
      --model RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic --sessions 24 --paras 160 --out results/kv-storage-$(date +%Y%m%d-%H%M%S).json
"""

import argparse
import json
import random
import threading
import time
import urllib.request

NOTIER = {"kv_transfer_params": {"max_load_tokens": 0, "max_offload_tokens": 0}}
QUESTION = "Which note mentions the highest run id? One sentence."


def prefix(seed, paras=30):
    rnd = random.Random(seed)
    topics = [
        "tensor parallelism",
        "KV cache paging",
        "speculative decoding",
        "FP8 quantization",
        "expert parallelism",
        "prefix caching",
        "continuous batching",
        "chunked prefill",
    ]
    out = [f"Agent session {seed:08x}: you are a Hermes coding agent. Answer in one short sentence.\n"]
    for k in range(paras):
        topic = topics[(seed * 7 + k) % len(topics)]
        out.append(
            f"Workspace note {k + 1}: the team documented {topic} behaviour under agentic load; observed that {topic} "
            f"interacts with request fan-out, retry budgets and session affinity; record findings as short "
            f"bullets, prefer measured numbers, cite run id W{(seed + k) % 97:02d}-{rnd.randint(0, 999):03d}."
        )
    return "\n".join(out)


def chat(url, model, system, extra=None, max_tokens=32):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": QUESTION}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if (ch.get("delta") or {}).get("content") and ttft is None:
                    ttft = time.perf_counter() - t0
    return (ttft or (time.perf_counter() - t0)) * 1000, usage


def metrics(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10) as resp:
            txt = resp.read().decode()
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        k, v = parts[0], parts[1]
        try:
            out[k] = out.get(k, 0.0) + float(v)
        except ValueError:
            pass
    return out


def pick(metric_map, *subs):
    """Sum of all series whose name contains every substring."""
    return sum(v for k, v in metric_map.items() if all(s in k for s in subs))


def pct(values, quantile):
    if not values:
        return None
    sorted_vals = sorted(values)
    return round(sorted_vals[min(len(sorted_vals) - 1, int(quantile * len(sorted_vals)))], 1)


def run_many(url, model, sessions, conc, extra=None, max_tokens=32):
    """Returns list of ttft ms in completion order."""
    out = []
    lock = threading.Lock()
    next_idx = 0

    def worker():
        nonlocal next_idx
        while True:
            with lock:
                if next_idx >= len(sessions):
                    return
                session_prompt = sessions[next_idx]
                next_idx += 1
            try:
                ttft, _ = chat(url, model, session_prompt, extra, max_tokens=max_tokens)
                with lock:
                    out.append(ttft)
            except Exception:
                with lock:
                    out.append(None)

    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def wait_fs_flat(url, baseline_bytes=0.0, timeout=600):
    """Wait until the warm replica's fs write byte counter advances past baseline and stabilizes."""
    last = baseline_bytes
    saw_advance = False
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur = pick(metrics(url), "kv_offload_tiering_write_bytes_total", "fs")
        if cur > baseline_bytes:
            saw_advance = True
        if saw_advance and cur == last:
            return cur
        last = cur
        time.sleep(5)
    return last


def deltas(m0, m1):
    def diff(*subs):
        return pick(m1, *subs) - pick(m0, *subs)

    return {
        "hbm_hit": diff("vllm:prefix_cache_hits_total"),
        "ext_q": diff("vllm:external_prefix_cache_queries_total"),
        "ext_hit": diff("vllm:external_prefix_cache_hits_total"),
        "cpu2gpu_MB": round(diff("kv_offload_total_bytes_total", "CPU_to_GPU") / 1e6, 1),
        "fs_read_MB": round(diff("kv_offload_tiering_read_bytes_total", "fs") / 1e6, 1),
        "fs_read_s": round(diff("kv_offload_tiering_read_time_total", "fs"), 2),
        "fs_lookup_async_s": round(diff("kv_offload_tiering_lookup_async_delay_seconds_sum", "fs"), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-url", required=True)
    ap.add_argument("--cold-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--paras", type=int, default=160)
    ap.add_argument("--warm-conc", type=int, default=8)
    ap.add_argument("--legs", default="tier,notier")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = []
    seed = args.seed
    for leg in args.legs.split(","):
        extra = NOTIER if leg == "notier" else None
        for phase in ("seq", "burst"):
            seed += 1
            sess = [prefix(seed * 1000 + i, args.paras) for i in range(args.sessions)]
            # Warm on replica A (pass extra=NOTIER in the notier control leg so A does not write through)
            wb0 = pick(metrics(args.warm_url), "kv_offload_tiering_write_bytes_total", "fs")
            t0 = time.time()
            warm = run_many(args.warm_url, args.model, sess, args.warm_conc, extra)
            wb = wait_fs_flat(args.warm_url, baseline_bytes=wb0) if leg == "tier" else wb0
            warm_s = time.time() - t0
            # Cold on replica B
            mb0 = metrics(args.cold_url)
            t1 = time.time()
            cold = run_many(args.cold_url, args.model, sess, 1 if phase == "seq" else args.sessions, extra)
            cold_s = time.time() - t1
            mb1 = metrics(args.cold_url)
            rec = {
                "label": args.label,
                "leg": leg,
                "phase": phase,
                "sessions": args.sessions,
                "paras": args.paras,
                "warm_ttft_p50": pct([x for x in warm if x], 0.5),
                "warm_s": round(warm_s, 1),
                "fs_written_phase_GB": round((wb - wb0) / 1e9, 2),
                "fs_written_total_GB": round(wb / 1e9, 2),
                "cold_ttft_p50": pct([x for x in cold if x], 0.5),
                "cold_ttft_p95": pct([x for x in cold if x], 0.95),
                "cold_ttft_max": pct([x for x in cold if x], 1.0),
                "cold_wall_s": round(cold_s, 1),
                "cold_errors": sum(1 for x in cold if x is None),
                "cold_B": deltas(mb0, mb1),
            }
            print(json.dumps(rec), flush=True)
            results.append(rec)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
