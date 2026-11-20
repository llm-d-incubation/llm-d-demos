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
"""88_kv_tiering_bench.py: A/B for the GPU pool's host KV tier (ENABLE_KV_TIERING=1), same pods.

Workload shaped like the Hermes agents: S sessions, each with its own ~P-token system prefix, one turn
per session per round (so every prefix is evicted from HBM before its next turn), R rounds, C concurrent.
Streaming chat completions, 48 output tokens.

Legs, run back to back against the same endpoint:
  notier   kv_transfer_params {"max_load_tokens": 0, "max_offload_tokens": 0}  -> recompute every turn
  tier     normal requests                                                      -> host tier restores evicted prefixes

Prints one JSON line per leg: wall time, req/s, output tok/s, TTFT p50/p95 for cold (round 1) and returning
(rounds >= 2) turns, and deltas of vLLM's prefix-cache / external-prefix-cache / offload byte counters.

Example (port-forward the GPU Service first):
  kubectl -n llm-d-gpu port-forward svc/llmd-gpu-service 18003:8000 &
  python3 scripts/88_kv_tiering_bench.py --url http://127.0.0.1:18003 \
      --model RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic --sessions 128 --rounds 5 --concurrency 32 \
      --paras 140 --legs notier,tier,notier --out results/kv-tiering-$(date +%Y%m%d-%H%M%S).json
"""
import argparse, json, random, statistics, threading, time, urllib.request

def prefix(seed, paras=30):
    rnd = random.Random(seed)
    topics = ["tensor parallelism", "KV cache paging", "speculative decoding", "FP8 quantization",
              "expert parallelism", "prefix caching", "continuous batching", "chunked prefill"]
    out = [f"Agent session {seed:08x}: you are a Hermes coding agent. Answer in one short sentence.\n"]
    for k in range(paras):
        t = topics[(seed * 7 + k) % len(topics)]
        out.append(f"Workspace note {k+1}: the team documented {t} behaviour under agentic load; observed that {t} "
                   f"interacts with request fan-out, retry budgets and session affinity; record findings as short "
                   f"bullets, prefer measured numbers, cite run id W{(seed+k)%97:02d}-{rnd.randint(0,999):03d}.")
    return "\n".join(out)

Q = "Which note mentions the highest run id? One sentence."

def metrics(url):
    try:
        txt = urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10).read().decode()
    except Exception as e:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        k, v = line.rsplit(" ", 1)
        try:
            out[k] = out.get(k, 0.0) + float(v)
        except ValueError:
            pass
    return out

def pick(m, *subs):
    """Sum of all series whose name contains every substring."""
    return sum(v for k, v in m.items() if all(s in k for s in subs))


def pct(v, q):
    if not v: return None
    s = sorted(v); return round(s[min(len(s) - 1, int(q * len(s)))], 1)

def run_leg(a, name, extra):
    sessions = [{"sid": i, "prefix": prefix(a.seed * 100 + i, a.paras), "msgs": []} for i in range(a.sessions)]
    results = []  # (round, ttft_ms, out_tokens, prompt_tokens)
    lock = threading.Lock()
    m0 = metrics(a.url); t_start = time.perf_counter()
    for r in range(a.rounds):
        q = [s for s in sessions]
        def worker():
            while True:
                with lock:
                    if not q: return
                    s = q.pop(0)
                body = {"model": a.model,
                        "messages": [{"role": "system", "content": s["prefix"]}] + s["msgs"] +
                                    [{"role": "user", "content": f"Turn {r+1}: which note mentions the highest run id? One sentence."}],
                        "max_tokens": a.max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
                if extra: body.update(extra)
                req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                t0 = time.perf_counter(); ttft = None; usage = None; text = []
                try:
                    with urllib.request.urlopen(req, timeout=600) as resp:
                        for raw in resp:
                            line = raw.decode().strip()
                            if not line.startswith("data:") or line == "data: [DONE]": continue
                            ev = json.loads(line[5:])
                            if ev.get("usage"): usage = ev["usage"]
                            for ch in ev.get("choices") or []:
                                c = (ch.get("delta") or {}).get("content")
                                if c:
                                    if ttft is None: ttft = time.perf_counter() - t0
                                    text.append(c)
                except Exception as e:
                    with lock: results.append((r, None, 0, 0)); 
                    continue
                ttft = (ttft or (time.perf_counter() - t0)) * 1000
                out = int((usage or {}).get("completion_tokens") or 0); ptok = int((usage or {}).get("prompt_tokens") or 0)
                with lock:
                    results.append((r, ttft, out, ptok))
                    s["msgs"] += [{"role": "user", "content": f"Turn {r+1}: which note mentions the highest run id? One sentence."},
                                  {"role": "assistant", "content": "".join(text).strip() or "ok"}]
        ths = [threading.Thread(target=worker) for _ in range(a.concurrency)]
        [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.perf_counter() - t_start; m1 = metrics(a.url)
    d = lambda *s: pick(m1, *s) - pick(m0, *s)
    cold = [t for (r, t, o, p) in results if r == 0 and t is not None]
    warm = [t for (r, t, o, p) in results if r >= 1 and t is not None]
    out_tok = sum(o for (_, t, o, _) in results if t is not None)
    errs = sum(1 for (_, t, _, _) in results if t is None)
    summary = {
        "leg": name, "sessions": a.sessions, "rounds": a.rounds, "concurrency": a.concurrency, "paras": a.paras,
        "wall_s": round(wall, 1), "requests": len(results), "errors": errs,
        "out_tok_s": round(out_tok / wall, 1), "req_s": round(len(results) / wall, 2),
        "ttft_cold_p50_ms": pct(cold, 0.5), "ttft_cold_p95_ms": pct(cold, 0.95),
        "ttft_return_p50_ms": pct(warm, 0.5), "ttft_return_p95_ms": pct(warm, 0.95),
        "hbm_hit_tokens": d("vllm:prefix_cache_hits_total"), "hbm_query_tokens": d("vllm:prefix_cache_queries_total"),
        "ext_hit_tokens": d("vllm:external_prefix_cache_hits_total"), "ext_query_tokens": d("vllm:external_prefix_cache_queries_total"),
        "cpu_to_gpu_GB": round(d("kv_offload_total_bytes_total", "CPU_to_GPU") / 1e9, 2),
        "gpu_to_cpu_GB": round(d("kv_offload_total_bytes_total", "GPU_to_CPU") / 1e9, 2),
    }
    print(json.dumps(summary), flush=True)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", type=int, default=64); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=16); ap.add_argument("--paras", type=int, default=140)
    ap.add_argument("--max-tokens", type=int, default=48); ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--legs", default="notier,tier"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    notier = {"kv_transfer_params": {"max_load_tokens": 0, "max_offload_tokens": 0}}
    res = []
    for leg in a.legs.split(","):
        # new seed per leg so no leg benefits from the previous leg's cache
        a.seed += 1
        res.append(run_leg(a, leg, notier if leg == "notier" else None))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
