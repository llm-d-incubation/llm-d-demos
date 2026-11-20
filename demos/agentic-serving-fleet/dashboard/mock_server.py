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

"""Standalone simulator backend for the "Agent Substrate x llm-d" observability dashboard.

Python 3 stdlib only. Serves ./index.html at "/" and simulates the JSON API:

  GET  api/state                                  full dashboard state (page polls at 4 Hz)
  POST api/burst     {"hold": true}               wake all agents at once
                     (+ optional "agents": N, "concurrency": C, like the real driver)
  POST api/traffic   {"rate": 100}                steady agent traffic in req/s (0 = off)
  POST api/strategy  {"mode": "balanced" | "steer8020" | "priority"}
  POST api/suspend   {}                           suspend all running agents (scale to zero)
  POST api/reconcile {}                           heal every agent back to suspended (phase "reconciling")
  POST api/kv_tier   {"mode": "without" | "with"}  KV cache tiering leg (screen 4); state.llmd.compare.kv_tier

Behaviour mirrors fleet_driver/main.go where it matters to the page:
  * a burst first runs a short preflight while the phase is still "idle" (a second POST gets a 409 "busy"),
    then resets totals, unique replies and the ticker (seq keeps counting) and starts the clock
  * the driver stays busy until every woken agent got its first reply: suspend is refused (409) until then
  * suspend first lets in-flight traffic drain (suspend clock not started, suspend_elapsed_ms null), then
    suspends with at most 200 calls in flight, so 1,000 agents take a few seconds; burst.running counts
    every sandbox that is still up (agent states 2, 3 and 4)
  * suspend and reconcile reset the traffic rate to 0
  * ramp entries are stamped with the END of each 1,000 ms step, include the in-progress step, and freeze
    once every woken agent got its first reply
  * per-pod e2e/ttft/queue/cache fields and band wait_ms are null while there is no data
  * "note" carries the driver's last status line (omitted when empty)
  * errors are HTTP 409 {"ok": false, "error": "..."}; success is {"ok": true, "result": ...}
Not mocked: burst.ramp_fine, api/summary, api/events.

Hermes Agent mode (--harness hermes; default --harness sandbox keeps the old payload, plus config.harness):
  * config.harness = "hermes" and a state.memory object: enabled, gen, taught, recall_ok, recall_fail,
    agents_recalled, max_survived, mean_survived, awake_pct, proxy_calls, proxy_failures
  * every agent has a codename (adjective-noun). Its first reply is tagged memory "taught"; later replies
    start with the codename and are tagged "recalled" (~1 % "forgot"), with survived = suspends since taught.
    Memory lives in the (mock) Hermes session, so it survives duty-cycle suspends, Suspend all and new bursts.
  GET  api/agent?i=N                              one agent (N = 1..1000): codename, session, recalls, wakes,
                                                  last replies, awake_pct, last prompt / cached tokens
  POST api/memory/reset {}                        new codenames + fresh sessions (idle only) -> result: gen
  awake_pct: per agent, % of wall time awake since its codename was given (agents taught >= 5 s ago);
  the fleet value is the mean over those agents (mock approximation of the driver's number).

Mock-only extra (NOT part of the real contract, handy for testing edge states):
  POST api/mock      {"pod_up": [true, false]} | {"fail_rate": 0.01} | {"suspend_fail_rate": 0.02}
                     | {"reset": true}

Routes match by suffix, so the page also works behind a path prefix,
e.g. http://127.0.0.1:8765/some/prefix/ .

Usage:  python3 mock_server.py [port] [--harness sandbox|hermes] [--no-cluster]     (default 8765, sandbox)
        config.cluster: sandbox 25 x c4-standard-4 (4 vCPU), hermes 18 x c4d-standard-16 (16 vCPU);
        --no-cluster omits it, like an older backend.
"""

import bisect
import heapq
import json
import math
import os
import random
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "index.html")

TOTAL = 1000
PODS = ("Pod 1", "Pod 2")
MODEL = "google/gemma-4-26B-A4B-it"
ACCEL = "Pod 1 + Pod 2"
MAX_TOKENS = 50
WINDOW_S = 3.0
TICK_S = 0.01
METRICS_EVERY_S = 0.1
HIST_EVERY_S = 0.5
HIST_KEEP = 120
TICKER_KEEP = 60
RAMP_MAX_PTS = 120          # the driver sends at most 120 ramp entries (1,000 ms steps)
MAX_SEQS = 128              # vLLM max_num_seqs per pod (mock): beyond this requests wait inside vLLM
SAT_RPS = 400.0             # request rate that would fully saturate the pool (mock)
SATURATE_AT_RPS = 300       # priority mode forms flow-control queues at/above this steady rate
BANDS = ((100, "premium"), (0, "standard"), (-10, "best-effort"))

# ---- Oct 10: CALIBRATION of the simulation to the live system (docs/RESUME_STATE.md). EVERY simulated throughput,
# latency and KV figure in this file derives from this table; no other literal tok/s, TTFT or KV-hit value exists.
# Provenance:
#   pool.gpu  GPU pool (Node pool B, 4 pods x 1 GB200, TP=1), closed loop, 24 agents per pool, 4 turns per session,
#             ~9.5k-token per-session workspace prefix, 32 output tokens. Measured 2026-10-09 19:30Z:
#             round-robin ("without", llmd-gpu-service) and llm-d router ("with", gateway/EPP). e2e_p95_ms was NOT
#             measured: derived as ttft_p95 + 1.5 x the median decode time (= 1.9 x e2e_p50 for round-robin).
#   pool.tpu  TPU pool (Node pool A, 1 pod on a 2x2x1 tpu7x slice), same workload through the gateway. NOT part of the
#             A/B: identical numbers in both routing modes, no baseline. OLDER MEASUREMENT, TO BE RECALIBRATED:
#             2026-10-09 01:50Z/03:29Z with 8 agents gave TTFT p50 190-215 ms, KV hit ~50 %, 267 tok/s. Used: TTFT p50
#             200, hit 50, tok_s scaled to the 24-agent closed loop but clearly below the GPU pool (560). ttft_p95 and
#             e2e_p50 are not measured (p95 = 2.2 x p50, between the two GPU ratios; e2e = TTFT + 32 x ~8 ms/token).
#   flow      Flow control measured 2026-10-09 03:29Z: without = one shared FIFO ~470 ms for everyone (Pro 461, Free
#             475); with = lanes Pro/interactive 70 ms, Standard 400 ms, Free/background 1103 ms.
#   dyn/noise Time constants and jitter chosen to look like the live dashboard (KV hit climbs over ~30 s after the
#             toggle, 1 s samples wobble a few %, an occasional p95 spike); not measurements.
#   kv_usage_pct / chip_util_steady are cosmetic (pods[].kv_usage_pct is not displayed; chips tint by utilisation).
CALIB = {
    "measured": "2026-10-09 19:30Z",
    "agents_per_pool": 24,       # closed-loop agents on each pool (LIVE_GATEWAY_WORKERS=48, split 50/50)
    "turns": 4,                  # turns per session (AB_TURNS=4)
    "prefix_tok": 9500,          # per-session workspace prefix (AB_PREFIX_PARAS=160)
    "turn_tok": 70,              # tokens a turn adds to the session (user message + reply)
    "out_tok": 32,               # output tokens per turn -> req_s = tok_s / out_tok
    "pool": {
        "gpu": {
            "without": {"tok_s": 916.0, "ttft_p50_ms": 164.0, "ttft_p95_ms": 409.0, "e2e_p50_ms": 408.0, "cache_hit_pct": 29.0},
            "with":    {"tok_s": 1615.0, "ttft_p50_ms": 104.0, "ttft_p95_ms": 188.0, "e2e_p50_ms": 258.0, "cache_hit_pct": 75.0},
        },
        "tpu": {
            "without": {"tok_s": 560.0, "ttft_p50_ms": 200.0, "ttft_p95_ms": 440.0, "e2e_p50_ms": 456.0, "cache_hit_pct": 42.0},
            "with":    {"tok_s": 920.0, "ttft_p50_ms": 132.0, "ttft_p95_ms": 265.0, "e2e_p50_ms": 340.0, "cache_hit_pct": 78.0},
        },
    },
    "e2e_p95_decode_x": 1.5,     # e2e_p95 = ttft_p95 + this x median decode time (not measured)
    "flow": {
        "without": {"interactive_ms": 461.0, "standard_ms": 470.0, "background_ms": 475.0},
        "with":    {"interactive_ms": 70.0, "standard_ms": 400.0, "background_ms": 1103.0},
    },
    "tier_mix": (0.2, 0.6, 0.2),  # premium / standard / best-effort share of requests (one agent in five per edge tier)
    "dyn": {
        "tau_load_s": 4.0,       # wake: closed-loop load ramps to steady state (95 % by ~12 s)
        "tau_warm_s": 5.0,       # wake: KV hit starts near 0 (every first turn misses) and warms up
        "tau_up_s": 12.0,        # round-robin -> router: KV hit closes ~90 % of the gap by ~30 s
        "tau_down_s": 5.0,       # router -> round-robin: decays within ~15 s
        "tau_off_s": 1.5,        # traffic stops: load drains
        "publish_warm_s": 6.0,   # a mode needs a full window before compare.routing shows it (live: ~6 s)
        "publish_every_s": 3.0,  # compare.routing re-published like a window
        "cold_ttft_x": 1.15,     # TTFT of a session's first (cold-prefix) turn vs the pool p50
        "warm_hit_ttft_x": 0.9,  # TTFT of a warm, cache-hit turn vs the pool p50
        "warm_miss_ttft_x": 1.1,
        "load_ttft_floor": 0.8,  # TTFT at zero load vs steady (less queueing)
        "extra_agents_tok_x": 0.6,   # marginal throughput per extra agent beyond the calibrated 24 (off-default split only)
        "extra_agents_ttft_x": 0.5,  # TTFT growth per 100 % extra agents (off-default split only)
    },
    "noise": {                   # AR(1) per second: x = 0.7 x + N(0, sigma); std = 1.4 sigma; relative unless noted
        "tok": 0.014, "ttft50": 0.018, "ttft95": 0.035, "e2e50": 0.015, "e2e95": 0.03,
        "hit_pts": 1.0,          # KV hit, percentage points
        "spike_p": 1.0 / 15.0, "spike": (1.3, 1.6),   # occasional p95 spike, one sample
        "pub": 0.005,            # compare.routing (published windows)
        "flow": 0.04,            # band wait times
        "chip_phi": 0.97, "chip_sigma": 0.012,       # slow per-chip drift (std ~0.05)
        "pod_phi": 0.95, "pod_sigma": 0.5, "pod_max": 4.0,   # per-pod KV hit offset (points)
    },
    "chip_util_steady": 0.65,    # per-chip utilisation at the calibrated steady state (0.55-0.75 with drift)
    "router_pod_taper": 0.8,     # router: prefix affinity concentrates sessions, pod j carries ~0.8^j of pod 0
    "router_pod_hit_x": 60.0,    # router: pods with more sessions hit more (points per unit share above 1/n)
    "kv_usage_pct": {"gpu": 34.0, "tpu": 41.0},   # cosmetic, not measured
}


def _calib_finish(c):
    """Fill the derived fields once: pools get {without, with} (TPU: identical), e2e_p95_ms, decode_ms, req_s."""
    for key, ent in list(c["pool"].items()):
        if "without" not in ent:
            ent = {"without": dict(ent), "with": dict(ent)}
            c["pool"][key] = ent
        for mode in ("without", "with"):
            m = ent[mode]
            m["decode_ms"] = m["e2e_p50_ms"] - m["ttft_p50_ms"]
            m["e2e_p95_ms"] = m["ttft_p95_ms"] + c["e2e_p95_decode_x"] * m["decode_ms"]
            m["req_s"] = m["tok_s"] / float(c["out_tok"])
    return c


_calib_finish(CALIB)
BAND_WAIT_MS = (CALIB["flow"]["with"]["interactive_ms"], CALIB["flow"]["with"]["standard_ms"], CALIB["flow"]["with"]["background_ms"])
SHARED_WAIT_MS = CALIB["flow"]["without"]["standard_ms"]   # one shared FIFO without flow control
STANDARD = 1
MILESTONE_FRACS = (0.10, 0.25, 0.50, 0.75, 1.0)
SUSPEND_CAP = 200           # the driver caps suspend calls in flight at 200 ...
SUSPEND_S = (0.35, 0.95)    # ... and one suspend call takes this long (mock), so 1,000 agents take ~3 s
DRAIN_MAX_S = 15.0          # suspend waits (at most this long) for in-flight traffic before its clock starts
PREFLIGHT_S = (0.2, 0.4)    # one ListActors round trip (mock); every burst and reconcile starts with one
GATE_S = 0.3                # the driver parks all agent goroutines for 300 ms before it starts the clock
RECREATE_S = 1.2            # delete + create of one crashed agent (mock)
RECONCILE_MIN_S = 1.5       # a reconcile always shows up for at least this long (mock)

JOKES = (
    "x = x.to('cuda', dtype=torch.bfloat16)",
    "probs = torch.softmax(logits, dim=-1)",
    "loader = DataLoader(ds, batch_size=32, shuffle=True)",
    "torch.save(model.state_dict(), 'ckpt.pt')",
    "model = torch.compile(model)",
    "n = sum(p.numel() for p in model.parameters() if p.requires_grad)",
    "torch.manual_seed(0); torch.cuda.manual_seed_all(0)",
    "x = x.flatten(1)",
    "with torch.no_grad(): out = model(x)",
    "opt = torch.optim.AdamW(model.parameters(), lr=3e-4)",
    "loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)",
    "y = torch.einsum('bij,bjk->bik', a, b)",
    "mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)",
    "x = F.scaled_dot_product_attention(q, k, v, is_causal=True)",
    "model.load_state_dict(torch.load('ckpt.pt', map_location='cpu'))",
    "scaler = torch.amp.GradScaler('cuda')",
    "with torch.autocast('cuda', dtype=torch.bfloat16): out = model(x)",
    "x = torch.cat([a, b], dim=1)",
    "idx = torch.argmax(logits, dim=-1)",
    "emb = nn.Embedding(vocab_size, d_model)",
    "x = x.permute(0, 2, 3, 1).contiguous()",
    "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)",
    "sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)",
    "x = F.layer_norm(x, x.shape[-1:])",
    "acc = (pred == target).float().mean().item()",
    "k_cache = torch.zeros(B, H, T, D, dtype=torch.float8_e4m3fn)",
    "x = x.unsqueeze(0).expand(B, -1, -1)",
    "model.eval(); model.requires_grad_(False)",
    "params = {n: p for n, p in model.named_parameters() if 'lora' in n}",
    "torch.cuda.synchronize(); t = time.perf_counter() - t0",
    "x = torch.nn.functional.gelu(x, approximate='tanh')",
    "ds = ds.map(tokenize, batched=True, remove_columns=['text'])",
    "logits = logits[:, -1, :] / temperature",
    "out = model.generate(ids, max_new_tokens=48, do_sample=False)",
    "x = torch.where(mask, x, torch.finfo(x.dtype).min)",
    "dist.all_reduce(t, op=dist.ReduceOp.SUM)",
    "w = w.view(out_f, in_f).t().contiguous()",
    "torch.cuda.empty_cache()",
    "x = F.dropout(x, p=0.1, training=self.training)",
    "pe = torch.arange(T)[:, None] / 10000 ** (torch.arange(0, D, 2) / D)",
)
OPENERS = ("",)              # code one-liners: no chatty openers/closers
CLOSERS = ("",)

# ---- Hermes Agent mode (--harness hermes)
HARNESS = "hermes"           # set from argv in main()
# config.cluster per harness; --no-cluster omits it (emulates an older backend without the field)
CLUSTERS = {"sandbox": {"nodes": 25, "node_type": "c4-standard-4", "vcpus_per_node": 4},
            "hermes": {"nodes": 25, "node_type": "e2-standard-4", "vcpus_per_node": 4}}
SEND_CLUSTER = True
CODE_ADJ = ("tensor", "gradient", "sparse", "fused", "quantized", "latent", "eager", "sharded", "cached", "vector",
            "stochastic", "compiled", "async", "scalar", "ternary", "dropout")
CODE_NOUN = ("otter", "falcon", "panda", "gecko", "badger", "lynx", "heron", "koala", "marmot", "orca", "puffin",
             "wombat", "tapir", "ibis", "newt", "yak")
FORGET_RATE = 0.01
TAUGHT_FMT = ("{cn} › {joke}", "{cn} (noted) › {joke}")
RECALL_FMT = ("{cn} › {joke}",)
FORGOT_FMT = ("(no codename) › {joke}",)
HERMES_SYS_TOKENS = 1150     # Hermes system prompt + tool schemas (mock), cached after the first turn
HERMES_TURN_TOKENS = 64      # each turn adds to the session history (mock)
UPSET = (ord("1"), ord("2"), ord("3"), ord("4"))

C0, C1, C2, C3, C4, C9 = (ord(c) for c in "012349")


def mono():
    return time.monotonic()


def unix_ms():
    return int(time.time() * 1000)


def base36(n):
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    s = ""
    while True:
        n, r = divmod(n, 36)
        s = digits[r] + s
        if not n:
            return s


RUN_TAG = base36(int(time.time()))


def agent_name(i):
    return "agent-%04d" % (i + 1)


def logistic(rnd, mu, s):
    u = min(max(rnd.random(), 1e-9), 1.0 - 1e-9)
    return mu + s * math.log(u / (1.0 - u))


# ---- Oct 8: simulation hardware (what the stack view's Accelerators row shows; editable via POST api/sim).
# The live server reads the same fields from node labels; nothing here is ever sent by the real driver.
def topo_chips(topo):
    """'2x2x2' -> 8; '' or garbage -> 0."""
    n = 1
    dims = [d for d in str(topo or "").lower().replace(" ", "").split("x") if d]
    if not dims:
        return 0
    try:
        for d in dims:
            n *= int(d)
    except ValueError:
        return 0
    return n


# Oct 10: defaults mirror the deployed benchmark cluster (TPU: 1 pod on a 2x2x1 tpu7x slice = 4 chips; GPU: 4 x TP=1
# GB200 = 4 chips), so Simulation and Live show the same shape unless the drawer overrides it. No detail strings.
SIM_HARDWARE = {
    "tpu": "TPU", "gpu": "GPU", "tpu_pods": 2, "gpu_pods": 4,
    "engine": "vLLM", "tpu_backend": "TPU", "gpu_backend": "CUDA", "model_label": "Gemma 4 26B-A4B · FP8", "ab_pool": "gpu",
    "tpu_chip": "TorchTPU", "tpu_detail": "vllm-torchtpu", "tpu_topology": "", "tpu_chips": 8,
    "gpu_chip": "Cloud GPU", "gpu_detail": "vLLM / SGLang", "gpu_topology": "", "gpu_chips": 4,
}
POOL_KEYS = ("tpu", "gpu")   # pod 0 = TPU pool, pod 1 = GPU pool
# Oct 9: simulated pod IPs behind each pool's headless Service (per-pod KV cards); pod j of pool p = base.(j+1)
KV_HOSTS = {"tpu": "llmd-tpu-service.llm-d-serving.svc.cluster.local:8000",
            "gpu": "llmd-gpu-service.llm-d-gpu.svc.cluster.local:8000"}

# ---- Oct 9: agent trajectory view (api/state.trajectories), same shape as the live server
TRAJ_KEEP = 12               # sessions kept, newest first
TRAJ_LIVE = 8                # sessions in flight at any time (half TPU, half GPU)
TRAJ_TURNS = CALIB["turns"]  # turns per session (AB_TURNS on the live server)
TRAJ_TIERS = ("premium", "standard", "best-effort")
TRAJ_TASKS = (               # (turn-1 user message, one-line reply) coding tasks like server.CODE_TASKS
    ("Move tensor x to the accelerator in bfloat16.", "x = x.to(device, dtype=torch.bfloat16)"),
    ("Compute softmax probabilities of logits along the last dim.", "probs = torch.softmax(logits, dim=-1)"),
    ("Build a DataLoader over ds with batch size 32 and shuffling.", "loader = DataLoader(ds, batch_size=32, shuffle=True)"),
    ("Save model's state_dict to ckpt.pt.", "torch.save(model.state_dict(), 'ckpt.pt')"),
    ("Wrap model with torch.compile.", "model = torch.compile(model)"),
    ("Count trainable parameters of model.", "n = sum(p.numel() for p in model.parameters() if p.requires_grad)"),
    ("Seed torch and CUDA with 0.", "torch.manual_seed(0); torch.cuda.manual_seed_all(0)"),
    ("Flatten a 4D tensor x to 2D keeping the batch dim.", "x = x.flatten(1)"),
    ("Run model(x) with gradients disabled.", "with torch.no_grad(): logits = model(x)"),
    ("Create an AdamW optimizer for model with lr 3e-4.", "opt = torch.optim.AdamW(model.parameters(), lr=3e-4)"),
    ("One training step: backward, optimizer step, zero grads.", "loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)"),
    ("Causal scaled dot-product attention over q, k, v.", "out = F.scaled_dot_product_attention(q, k, v, is_causal=True)"),
    ("Clip model's gradient norm to 1.0.", "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)"),
    ("Greedy next-token ids from logits.", "ids = logits[:, -1].argmax(dim=-1)"),
    ("Allocate an FP8 KV cache of shape B,H,T,D.", "kv = torch.empty(B, H, T, D, dtype=torch.float8_e4m3fn, device=device)"),
    ("All-reduce sum tensor t across ranks.", "torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)"),
)
TRAJ_FOLLOWUPS = (           # (follow-up user message, reply) for turns 2..N (abbench._Session.next_user)
    ("Summarize the most important note so far in one sentence.", "Prefix caching keeps the shared workspace context warm across turns, so only the new tokens are prefilled."),
    ("Which note mentions the highest run id? One sentence.", "Note 11 cites run W07-91, the highest id in this workspace."),
    ("Give one concrete next step for this workspace.", "Pin sessions to the pod holding their prefix and measure TTFT before and after."),
    ("What is the main risk you see? One sentence.", "Round-robin spreads a session across pods, so every pod re-prefills the same prefix."),
    ("Name one metric we should track. One sentence.", "Track vllm:prefix_cache_hits_total over queries per pod, per minute."),
)


def pool_calib(key, mode):
    """Calibrated steady state of one pool in one routing mode (the TPU pool is identical in both)."""
    ent = CALIB["pool"].get(key) or CALIB["pool"]["gpu"]
    return ent["with" if mode == "with" else "without"]


def agents_scale(n):
    """(tok_s factor, latency factor) for n closed-loop agents on a pool calibrated at CALIB agents_per_pool.
    Below the calibrated count throughput is proportional and latency a little lower (less queueing); above it
    every extra agent adds extra_agents_tok_x of its share and latency grows (off-default traffic split only)."""
    d = CALIB["dyn"]
    x = max(0.0, n) / float(CALIB["agents_per_pool"])
    if x <= 1.0:
        return x, d["load_ttft_floor"] + (1.0 - d["load_ttft_floor"]) * x
    return 1.0 + d["extra_agents_tok_x"] * (x - 1.0), 1.0 + d["extra_agents_ttft_x"] * (x - 1.0)


def settled(key, mode, n_agents, jit=None):
    """Calibrated steady-state numbers of a pool with n_agents closed-loop agents, in the API shape (what the
    'vs round-robin' comparison shows). jit: optional dict of small relative jitters per key (published windows)."""
    c = pool_calib(key, mode)
    ft, fl = agents_scale(n_agents)
    j = lambda k: 1.0 + (jit or {}).get(k, 0.0)
    return {"tok_s": int(round(c["tok_s"] * ft * j("tok_s"))),
            "cache_hit_pct": round(c["cache_hit_pct"] * j("cache_hit_pct"), 1),
            "ttft_p50_ms": int(round(c["ttft_p50_ms"] * fl * j("ttft_p50_ms"))),
            "ttft_p95_ms": int(round(c["ttft_p95_ms"] * fl * j("ttft_p95_ms"))),
            "e2e_p50_ms": int(round(c["e2e_p50_ms"] * fl * j("e2e_p50_ms"))),
            "e2e_p95_ms": int(round(c["e2e_p95_ms"] * fl * j("e2e_p95_ms")))}


# ---- Oct 10: KV cache tiering (screen 4, api/state compare.kv_tier, POST api/kv_tier). Self-contained block:
# one calibration dict + one helper class, hooked into Sim.snapshot()/_metrics() and the POST table only.
# Calibrated to llmd-harness/results/kv-tiering-gemma4-b200-s128-c32-20261009.json (PR #3): B200 pair TP=2,
# HBM KV capped at 8 GiB/GPU, 64 GiB host pool, 128 sessions x ~8.1k-token prefix, 5 turns, 32 concurrent.
#   notier: returning-turn TTFT 1059/1804 ms, 345 tok/s, 16.5 req/s, host hit 0 %, restore 0
#   tier:   returning-turn TTFT  303/540  ms, 931 tok/s, 44.6 req/s, host hit 89 % (4.67M of 5.25M prefix tokens),
#           restore 216.5 GB / 14.3 s = 15.1 GB/s, offload 27.6 GB / 14.3 s = 1.9 GB/s, ~46 KB per token at 8k.
# Rates are the measured replica's own numbers (not multiplied by the sim's pod count); capacity is derived from the
# sim's pool shape: every accelerator is capped at hbm_cap_bytes, every replica (pod) has one host_bytes pool, so
#   HBM tokens  = hbm_cap_bytes x accelerators / bytes_per_token   (8 GiB x 4 / 46 KiB = ~729k tokens = ~90 sessions)
#   host tokens = host_bytes x replicas / bytes_per_token          (64 GiB x 4 / 46 KiB = ~5.8M tokens = ~720 sessions)
# and 128+ live sessions x 8.1k tokens cannot all stay in HBM, which is exactly why the host tier gets exercised.
CALIB_KV = {
    "pool": "gpu",                      # the sim shows the tier on the GPU pool (Node pool B in neutral mode)
    "hbm_cap_bytes": 8 * 2 ** 30,       # demo-only cap per accelerator (KV_TIER_HBM_CAP_BYTES), never a production setting
    "host_bytes": 64 * 2 ** 30,         # host-DRAM pool per replica (KV_TIER_CPU_BYTES default)
    "storage_bytes": 1024 * 2 ** 30,    # shared storage pool across replicas (Managed Lustre RWX)
    "bytes_per_token": 46 * 1024,       # Gemma 4 host copy at ~8k prompts (README: ~46 KB/token)
    "prefix_tokens": 8100,              # one agent session's workspace prefix in the bench
    "without": {"ttft_return_p50_ms": 1059.0, "ttft_return_p95_ms": 1804.0, "tok_s": 345.0, "req_s": 16.5,
                "hbm_hit_pct": 5.0, "host_hit_pct": 0.0, "storage_hit_pct": 0.0,
                "restore_gbps": 0.0, "offload_gbps": 0.0, "storage_read_gbps": 0.0, "storage_write_gbps": 0.0,
                "hbm_used_pct": 96.0, "host_used_pct": 0.0, "storage_used_pct": 0.0, "prefix_tok_s": 5245262.0 / 38.9},
    "with": {"ttft_return_p50_ms": 303.0, "ttft_return_p95_ms": 540.0, "tok_s": 931.0, "req_s": 44.6,
             "hbm_hit_pct": 5.0, "host_hit_pct": 74.0, "storage_hit_pct": 15.0,
             "restore_gbps": 15.1, "offload_gbps": 1.9, "storage_read_gbps": 4.8, "storage_write_gbps": 1.6,
             "hbm_used_pct": 97.0, "host_used_pct": 70.0, "storage_used_pct": 18.0, "prefix_tok_s": 5245184.0 / 14.3},
    "tau_up_s": 8.0,                    # host hit / restore / TTFT settle in with τ ≈ 8 s after the toggle ...
    "tau_down_s": 4.0,                  # ... and fall back with τ ≈ 4 s
    "tau_offload_s": 2.5,               # eviction traffic to the host starts almost at once
    "tau_host_fill_s": 10.0,            # host pool fills as prefixes get evicted; keeps them when the tier is switched off
    "tau_host_drain_s": 40.0,
    "settled_frac": 0.9,                # a leg counts as measured once it is this close to its calibrated value
    "noise_tau_s": 1.6,                 # AR(1) jitter correlation time (no sin() periodicity anywhere)
    "noise_pct": 0.035,
    # Oct 13: what the tier adds to the ROUTER's own headline (compare.routing.with) at full activation. The bench
    # above is one pool under a forced-eviction workload; across the cluster (tier on one pool, mixed turns) the
    # lift is smaller: +45% tok/s, -28% latency, +12 pts KV hit. Scaled by the tier activation x (0..1).
    "cluster_gain": {"tok_s": 0.45, "lat": 0.28, "hit_pts": 12.0},
    # Oct 13: KV cache capacity in HBM per pool (what vLLM's cache_config reports as num_gpu_blocks x block bytes).
    # The tiered pool reports its demo cap (hbm_cap_bytes x accelerators). No estimate for the other pool: the UI
    # shows '—' until it is measured or set (settings, planned). Live: server.py should fill by_pool.<pool>.kv_cap_bytes.
    "kv_cap_bytes": {"tpu": None, "gpu": None},
    # Oct 14: Tier 3 lives on two scale-out replicas of the tiered pool (4 GiB HBM cap and 4 GiB host pool on purpose,
    # one shared Lustre). Calibrated to the dashboard's storage ring (6 x 5 sessions, ~10.8k-token prefixes) on GB200
    # TP=1 with same-region Lustre: a returning turn lands on the other replica and restores from storage; one turn in
    # four opts out and recomputes as the control. Writes stop once the fixed session set is on storage.
    "storage": {
        "replicas": 2, "hbm_cap_bytes": 4 * 2 ** 30, "host_bytes": 4 * 2 ** 30, "storage_bytes": 1024 * 2 ** 30,
        "hbm_hit_pct": 0.0, "host_hit_pct": 0.0, "storage_hit_pct": 96.0,
        "read_gbps": 1.95, "write_gbps": 0.35, "restore_gbps": 1.95, "req_s": 7.0, "tokens_per_turn": 10800,
        "hbm_used_pct": 18.0,
        "restore": {"ttft_return_p50_ms": 415, "ttft_return_p95_ms": 520},
        "recompute": {"ttft_return_p50_ms": 250, "ttft_return_p95_ms": 310},
        "tau_up_s": 10.0, "tau_down_s": 3.0,
    },
}


class KVTier:
    """Simulated HBM -> host-DRAM -> shared-storage KV tier for one pool. Call step(now, active) from the metrics tick,
    mode(...) from POST api/kv_tier, state(hardware) from snapshot()."""

    def __init__(self, rnd):
        self.rnd = rnd
        self.mode = "without"
        self.enabled = os.environ.get("KV_TIER_ENABLED", os.environ.get("ENABLE_KV_TIERING", "0")) in ("1", "true", "yes")
        self.storage_enabled = os.environ.get("ENABLE_KV_STORAGE_TIER", "0") in ("1", "true", "yes")
        self.x = 0.0            # tier activation 0..1 (host hit, restore, TTFT follow it)
        self.xo = 0.0           # offload activation (faster)
        self.xh = 0.0           # host pool fill level 0..1
        self.n1 = 0.0           # AR(1) noise states
        self.n2 = 0.0
        self.n3 = 0.0
        self.leg_t0 = None      # when the current leg started running
        self.legs = {"without": None, "with": None}
        self.tok_restored = 0.0
        self.gb_restored = 0.0
        self.gb_offloaded = 0.0
        self.gb_storage_read = 0.0
        self.gb_storage_written = 0.0
        self.hist = deque([0.0] * 60, maxlen=60)
        self.next_hist = None
        self.last_t = None
        self.xs = 0.0           # Oct 14: Tier 3 activation (storage ring on the scale-out replicas)
        self.st_gb_read = 0.0
        self.st_gb_written = 0.0
        self.st_tok = 0.0
        self.st_n = 0
        self.live = self._live(0.0, False)

    def reset(self):
        self.__init__(self.rnd)

    def set_mode(self, mode):
        if mode not in ("without", "with", "storage"):
            return False, "mode must be without, with or storage"
        if mode == "storage" and not self.storage_enabled:
            return False, "no storage tier configured"
        if mode != self.mode:
            self._freeze_leg()
            self.mode = mode
            self.leg_t0 = None
        return True, self.mode

    def pool_mode(self):
        """Tier 3 keeps the pool itself on its host-DRAM leg; the storage tier is on the scale-out replicas."""
        return "with" if self.mode == "storage" else self.mode

    # ---- dynamics
    def _ar1(self, cur, dt):
        rho = math.exp(-dt / CALIB_KV["noise_tau_s"]) if dt > 0 else 1.0
        return rho * cur + math.sqrt(max(0.0, 1.0 - rho * rho)) * self.rnd.gauss(0.0, 1.0)

    @staticmethod
    def _relax(cur, tgt, dt, tau):
        return cur + (tgt - cur) * (1.0 - math.exp(-dt / tau)) if dt > 0 else cur

    def step(self, now, active):
        dt = 0.0 if self.last_t is None else max(0.0, min(0.5, now - self.last_t))
        self.last_t = now
        c = CALIB_KV
        on = bool(active and self.enabled and self.mode in ("with", "storage"))
        son = bool(active and self.enabled and self.storage_enabled and self.mode == "storage")
        running = bool(active and self.enabled)
        cs = c.get("storage") or {}
        self.xs = self._relax(self.xs, 1.0 if son else 0.0, dt, cs.get("tau_up_s", 10.0) if son else cs.get("tau_down_s", 3.0))
        if son and dt > 0:
            self.st_gb_read += cs.get("read_gbps", 0.0) * self.xs * dt
            self.st_gb_written += cs.get("write_gbps", 0.0) * 4.0 * self.xs * (1.0 - self.xs) * dt
            self.st_tok += cs.get("read_gbps", 0.0) * self.xs * 1e9 / c["bytes_per_token"] * dt
            self.st_n += cs.get("req_s", 0.0) * self.xs * dt
        if running and self.leg_t0 is None:
            self.leg_t0 = now
        if not running:
            self.leg_t0 = None
        self.x = self._relax(self.x, 1.0 if on else 0.0, dt, c["tau_up_s"] if on else c["tau_down_s"])
        self.xo = self._relax(self.xo, 1.0 if on else 0.0, dt, c["tau_offload_s"])
        self.xh = self._relax(self.xh, 1.0 if on else 0.0, dt, c["tau_host_fill_s"] if on else c["tau_host_drain_s"])   # Oct 13: HBM only = host pool drains
        self.n1 = self._ar1(self.n1, dt)
        self.n2 = self._ar1(self.n2, dt)
        self.n3 = self._ar1(self.n3, dt)
        live = self._live(self.x if running else 0.0, running)
        if running:
            self.tok_restored += live["restore_gbps"] * 1e9 / c["bytes_per_token"] * dt
            self.gb_restored += live["restore_gbps"] * dt
            self.gb_offloaded += live["offload_gbps"] * dt
            if self.storage_enabled:
                self.gb_storage_read += live.get("storage_read_gbps", 0.0) * dt
                self.gb_storage_written += live.get("storage_write_gbps", 0.0) * dt
        live["tokens_restored_total"] = int(self.tok_restored)
        live["gb_restored_total"] = round(self.gb_restored, 1)
        live["gb_offloaded_total"] = round(self.gb_offloaded, 1)
        live["gb_storage_read_total"] = round(self.gb_storage_read, 2)
        live["gb_storage_written_total"] = round(max(self.gb_storage_written, 54.0 * self.xh if self.storage_enabled else 0.0), 1)
        if self.next_hist is None or now >= self.next_hist:
            self.next_hist = (self.next_hist or now) + 1.0
            if self.next_hist < now:
                self.next_hist = now + 1.0
            self.hist.append(live["restore_gbps"])
        live["restore_gbps_hist"] = [round(v, 2) for v in self.hist]
        self.live = live
        # the running leg publishes the dynamic path; a leg that has run long enough freezes at its calibrated values
        if running:
            if on and self.legs["without"] is None:
                cal0 = CALIB_KV["without"]
                self.legs["without"] = {
                    "ttft_return_p50_ms": int(round(cal0["ttft_return_p50_ms"])), "ttft_return_p95_ms": int(round(cal0["ttft_return_p95_ms"])),
                    "hbm_hit_pct": cal0["hbm_hit_pct"], "host_hit_pct": cal0["host_hit_pct"],
                    "storage_hit_pct": 0.0 if self.storage_enabled else None,
                    "restore_gbps": cal0["restore_gbps"], "offload_gbps": cal0["offload_gbps"],
                    "tokens_restored": 0, "tok_s": int(round(cal0["tok_s"])), "req_s": cal0["req_s"],
                    "live": False, "settled": True,
                }
            settled = (self.x >= c["settled_frac"]) if on else (self.x <= 1.0 - c["settled_frac"])
            self.legs[self.pool_mode()] = dict(self._leg_from_live(live), live=True, settled=settled)
            other = "with" if self.pool_mode() == "without" else "without"
            if self.legs[other]:
                self.legs[other]["live"] = False
        else:
            for k in self.legs:
                if self.legs[k]:
                    self.legs[k]["live"] = False

    def _freeze_leg(self):
        """Switching away: hold the calibrated numbers for a leg that settled, else its last dynamic values."""
        pm = self.pool_mode()
        leg = self.legs.get(pm)
        if leg and leg.get("settled"):
            cal = CALIB_KV[pm]
            st_h = cal.get("storage_hit_pct", 0.0) if self.storage_enabled else 0.0
            ho_h = cal["host_hit_pct"] if self.storage_enabled else (cal["host_hit_pct"] + cal.get("storage_hit_pct", 0.0))
            self.legs[pm] = {
                "ttft_return_p50_ms": int(round(cal["ttft_return_p50_ms"])), "ttft_return_p95_ms": int(round(cal["ttft_return_p95_ms"])),
                "hbm_hit_pct": cal["hbm_hit_pct"], "host_hit_pct": ho_h,
                "storage_hit_pct": st_h if self.storage_enabled else None,
                "restore_gbps": cal["restore_gbps"], "offload_gbps": cal["offload_gbps"],
                "tokens_restored": leg.get("tokens_restored", 0), "tok_s": int(round(cal["tok_s"])), "req_s": cal["req_s"],
                "live": False, "settled": True,
            }
        elif leg:
            leg["live"] = False

    def _live(self, x, running=True):
        c, w0, w1 = CALIB_KV, CALIB_KV["without"], CALIB_KV["with"]
        mix = lambda k: w0[k] + (w1[k] - w0[k]) * x
        j = lambda v, n: max(0.0, v * (1.0 + c["noise_pct"] * n))
        _acc, rep = self.shape()
        if not running:
            host_idle = self.xh * w1["host_used_pct"]
            st_idle = self.xh * w1["storage_used_pct"] if self.storage_enabled else 0.0
            return {"hbm_hit_pct": None, "host_hit_pct": None, "storage_hit_pct": None,
                    "restore_gbps": 0.0, "offload_gbps": 0.0, "storage_read_gbps": 0.0, "storage_write_gbps": 0.0,
                    "hbm_used_pct": 0.0, "host_used_pct": round(host_idle, 1), "storage_used_pct": round(st_idle, 1),
                    "ttft_return_p50_ms": None, "ttft_return_p95_ms": None, "tok_s": None, "req_s": None, "prefix_tok_s": None,
                    "sessions_hbm": 0, "sessions_host": self._sessions(host_idle / 100.0, "host"),
                    "sessions_storage": self._sessions(st_idle / 100.0, "storage") if self.storage_enabled else 0,
                    "pods": [{"name": f"pod-{i + 1}", "cache_hit_pct": None, "storage_hit_pct": None,
                              "storage_read_gb": 0.0, "storage_write_gb": 0.0, "running": 0, "is_new": False}
                             for i in range(rep)]}
        host_used = self.xh * w1["host_used_pct"]
        st_used = self.xh * w1["storage_used_pct"] if self.storage_enabled else 0.0
        ho_pct = round(min(100.0 - 0.1, j(mix("host_hit_pct") + (0.0 if self.storage_enabled else mix("storage_hit_pct")), self.n1)), 1)
        st_pct = round(min(100.0 - ho_pct, j(mix("storage_hit_pct"), 0.8 * self.n3)), 1) if self.storage_enabled else None
        tot_hit = round(min(100.0, mix("hbm_hit_pct") + ho_pct + (st_pct or 0.0)), 1)
        pods = []
        for i in range(rep):
            is_new = bool(self.storage_enabled and rep >= 2 and i == rep - 1 and x > 0.15)
            p_hit = round(min(100.0, max(0.0, tot_hit + (i - (rep - 1) / 2.0) * 0.8)), 1) if x > 0.05 else round(mix("hbm_hit_pct"), 1)
            p_st = round(min(p_hit, 88.4 * x), 1) if is_new else (round((st_pct or 0.0) * 0.25, 1) if self.storage_enabled else None)
            pods.append({
                "name": f"pod-{i + 1}",
                "cache_hit_pct": p_hit,
                "storage_hit_pct": p_st,
                "storage_read_gb": round(9.06 * x, 2) if is_new else 0.0,
                "storage_write_gb": round(2.1 * self.xh, 1) if is_new else round(27.0 * self.xh, 1),
                "running": max(1, int(round(mix("req_s") / max(1, rep)))),
                "is_new": is_new,
            })
        return {
            "hbm_hit_pct": round(min(100.0, j(mix("hbm_hit_pct"), 0.6 * self.n3)), 1),
            "host_hit_pct": ho_pct,
            "storage_hit_pct": st_pct,
            "restore_gbps": round(j(w1["restore_gbps"] * x, self.n1 * 1.4), 2),
            "offload_gbps": round(j(w1["offload_gbps"] * self.xo, self.n2 * 2.0), 2),
            "storage_read_gbps": round(j(w1["storage_read_gbps"] * x, self.n3 * 1.2), 2) if self.storage_enabled else 0.0,
            "storage_write_gbps": round(j(w1["storage_write_gbps"] * self.xo, self.n2 * 1.5), 2) if self.storage_enabled else 0.0,
            "hbm_used_pct": round(min(100.0, j(mix("hbm_used_pct"), 0.2 * self.n2)), 1),
            "host_used_pct": round(min(100.0, host_used), 1),
            "storage_used_pct": round(min(100.0, st_used), 1),
            "ttft_return_p50_ms": int(round(j(mix("ttft_return_p50_ms"), -self.n1 + 0.5 * self.n2))),
            "ttft_return_p95_ms": int(round(j(mix("ttft_return_p95_ms"), -self.n1 + 0.8 * self.n3))),
            "tok_s": int(round(j(mix("tok_s"), self.n2))),
            "req_s": round(j(mix("req_s"), self.n2), 1),
            "prefix_tok_s": int(round(mix("prefix_tok_s"))),
            "sessions_hbm": self._sessions(mix("hbm_used_pct") / 100.0, "hbm"),
            "sessions_host": self._sessions(host_used / 100.0, "host"),
            "sessions_storage": self._sessions(st_used / 100.0, "storage") if self.storage_enabled else 0,
            "pods": pods,
        }

    def _leg_from_live(self, live):
        leg = {k: live.get(k) for k in ("ttft_return_p50_ms", "ttft_return_p95_ms", "hbm_hit_pct", "host_hit_pct",
                                         "storage_hit_pct", "restore_gbps", "offload_gbps", "tok_s", "req_s")}
        leg["tokens_restored"] = live.get("tokens_restored_total", 0)
        return leg

    # ---- capacity (pool shape comes from the sim hardware; see the derivation above CALIB_KV)
    def shape(self):
        return getattr(self, "_shape", (1, 1))

    def set_shape(self, accelerators, replicas):
        self._shape = (max(1, int(accelerators or 1)), max(1, int(replicas or 1)))

    def _sessions(self, frac, tier):
        acc, rep = self.shape()
        c = CALIB_KV
        if tier == "hbm":
            cap_b = c["hbm_cap_bytes"] * acc
        elif tier == "host":
            cap_b = c["host_bytes"] * rep
        else:
            cap_b = c["storage_bytes"]
        tokens = cap_b / float(c["bytes_per_token"])
        return int(round(max(0.0, frac) * tokens / c["prefix_tokens"]))

    def storage_block(self):
        """compare.kv_tier.storage for the simulation: same shape as abbench.storage_state()."""
        if not self.storage_enabled:
            return None
        cs = CALIB_KV["storage"]
        xs = self.xs
        j = lambda v, n: max(0.0, v * (1.0 + CALIB_KV["noise_pct"] * n))
        on = self.mode == "storage"
        rep = int(cs["replicas"])
        st_pct = round(cs["storage_hit_pct"] * xs, 1) if on else None
        ho_pct = round(cs["host_hit_pct"] * xs, 1) if on else None
        hb_pct = round(cs["hbm_hit_pct"] * xs + (0.0 if on else 0.0), 1) if on else None
        pods = []
        for i in range(rep):
            pods.append({
                "name": f"pod-{i + 1}",
                "cache_hit_pct": round(min(100.0, (hb_pct or 0.0) + (ho_pct or 0.0) + (st_pct or 0.0)), 1) if on else None,
                "ext_hit_pct": round((ho_pct or 0.0) + (st_pct or 0.0), 1) if on else None,
                "storage_read_gb": round(self.st_gb_read / rep, 2),
                "storage_write_gb": round(self.st_gb_written / rep, 1),
                "running": int(round(cs["req_s"] * xs / rep * 0.3)) if on else 0,
                "is_new": False,
            })
        n_rest = int(self.st_n * 0.75)
        n_rec = int(self.st_n * 0.25)
        leg = lambda k, n: ({"ttft_return_p50_ms": int(round(j(cs[k]["ttft_return_p50_ms"], 0.5 * self.n1))),
                             "ttft_return_p95_ms": int(round(j(cs[k]["ttft_return_p95_ms"], 0.5 * self.n2))),
                             "req_s": round(cs["req_s"] * xs * (0.75 if k == "restore" else 0.25), 1), "n": n, "live": on}
                            if n >= 3 else None)
        return {
            "enabled": True, "source": "simulated", "replicas": rep,
            "hbm_cap_bytes": cs["hbm_cap_bytes"], "host_bytes": cs["host_bytes"], "storage_bytes": cs["storage_bytes"],
            "active": bool(on and xs > 0.02), "errors": 0,
            "live": {
                "hbm_hit_pct": hb_pct, "host_hit_pct": ho_pct, "storage_hit_pct": st_pct,
                "read_gbps": round(j(cs["read_gbps"] * xs, self.n3), 2) if on else 0.0,
                # Writes and offloads only happen while the session set is still being written through (the ramp).
                "write_gbps": round(j(cs["write_gbps"] * 4.0 * xs * (1.0 - xs), self.n2), 2) if on else 0.0,
                "restore_gbps": round(j(cs["restore_gbps"] * xs, self.n1), 2) if on else 0.0,
                "offload_gbps": round(j(cs["restore_gbps"] * 0.8 * xs * (1.0 - xs), self.n2), 2) if on else 0.0,
                "hbm_used_pct": round(min(100.0, j(cs["hbm_used_pct"] * xs, 2.0 * self.n3)), 1) if on else 0.0,
                "gb_read_total": round(self.st_gb_read, 2), "gb_written_total": round(self.st_gb_written, 1),
                "tokens_restored_total": int(self.st_tok),
                "pods": pods,
            },
            "legs": {"restore": leg("restore", n_rest), "recompute": leg("recompute", n_rec)},
        }

    def state(self):
        acc, rep = self.shape()
        c = CALIB_KV
        return {
            "enabled": self.enabled, "storage_enabled": self.storage_enabled, "pool": c["pool"], "mode": self.mode,
            "storage": self.storage_block(),
            "hbm_cap_bytes": c["hbm_cap_bytes"], "host_bytes": c["host_bytes"],
            "storage_bytes": c["storage_bytes"] if self.storage_enabled else None,
            "bytes_per_token": c["bytes_per_token"],
            "prefix_tokens": c["prefix_tokens"], "accelerators": acc, "replicas": rep,
            "live": dict(self.live),
            "without": dict(self.legs["without"]) if self.legs["without"] else None,
            "with": dict(self.legs["with"]) if self.legs["with"] else None,
        }


def percentiles(v):
    """Same shape and rounding as the driver's pct struct."""
    if not v:
        return {"n": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "min": 0, "avg": 0}
    s = sorted(v)
    at = lambda q: round(s[int(min(len(s) - 1, q * len(s)))])
    return {"n": len(s), "p50": at(0.5), "p90": at(0.9), "p99": at(0.99), "max": round(s[-1]),
            "min": round(s[0]), "avg": round(sum(s) / len(s))}


class Req:
    __slots__ = ("agent", "steady", "first", "band", "pod", "t_arrive", "queue_ms", "prompt",
                 "completion", "text", "ttft", "e2e", "vq", "mode", "suspend_after", "joke")


class Sim:
    def __init__(self):
        self.lock = threading.Lock()
        self.rnd = random.Random()
        self.fail_rate = 0.0
        self.suspend_fail_rate = 0.0
        self.pod_up = [True, True]
        # Oct 8: simulation settings (POST api/sim). Pod 0 = the TPU pool, pod 1 = the GPU pool; traffic splits
        # split_tpu_pct / (100 - split_tpu_pct) across them. hardware is what the stack view's Accelerators row shows.
        self.sim = {"split_tpu_pct": 50.0, "hardware": dict(SIM_HARDWARE)}
        self.by_pool_base = {"without": None, "with": None}   # per-pool numbers measured in each routing mode
        # Oct 9: simulated agent trajectories (api/state.trajectories); traj_live = the sessions still taking turns
        self.trajectories = deque(maxlen=TRAJ_KEEP)
        self.traj_live = []
        self.traj_seq = 0
        self.traj_idx = 0
        self.kv_tier = KVTier(self.rnd)     # Oct 10: KV cache tiering (screen 4); see CALIB_KV
        self._reset_all()

    # ------------------------------------------------------------------ setup
    def _reset_all(self):
        now = mono()
        self.agents = bytearray(b"0" * TOTAL)
        self.cnt = {C0: TOTAL, C1: 0, C2: 0, C3: 0, C4: 0, C9: 0}
        self.phase = "idle"
        self.busy = False
        self.note = ""
        self.burst_no = 0
        self.t0 = None                      # no burst yet
        self.bid = ""
        self.events = []
        self.evn = 0
        self.totals = {"requests": 0, "replies": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.unique = set()
        self.ticker = deque(maxlen=TICKER_KEEP)
        self.seq = 0
        self.hdr_seq = 0
        self.rate = 0
        self.strategy = "balanced"
        self.routing_mode = "without"
        self.flow_mode = "without"
        self.comp_routing = {"without": None, "with": None}
        self.comp_flow = {"without": None, "with": None}
        self.by_pool_base = {"without": None, "with": None}
        self.tr = {"inflight": 0, "sent": 0, "replies": 0, "failed": 0, "skipped": 0}
        self.p1 = 0.5
        self.arr_acc = 0.0
        self.pressure = 0.0
        self.sat = 0.0
        self.dispatch_log = deque()
        self.done_log = deque()
        self.pod_inflight = [0, 0]
        self.pod_total = [0, 0]
        self.band_queue = [0, 0, 0]
        self.susp_active = 0
        self.susp_wait = deque()
        self.susp_outstanding = 0
        self.drain_deadline = 0.0
        self.history = deque(maxlen=HIST_KEEP)
        self.last_tick = now
        self.next_metrics = now
        self.next_hist = now
        self.metrics = None
        self.sample_unix = unix_ms()
        self.wakes = [0] * TOTAL            # sandbox resumes per agent
        self.awake_acc = [0.0] * TOTAL      # seconds awake (states 1-4) per agent
        self.awake_since = [0.0] * TOTAL
        self.t_start = now
        self.gen = 0
        if hasattr(self, "kv_tier"):
            self.kv_tier.reset()            # Oct 10: a zeroed mock starts the KV tier from "not run yet" too
        self._mem_reset(1)
        self._dyn_reset(now)
        self._metrics(now)

    def _mem_reset(self, gen):
        """Fresh Hermes sessions: new codenames, nobody taught yet (api/memory/reset)."""
        rnd = random.Random(7919 * gen + 17)
        self.gen = gen
        self.codename = ["%s-%s" % (rnd.choice(CODE_ADJ), rnd.choice(CODE_NOUN)) for _ in range(TOTAL)]
        self.m_taught = [False] * TOTAL
        self.m_taught_at = [0.0] * TOTAL     # mono time of the teaching reply
        self.m_awake_at_teach = [0.0] * TOTAL
        self.m_rests = [0] * TOTAL           # suspends since taught
        self.m_ok = [0] * TOTAL
        self.m_fail = [0] * TOTAL
        self.m_survived = [0] * TOTAL        # survived count at the last correct recall
        self.m_recalled = [False] * TOTAL    # recalled correctly after >= 1 suspend
        self.m_turns = [0] * TOTAL
        self.m_last = [deque(maxlen=4) for _ in range(TOTAL)]
        self.m_prompt = [0] * TOTAL
        self.m_cached = [0] * TOTAL
        self.m_taught_n = self.m_ok_n = self.m_fail_n = self.m_recalled_n = self.m_max_surv = 0
        self.m_calls = self.m_call_fail = 0
        self._traj_clear()                   # old codenames would be stale in the trajectory view

    # ------------------------------------------------------------ Oct 9: agent trajectories (mock)
    def _traj_clear(self):
        if getattr(self, "trajectories", None) is not None:
            self.trajectories.clear()
            self.traj_live = []

    def _traj_pool_latency(self, pool, mode, turn, rnd):
        """(ttft_ms, ms_per_token, cache_hit) for one simulated turn, from CALIB via the pool's current dynamic state
        (so the GPU sessions follow the KV-hit climb after a toggle). A session's first turn always misses (fresh
        per-session prefix); later turns hit with the probability that reproduces the pool's hit rate over CALIB
        turns (round-robin 29 % -> 39 % of turns 2-4; llm-d router 75 % -> every later turn; TPU 50 % -> 2 in 3)."""
        d = CALIB["dyn"]
        v = self.pool_now.get(pool)
        if v is None or v["tok_s"] <= 0.0:
            c = pool_calib(pool, mode or "without")
            v = {"ttft_p50_ms": c["ttft_p50_ms"], "decode_ms": c["decode_ms"], "cache_hit_pct": c["cache_hit_pct"]}
        turns = float(CALIB["turns"])
        p_hit = min(1.0, (v["cache_hit_pct"] / 100.0) * turns / max(1.0, turns - 1.0))
        hit = turn > 1 and rnd.random() < p_hit
        ttft = v["ttft_p50_ms"] * (d["cold_ttft_x"] if turn == 1 else d["warm_hit_ttft_x"] if hit else d["warm_miss_ttft_x"])
        tok_ms = v["decode_ms"] / float(CALIB["out_tok"])
        ttft = max(12.0, ttft * rnd.uniform(0.88, 1.14))
        return ttft, tok_ms, hit

    def _traj_spawn(self, now):
        rnd = self.rnd
        self.traj_seq += 1
        self.traj_idx += 1
        i = (self.traj_idx * 37 + 11) % TOTAL
        pool = POOL_KEYS[self.traj_seq % 2]          # alternate: half TPU, half GPU
        ab_pool = self.sim["hardware"].get("ab_pool") or "gpu"
        tr = {
            "id": "%s-g%d-%d" % (agent_name(i), self.gen, self.traj_seq),
            "agent": agent_name(i), "codename": self.codename[i],
            "pool": pool, "mode": self.routing_mode if pool == ab_pool else None,
            "tier": TRAJ_TIERS[i % 3],
            "started_ms": int(round(self._ms(now))),
            "done": False,
            "turns": [],
            "_next": now + rnd.uniform(0.2, 1.2),
            "_task": rnd.randrange(len(TRAJ_TASKS)),
        }
        self.trajectories.append(tr)
        self.traj_live.append(tr)

    def _traj_step(self, now):
        """Advance the simulated sessions: a turn every ~1-3 s, TRAJ_TURNS turns each, new sessions replace done ones.
        Runs inside the tick lock; nothing moves while the fleet is not running (suspended = blank, like the ticker)."""
        if not (self.phase == "running" and self.rate > 0 and self.t0 is not None):
            return
        rnd = self.rnd
        while len(self.traj_live) < TRAJ_LIVE:
            self._traj_spawn(now)
        still = []
        for tr in self.traj_live:
            if now < tr["_next"]:
                still.append(tr)
                continue
            turn = len(tr["turns"]) + 1
            if turn == 1:
                user, reply = TRAJ_TASKS[tr["_task"]]
            else:
                user, reply = TRAJ_FOLLOWUPS[(tr["_task"] + turn) % len(TRAJ_FOLLOWUPS)]
            # the GPU sessions follow the toggle turn by turn (a session can straddle a mode switch)
            mode = self.routing_mode if tr["mode"] is not None else None
            if mode is not None:
                tr["mode"] = mode
            ttft, tok_ms, hit = self._traj_pool_latency(tr["pool"], mode, turn, rnd)
            out_tok = CALIB["out_tok"] - rnd.choice((0, 0, 0, 0, 1, 3))   # max_tokens; the model rarely stops early
            tr["turns"].append({
                "turn": turn, "user": user[:90], "reply": reply[:140],
                "ttft_ms": int(round(ttft)), "e2e_ms": int(round(ttft + out_tok * tok_ms * rnd.uniform(0.9, 1.1))),
                "out_tok": out_tok, "prompt_tok": CALIB["prefix_tok"] + CALIB["turn_tok"] * (turn - 1) + rnd.randint(0, 25),
                "t_ms": int(round(self._ms(now))), "cache_hit": bool(hit),
            })
            if turn >= TRAJ_TURNS:
                tr["done"] = True
            else:
                tr["_next"] = now + rnd.uniform(1.0, 3.0)
                still.append(tr)
        self.traj_live = still

    def _traj_json(self):
        """Newest first, private bookkeeping keys stripped, turns copied."""
        return [dict(((k, v) for k, v in tr.items() if not k.startswith("_")), turns=list(tr["turns"]))
                for tr in reversed(self.trajectories)]

    def _mock_pods(self, key, hit, now, active, agents=0, h=0.0):
        """Per-pod KV cards for one pool (Oct 9/10): tpu_pods / gpu_pods simulated pods. Round-robin spreads the pool's
        agents evenly (in flight ~agents/n each, hit within a few points of the pool); the llm-d router's prefix
        affinity concentrates sessions (pod j carries ~taper^j of pod 0, e.g. 8/6/5/5 of 24) and the busier pods hit
        more. h = router progress 0..1 (follows the KV-hit climb), so the cards move smoothly with the toggle.
        `running` sums to the agents on the pool minus 0-2 between requests. Idle: hit None, running 0."""
        n = max(1, int(self.sim["hardware"].get(key + "_pods") or (1 if key == "tpu" else 2)))
        st = self.dyn["pod"].setdefault(key, {"off": [], "swap": (0, 0), "short": 0})
        while len(st["off"]) < n:
            st["off"].append(0.0)
        del st["off"][n:]
        if hit is None or not active:
            return [{"name": "pod-%d" % (j + 1), "cache_hit_pct": None, "running": 0} for j in range(n)]
        taper = CALIB["router_pod_taper"]
        w = [taper ** j for j in range(n)]
        tw = sum(w)
        w = [x / tw for x in w]
        share = [(1.0 - h) / n + h * w[j] for j in range(n)]
        # in flight: largest-remainder split of the pool's agents, minus a few between requests, +1/-1 swap per second
        want = [agents * s for s in share]
        run = [int(math.floor(x)) for x in want]
        rest = int(round(agents)) - sum(run)
        for j in sorted(range(n), key=lambda j: want[j] - run[j], reverse=True)[:max(0, rest)]:
            run[j] += 1
        for _ in range(st["short"]):
            j = max(range(n), key=lambda j: run[j])
            if run[j] > 0:
                run[j] -= 1
        a, b = st["swap"]
        if n > 1 and a != b and a < n and b < n and run[a] > 0:
            run[a] -= 1
            run[b] += 1
        # hit: the pool figure + affinity offset (busier pods hit more) + slow per-pod drift, re-centred on the pool
        off = [h * CALIB["router_pod_hit_x"] * (w[j] - 1.0 / n) + st["off"][j] for j in range(n)]
        mean = sum(share[j] * off[j] for j in range(n))
        pods = []
        for j in range(n):
            v = round(max(1.0, min(99.0, hit + off[j] - mean)), 1)
            pods.append({"name": "pod-%d" % (j + 1), "cache_hit_pct": v, "running": max(0, run[j])})   # no addresses in the public state
        return pods

    # ------------------------------------------------------------ Oct 10: calibrated pool dynamics
    def _dyn_reset(self, now):
        """Fresh dynamic state: nothing loaded, KV cold, router progress where the current mode says."""
        self.dyn = {"t": now, "load": 0.0, "warm": 0.0, "h": 1.0 if self.routing_mode == "with" else 0.0,
                    "mode_since": now, "noise": {}, "noise_t": now, "spike": 1.0, "pub": {}, "pub_t": 0.0,
                    "chip": {}, "pod": {}}
        self.pool_now = {}
        self.ab_hist = None

    def _ar1(self, name, sigma, phi=0.7):
        x = phi * self.dyn["noise"].get(name, 0.0) + self.rnd.gauss(0.0, sigma)
        self.dyn["noise"][name] = x
        return x

    def _dyn_step(self, now, active):
        """Integrate load / KV warm-up / router progress (first-order, CALIB time constants) and, once per second,
        the filtered random walks that make 1 s samples jitter without anything periodic."""
        d, nz, dy = CALIB["dyn"], CALIB["noise"], self.dyn
        dt = max(0.0, min(1.0, now - dy["t"]))
        dy["t"] = now
        step = lambda x, tgt, tau: x + (tgt - x) * (1.0 - math.exp(-dt / tau)) if dt > 0 else x
        if active:
            dy["load"] = step(dy["load"], 1.0, d["tau_load_s"])
            dy["warm"] = step(dy["warm"], 1.0, d["tau_warm_s"])
        else:
            dy["load"] = step(dy["load"], 0.0, d["tau_off_s"])
            dy["warm"] = step(dy["warm"], 0.0, d["tau_off_s"])
            if dy["load"] < 0.01:
                dy["load"] = dy["warm"] = 0.0
        tgt = 1.0 if self.routing_mode == "with" else 0.0
        dy["h"] = step(dy["h"], tgt, d["tau_up_s"] if tgt > dy["h"] else d["tau_down_s"])
        if now >= dy["noise_t"]:
            dy["noise_t"] = max(dy["noise_t"] + 1.0, now - 1.0)
            for key in POOL_KEYS:
                for k in ("tok", "ttft50", "ttft95", "e2e50", "e2e95"):
                    self._ar1(key + "." + k, nz[k])
                self._ar1(key + ".hit", nz["hit_pts"])
                n_chips = max(1, int(self.sim["hardware"].get(key + "_chips") or 1))
                ch = dy["chip"].setdefault(key, [])
                while len(ch) < n_chips:
                    ch.append(self.rnd.gauss(0.0, 0.04))
                del ch[n_chips:]
                for c in range(n_chips):
                    ch[c] = nz["chip_phi"] * ch[c] + self.rnd.gauss(0.0, nz["chip_sigma"])
                st = dy["pod"].get(key)
                if st:
                    st["off"] = [max(-nz["pod_max"], min(nz["pod_max"], nz["pod_phi"] * x + self.rnd.gauss(0.0, nz["pod_sigma"]))) for x in st["off"]]
                    n = len(st["off"])
                    st["short"] = self.rnd.choice((0, 0, 1, 1, 2))
                    st["swap"] = (self.rnd.randrange(n), self.rnd.randrange(n)) if n > 1 else (0, 0)
            for k in ("int", "std", "bg"):
                self._ar1("flow." + k, nz["flow"])
            dy["spike"] = self.rnd.uniform(*nz["spike"]) if self.rnd.random() < nz["spike_p"] else 1.0

    def _pool_model(self, key, n_agents):
        """Noise-free state of one pool right now: interpolates between its calibrated round-robin and router states
        by the KV hit rate (which climbs with router progress h and warms up after a wake), scaled by load/agents."""
        dy = self.dyn
        rr, wi = pool_calib(key, "without"), pool_calib(key, "with")
        span = wi["cache_hit_pct"] - rr["cache_hit_pct"]
        hit = (rr["cache_hit_pct"] + span * dy["h"]) * dy["warm"]
        if span > 0:                     # the A/B pool: TTFT / E2E / tok_s follow the hit rate (a cold start sits below RR)
            he = max(-0.7, min(1.0, (hit - rr["cache_hit_pct"]) / span))
            cold = 1.0
        else:                            # not part of the A/B: only the cold start moves it
            he = 0.0
            cold = 1.0 + (CALIB["dyn"]["cold_ttft_x"] - 1.0) * (1.0 - dy["warm"])
        lerp = lambda k: rr[k] + (wi[k] - rr[k]) * he
        ft, fl = agents_scale(n_agents * dy["load"])
        fl *= cold
        return {"tok_s": lerp("tok_s") * ft, "cache_hit_pct": hit,
                "ttft_p50_ms": lerp("ttft_p50_ms") * fl, "ttft_p95_ms": lerp("ttft_p95_ms") * fl,
                "e2e_p50_ms": lerp("e2e_p50_ms") * fl, "e2e_p95_ms": lerp("e2e_p95_ms") * fl,
                "decode_ms": lerp("decode_ms") * fl,
                "tok_steady": lerp("tok_s") * agents_scale(n_agents)[0] * (1.0 if n_agents else 0.0)}

    def _pool_sample(self, key, n_agents):
        """The pool model plus this second's jitter (what by_pool, history and the pod cards show)."""
        v = self._pool_model(key, n_agents)
        nz = self.dyn["noise"]
        g = lambda k: nz.get(key + "." + k, 0.0)
        return {"tok_s": max(0.0, v["tok_s"] * (1.0 + g("tok"))),
                "cache_hit_pct": max(0.0, min(99.0, v["cache_hit_pct"] + g("hit") * min(1.0, v["cache_hit_pct"] / 10.0))),
                "ttft_p50_ms": v["ttft_p50_ms"] * (1.0 + g("ttft50")),
                "ttft_p95_ms": v["ttft_p95_ms"] * (1.0 + g("ttft95")) * self.dyn["spike"],
                "e2e_p50_ms": v["e2e_p50_ms"] * (1.0 + g("e2e50")),
                "e2e_p95_ms": v["e2e_p95_ms"] * (1.0 + g("e2e95")) * (1.0 + (self.dyn["spike"] - 1.0) * 0.5),
                "decode_ms": v["decode_ms"], "tok_steady": v["tok_steady"]}

    def _pool_agents(self, share):
        """Closed-loop agents on each pool: 2 x CALIB agents_per_pool split by the configured share (50/50 = 24 each)."""
        total = 2 * CALIB["agents_per_pool"]
        a = int(round(total * share[0]))
        return [a, total - a]

    def _routing_json(self, mode):
        """compare.routing.<mode>: the published window plus req_s = tok_s / output tokens per turn."""
        c = self.comp_routing.get(mode)
        if not c:
            return None
        c = dict(c)
        if mode == "with" and hasattr(self, "kv_tier") and self.kv_tier.enabled and self.kv_tier.x > 0.001:
            # Oct 13: KV tiering is a router feature; with the tier on, the router's headline carries the lift
            x, g = self.kv_tier.x, CALIB_KV["cluster_gain"]
            c["tok_s"] = int(round(c["tok_s"] * (1.0 + g["tok_s"] * x)))
            for k in ("ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms", "e2e_p95_ms"):
                c[k] = int(round(c[k] * (1.0 - g["lat"] * x)))
            c["cache_hit_pct"] = round(min(99.0, c["cache_hit_pct"] + g["hit_pts"] * x), 1)
        return dict(c, req_s=round(c["tok_s"] / float(CALIB["out_tok"]), 1))

    def _new_burst(self, bid, t, n, hold, wake_only=False):
        """Fresh per-burst bookkeeping (the driver's newBurst + T0)."""
        self.bid = bid
        self.t0 = t
        self.t0_unix = unix_ms() - int(round((mono() - t) * 1000))
        self.n = n
        self.hold = hold
        self.wake_only = wake_only
        self.traffic_started = not wake_only
        self.duty_cycle = False
        self.traffic_t0 = None
        self.duty_Settled = False
        self.duty_acc = 0.0
        self.ramp_wake_pts = None
        self.ramp_duty_pts = []
        self.next_duty_ramp = None
        self.conc = 0
        self.susp_t0 = None
        self.all_running_ms = None
        self.wake_done_ms = None
        self.first_done_ms = None
        self.all_susp_ms = None
        self.peak = 0
        self.woke = 0
        self.wake_failed = 0
        self.ms_keys = []
        for f in MILESTONE_FRACS:
            k = max(1, int(round(f * n)))
            if k not in self.ms_keys:
                self.ms_keys.append(k)
        self.milestones = {k: None for k in self.ms_keys}
        self.wake_lat = []
        self.wake_start = [0.0] * TOTAL
        self.run_ts = []                    # sorted ms since t0 at which burst agents became running
        self.susp_ts = []                   # sorted ms since t0 at which burst agents were suspended (ok)
        self.done_recs = []                 # sorted (ms since t0, tokens) of first replies (ok)
        self.ramp_frozen = None
        self.life_pending = n
        self.wake_wait = deque()
        self.wake_active = 0
        self.stragglers = set()

    def _set(self, i, c):
        old = self.agents[i]
        if old != c:
            self.cnt[old] -= 1
            self.cnt[c] += 1
            self.agents[i] = c
            was_up, is_up = old in UPSET, c in UPSET
            if is_up and not was_up:
                self.wakes[i] += 1
                self.awake_since[i] = mono()
            elif was_up and not is_up:
                self.awake_acc[i] += mono() - self.awake_since[i]
                if c == C0 and self.m_taught[i]:
                    self.m_rests[i] += 1        # snapshot to zero compute: the codename must survive this

    def _awake_s(self, i, now):
        a = self.awake_acc[i]
        if self.agents[i] in UPSET:
            a += now - self.awake_since[i]
        return a

    def _up(self):
        """Sandboxes up, like the driver's d.up: running, requesting and suspending (plus waking during duty cycle)."""
        if getattr(self, "duty_cycle", False):
            return self.cnt[C1] + self.cnt[C2] + self.cnt[C3] + self.cnt[C4]
        return self.cnt[C2] + self.cnt[C3] + self.cnt[C4]

    def _idle_pct(self):
        return round(100.0 * (TOTAL - self._up()) / float(TOTAL), 1)

    def _ms(self, t):
        return (t - self.t0) * 1000.0 if self.t0 is not None else 0.0

    def _sched(self, t, kind, payload):
        heapq.heappush(self.events, (t, self.evn, kind, payload))
        self.evn += 1

    def _preflight_s(self, rest, dead):
        rnd = self.rnd
        d = rnd.uniform(*PREFLIGHT_S)
        if rest:
            d += math.ceil(rest / float(SUSPEND_CAP)) * rnd.uniform(*SUSPEND_S) + rnd.uniform(*PREFLIGHT_S)
        if dead:
            d += math.ceil(dead / float(SUSPEND_CAP)) * RECREATE_S
        return d + rnd.uniform(*PREFLIGHT_S)

    def _leftovers(self):
        return self.cnt[C1] + self.cnt[C2] + self.cnt[C3] + self.cnt[C4], self.cnt[C9]

    # --------------------------------------------------------------- controls
    def start_burst(self, body):
        hold = body.get("hold")
        hold = hold if isinstance(hold, bool) else True
        wake_only = bool(body.get("wake_only"))
        n = body.get("agents")
        n = int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) else 0
        if n <= 0 or n > TOTAL:
            n = TOTAL
        conc = body.get("concurrency")
        conc = int(conc) if isinstance(conc, (int, float)) and not isinstance(conc, bool) and conc >= 0 else 0
        with self.lock:
            if self.busy or self.phase != "idle":
                return False, "busy (phase %s)" % self.phase
            self.busy = True
            self.burst_no += 1
            bid = "b-%s-%04d" % (RUN_TAG, self.burst_no)
            now = mono()
            rest, dead = self._leftovers()
            pre = self._preflight_s(rest, dead)
            self._sched(now + pre, "preflight_done", None)
            self._sched(now + pre + GATE_S, "begin", (bid, n, hold, conc, wake_only))
            return True, bid

    def _seed_comparison_history(self):
        """Traffic begins (agents woke): start the calibrated dynamics from cold - load and KV hit ramp up over the next
        ~10-15 s like the live pools, nothing is pre-filled. compare.routing / compare.flow for the current mode are
        published by _metrics once the mode has had a full window (publish_warm_s); earlier measurements of the other
        mode are kept (they stay the 'vs round-robin' baseline until a reconcile or a mock reset clears them)."""
        now = mono()
        self.dyn["load"] = self.dyn["warm"] = 0.0
        self.dyn["h"] = 1.0 if self.routing_mode == "with" else 0.0
        self.dyn["mode_since"] = now
        self.dyn["pub_t"] = 0.0
        for k in ("without", "with"):
            if self.comp_routing[k]:
                self.comp_routing[k]["live"] = False
            if self.comp_flow[k]:
                self.comp_flow[k]["live"] = False

    def _start_duty_cycle(self, now):
        """Transition awake fleet into ~90% idle duty cycle (staggered request -> suspend -> random resume)."""
        if self.ramp_wake_pts is None and self.t0 is not None:
            end = self.wake_done_ms if self.wake_done_ms is not None else max(1000.0, (now - self.t0) * 1000.0)
            npts = min(max(1, int(math.ceil(end / 1000.0))), 10)
            self.ramp_wake_pts = []
            for k in range(npts):
                t = (k + 1) * 1000
                tt = min(t, end)
                woke = bisect.bisect_right(self.run_ts, tt)
                self.ramp_wake_pts.append({"t_ms": t, "running": woke, "woke": woke, "replies": 0, "tokens": 0})
        self.traffic_started = True
        self.duty_cycle = True
        self.traffic_t0 = now
        self.duty_Settled = False
        self.duty_acc = 0.0
        self.ramp_frozen = None
        self.next_duty_ramp = now + 1.0
        self._seed_comparison_history()
        running_idxs = [i for i in range(self.n) if self.agents[i] in (C1, C2)]
        self.rnd.shuffle(running_idxs)
        # Stagger initial wave: first 100 agents stagger a request over 0..0.9s then suspend;
        # remaining ~900 stagger-suspend over 0.03..1.5s so the fleet settles smoothly to ~100 active (~90% idle).
        for rank, i in enumerate(running_idxs):
            if rank < 100:
                delay = 0.01 + 0.85 * (rank / 100.0) + self.rnd.uniform(0.0, 0.08)
                self._sched(now + delay, "duty_send", (i, True))
            else:
                delay = 0.02 + 1.45 * ((rank - 100) / max(1.0, float(len(running_idxs) - 100))) + self.rnd.uniform(0.0, 0.12)
                self._sched(now + delay, "duty_park", i)

    def simulate_traffic(self, body):
        stop = bool(body.get("stop"))
        toggle = bool(body.get("toggle"))
        with self.lock:
            if self.phase not in ("running", "waking"):
                return False, "wake agents first (phase %s)" % self.phase
            if stop or (toggle and self.rate > 0 and getattr(self, "duty_cycle", False) and getattr(self, "manual_traffic_toggle", False)):
                self.rate = 0
                self.manual_traffic_toggle = False
                return True, 0
            self.rate = 300
            self.manual_traffic_toggle = True
            now = mono()
            if self.t0 is not None and not getattr(self, "duty_cycle", False):
                self._start_duty_cycle(now)
            return True, self.rate

    def set_traffic(self, body):
        rate = body.get("rate")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not (0 <= rate <= 2000):
            return False, "rate must be 0..2000"
        with self.lock:
            self.rate = int(rate) if float(rate).is_integer() else float(rate)
            if self.rate > 0 and self.phase == "running" and self.t0 is not None and not getattr(self, "duty_cycle", False):
                self._start_duty_cycle(mono())
            return True, self.rate

    def _set_routing(self, routing):
        """Routing toggle: the KV hit (and with it TTFT / E2E / tok_s) moves from here with CALIB time constants."""
        if routing != self.routing_mode:
            self.routing_mode = routing
            self.dyn["mode_since"] = mono()
            self.dyn["pub_t"] = 0.0

    def set_strategy(self, body):
        mode = body.get("mode")
        routing = body.get("routing")
        flow = body.get("flow")
        with self.lock:
            if routing in ("without", "with"):
                self._set_routing(routing)
                if routing == "without" and hasattr(self, "kv_tier"):
                    self.kv_tier.set_mode("without")   # Oct 13: the KV tier is a feature of the llm-d router; no router, no tier
                if self.phase == "running" and self.t0 is not None and not getattr(self, "duty_cycle", False):
                    self.rate = 300
                    self._start_duty_cycle(mono())
                return True, self.routing_mode
            if flow in ("without", "with"):
                self.flow_mode = flow
                self.strategy = "priority" if flow == "with" else "balanced"
                if self.phase == "running":
                    self.rate = 300
                    if self.t0 is not None and not getattr(self, "duty_cycle", False):
                        self._start_duty_cycle(mono())
                return True, self.flow_mode
            if mode in ("without_llmd", "with_llmd"):
                self._set_routing("with" if mode == "with_llmd" else "without")
                return True, self.routing_mode
            if mode in ("without_flow", "with_flow"):
                self.flow_mode = "with" if mode == "with_flow" else "without"
                self.strategy = "priority" if self.flow_mode == "with" else "balanced"
                if self.phase == "running" and self.rate > 0:
                    self.rate = 300
                return True, self.flow_mode
            if mode not in ("balanced", "steer8020", "priority"):
                return False, "mode must be balanced, steer8020, priority, without_llmd, with_llmd, without_flow, or with_flow"
            self.strategy = mode
            self.flow_mode = "with" if mode == "priority" else "without"
            if self.phase == "running" and self.rate > 0:
                self.rate = 300
            return True, mode

    def set_kv_tier(self, body):
        """POST api/kv_tier {"mode": "without"|"with"} (Oct 10, screen 4). Mock-only extra: {"enabled": bool} to
        rehearse the empty state of a pool deployed without the tier. -> result: {"mode", "enabled"}."""
        with self.lock:
            if "enabled" in body:
                self.kv_tier.enabled = bool(body["enabled"])
            if "mode" in body:
                ok, res = self.kv_tier.set_mode(body.get("mode"))
                if not ok:
                    return False, res
            return True, {"mode": self.kv_tier.mode, "enabled": self.kv_tier.enabled}

    def suspend(self, body):
        with self.lock:
            if self.busy or self.phase in ("waking", "suspending"):
                return False, "busy (phase %s)" % self.phase
            self.busy = True
            self.phase = "suspending"
            self.rate = 0
            self.manual_traffic_toggle = False
            now = mono()
            self.drain_deadline = now + DRAIN_MAX_S
            self._sched(now, "drain", None)
            return True, None

    def reconcile(self, body):
        with self.lock:
            if self.busy or self.phase != "idle":
                return False, "busy (phase %s)" % self.phase
            self.busy = True
            self.phase = "reconciling"
            self.rate = 0
            rest, dead = self._leftovers()
            dur = max(RECONCILE_MIN_S, self._preflight_s(rest, dead))
            self._sched(mono() + dur, "recon_done", (rest, dead))
            return True, None

    def memory_reset(self, body):
        with self.lock:
            if HARNESS != "hermes":
                return False, "memory reset needs --harness hermes"
            if self.busy or self.phase != "idle":
                return False, "busy (phase %s)" % self.phase
            self._mem_reset(self.gen + 1)
            self.note = "memory reset: generation %d, new codenames and fresh Hermes sessions" % self.gen
            return True, self.gen

    def mock_control(self, body):
        with self.lock:
            if body.get("reset"):
                keep = (self.fail_rate, self.suspend_fail_rate, list(self.pod_up))
                self._reset_all()
                self.fail_rate, self.suspend_fail_rate, self.pod_up = keep
            if "fail_rate" in body:
                self.fail_rate = max(0.0, min(1.0, float(body["fail_rate"])))
            if "suspend_fail_rate" in body:
                self.suspend_fail_rate = max(0.0, min(1.0, float(body["suspend_fail_rate"])))
            if "pod_up" in body and isinstance(body["pod_up"], list) and len(body["pod_up"]) == 2:
                self.pod_up = [bool(x) for x in body["pod_up"]]
            return True, None

    def set_sim(self, body):
        """POST api/sim  {"split_tpu_pct": 50} | {"hardware": {tpu_chip, tpu_detail, tpu_topology, tpu_chips, gpu_chip, gpu_detail, gpu_topology, gpu_chips}}
        -> result: the effective settings. Simulation only; the live server has no such endpoint.
        Pod A = the tpu_* fields, Pod B = the gpu_* fields (the keys are the live config.hardware contract; the
        settings drawer labels them Pod A / Pod B). A chip count of 0 with a topology set derives the count from it."""
        with self.lock:
            if "split_tpu_pct" in body:
                try:
                    self.sim["split_tpu_pct"] = max(0.0, min(100.0, float(body["split_tpu_pct"])))
                except (TypeError, ValueError):
                    return False, "split_tpu_pct must be a number 0..100"
            hw = body.get("hardware")
            if isinstance(hw, dict):
                cur = self.sim["hardware"]
                for k in ("tpu_chip", "tpu_detail", "tpu_topology", "gpu_chip", "gpu_detail", "gpu_topology"):
                    if k in hw:
                        cur[k] = str(hw[k] or "").strip()
                for k in ("tpu_chips", "gpu_chips"):
                    if k in hw:
                        try:
                            cur[k] = max(0, int(hw[k] or 0))
                        except (TypeError, ValueError):
                            return False, "%s must be an integer" % k
                for p in ("tpu", "gpu"):   # chips follow the topology unless given explicitly
                    if cur.get(p + "_topology") and (p + "_topology" in hw) and not (hw.get(p + "_chips") or 0):
                        cur[p + "_chips"] = topo_chips(cur[p + "_topology"])
            return True, {"split_tpu_pct": self.sim["split_tpu_pct"], "hardware": dict(self.sim["hardware"])}

    # ------------------------------------------------------------ burst events
    def _ev_preflight_done(self, t, _):
        for i in range(TOTAL):              # leftovers put to rest / re-created (applyStates at the end)
            if self.agents[i] != C0:
                self._set(i, C0)

    def _ev_begin(self, t, p):
        bid, n, hold, conc, wake_only = p
        self._new_burst(bid, t, n, hold, wake_only)
        self.manual_traffic_toggle = False
        self.conc = conc
        self.totals = {"requests": 0, "replies": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.unique = set()
        self.ticker.clear()
        self._traj_clear()
        self.note = ""
        self.phase = "waking"
        order = list(range(n))
        self.rnd.shuffle(order)
        self.wake_wait = deque(order)
        self.stragglers = set(self.rnd.sample(range(n), min(n, self.rnd.randint(3, 6))))
        self._wake_fill(t)

    def _wake_fill(self, t):
        rnd = self.rnd
        while self.wake_wait and (self.conc <= 0 or self.wake_active < self.conc):
            i = self.wake_wait.popleft()
            self.wake_active += 1
            self._set(i, C1)
            self.wake_start[i] = self._ms(t)
            if i in self.stragglers:
                lat = rnd.uniform(1.25, 1.6)
            else:
                lat = logistic(rnd, 0.60, 0.105)
                lat = min(max(lat, 0.12 + rnd.random() * 0.05), 1.16 + rnd.random() * 0.05)
            self._sched(t + lat, "wake", i)

    def _ev_wake(self, t, i):
        self.wake_active -= 1
        self._wake_fill(t)
        if self.agents[i] != C1:
            return
        ms = self._ms(t)
        if self.fail_rate and self.rnd.random() < self.fail_rate:
            self._set(i, C9)
            self.wake_failed += 1
            self.note = "%s wake failed: mock failure (fail_rate)" % agent_name(i)
            self._check_wake_done(ms)
            if not (self.wake_only and not self.traffic_started):
                self._life_done(t)
            return
        if self.wake_only and not self.traffic_started:
            self._set(i, C2)
        else:
            self._set(i, C3)
        self.woke += 1
        bisect.insort(self.run_ts, ms)
        self.wake_lat.append(ms - self.wake_start[i])
        up = self._up()
        if up > self.peak:
            self.peak = up
        for k in self.ms_keys:
            if up >= k and self.milestones[k] is None:
                self.milestones[k] = ms
        if up >= self.n and self.all_running_ms is None:
            self.all_running_ms = ms
        self._check_wake_done(ms)
        if not (self.wake_only and not self.traffic_started):
            r = self._new_req(i, False, t, suspend_after=getattr(self, "duty_cycle", False))
            self._dispatch(t, r, self.rnd.uniform(0.2, 2.5), e2e=self.rnd.uniform(150, 900))

    def _check_wake_done(self, ms):
        if self.woke + self.wake_failed >= self.n and self.wake_done_ms is None:
            self.wake_done_ms = ms
            if self.hold:
                self.phase = "running"
                if self.wake_only and not self.traffic_started:
                    self.busy = False
                elif self.rate == 0:
                    self.rate = 300

    def _life_done(self, t):
        """One agent goroutine finished (woke + first reply, or failed; one-shot: also suspended)."""
        self.life_pending -= 1
        if self.life_pending <= 0 and self.first_done_ms is None:
            self.first_done_ms = self._ms(t)
            self.busy = False
            if not self.hold:
                self.phase = "idle"
            else:
                if self.phase == "waking":
                    self.phase = "running"
                if self.phase == "running" and not (self.wake_only and not self.traffic_started):
                    if self.rate == 0:
                        self.rate = 300
                    if not getattr(self, "duty_cycle", False):
                        self._start_duty_cycle(t)

    # -------------------------------------------------------- suspend events
    def _ev_drain(self, t, _):
        if self.tr["inflight"] > 0 and t < self.drain_deadline:
            self._sched(t + 0.02, "drain", None)
            return
        if self.t0 is None:                 # no burst yet: the driver creates a bookkeeping burst
            self.burst_no += 1
            self._new_burst("s-%s-%04d" % (RUN_TAG, self.burst_no), t, TOTAL, True)
        self.duty_cycle = False
        self.susp_t0 = t
        self.all_susp_ms = None
        idx = [i for i in range(TOTAL) if self.agents[i] in (C1, C2, C3, C4)]
        if self._up() == 0:
            self.all_susp_ms = 0.0
        self.rnd.shuffle(idx)
        self.susp_outstanding = len(idx)
        for i in idx:
            self._susp_request(t, i)
        if not idx:
            self._finish_suspend(t)

    def _susp_request(self, t, i):
        if self.susp_active < SUSPEND_CAP:
            self._susp_start(t, i)
        else:
            self.susp_wait.append(i)

    def _susp_start(self, t, i):
        self.susp_active += 1
        self._set(i, C4)
        self._sched(t + self.rnd.uniform(*SUSPEND_S), "sdone", i)

    def _ev_sdone(self, t, i):
        self.susp_active -= 1
        if self.susp_wait:
            self._susp_start(t, self.susp_wait.popleft())
        if self.agents[i] == C4:
            if self.suspend_fail_rate and self.rnd.random() < self.suspend_fail_rate:
                self._set(i, C2)
                self.note = "%s suspend failed: mock failure (suspend_fail_rate)" % agent_name(i)
            else:
                self._set(i, C0)
                if self.t0 is not None:
                    bisect.insort(self.susp_ts, self._ms(t))
        if self.susp_t0 is not None and self.all_susp_ms is None and self._up() == 0:
            self.all_susp_ms = (t - self.susp_t0) * 1000.0
        if self.phase == "suspending":
            self.susp_outstanding -= 1
            if self.susp_outstanding <= 0:
                self._finish_suspend(t)
        elif self.t0 is not None and not self.hold:
            self._life_done(t)

    def _finish_suspend(self, t):
        done_ms = (t - self.susp_t0) * 1000.0 if self.susp_t0 is not None else 0.0
        if self._up() <= 0 and self.all_susp_ms is None:
            self.all_susp_ms = done_ms
        self.phase = "idle"
        self.busy = False
        self._traj_clear()                  # suspended fleet = no agent sessions in flight (live server does the same)

    def _ev_recon_done(self, t, p):
        rest, dead = p
        for i in range(TOTAL):
            if self.agents[i] != C0:
                self._set(i, C0)
        self.comp_routing = {"without": None, "with": None}
        self.comp_flow = {"without": None, "with": None}
        self.by_pool_base = {"without": None, "with": None}
        self.history.clear()
        self._dyn_reset(t)
        self.note = ("preflight: put %d leftovers to rest (0 failed), re-created %d (0 failed); "
                     "at T0: %d suspended, 0 paused, 0 not at rest" % (rest, dead, TOTAL))
        self.phase = "idle"
        self.busy = False

    # ------------------------------------------------------------ LLM requests
    def _new_req(self, i, steady, t, suspend_after=False):
        rnd = self.rnd
        r = Req()
        r.agent, r.steady, r.first, r.t_arrive, r.mode = i, steady, not steady, t, self.strategy
        r.suspend_after = suspend_after
        self.hdr_seq += 1
        m = self.hdr_seq % 5
        r.band = 0 if m == 0 else (2 if m == 4 else STANDARD)
        r.prompt = rnd.randint(284, 297)
        r.joke = rnd.choice(JOKES)
        r.text = (rnd.choice(OPENERS) + r.joke + rnd.choice(CLOSERS)).strip()
        r.completion = max(20, min(45, int(len(r.text) / 4.0) + rnd.randint(10, 20)))
        return r

    def _service_latency(self, pod, completion):
        """One substrate request on pod 0 (TPU pool) / 1 (GPU pool): TTFT and E2E drawn around the pool's current
        calibrated state (log-normal around the p50), so the ticker's latency_ms reads like the measured E2E."""
        key = POOL_KEYS[pod]
        v = self.pool_now.get(key)
        if v is None or v["tok_s"] <= 0.0:
            c = pool_calib(key, self.routing_mode)
            v = {"ttft_p50_ms": c["ttft_p50_ms"], "decode_ms": c["decode_ms"]}
        rnd = self.rnd
        ttft = max(12.0, v["ttft_p50_ms"] * math.exp(rnd.gauss(0.0, 0.35)))
        e2e = ttft + v["decode_ms"] * (completion / float(CALIB["out_tok"])) * rnd.uniform(0.85, 1.2)
        return ttft, max(ttft + 10.0, e2e)

    def _queue_wait_ms(self, band):
        base = BAND_WAIT_MS[band] if self.strategy == "priority" else SHARED_WAIT_MS
        return base * self.pressure * self.rnd.uniform(0.55, 1.45)

    def _dispatch(self, t, r, queue_ms, e2e=None):
        rnd = self.rnd
        r.queue_ms = queue_ms
        up = self.pod_up
        if up[0] and up[1]:
            pod = 0 if rnd.random() < self.p1 else 1
        elif up[0] or up[1]:
            pod = 0 if up[0] else 1
        else:
            self._fail(t, r)
            return
        r.pod = pod
        self.pod_inflight[pod] += 1
        self.dispatch_log.append((t, pod, r.band, queue_ms))
        if e2e is None:
            r.ttft, r.e2e = self._service_latency(pod, r.completion)
            r.vq = rnd.uniform(0.5, 4.0)
        else:
            r.e2e = e2e
            r.ttft = e2e * rnd.uniform(0.22, 0.4)
            r.vq = max(0.0, e2e - 260.0) * rnd.uniform(0.4, 0.6)
        self._sched(t + r.e2e / 1000.0, "reply", r)

    def _req_finished(self, t, r):
        i = r.agent
        if getattr(r, "suspend_after", False):
            if self.phase == "running" and self.agents[i] in (C2, C3):
                self._set(i, C2)
                self._sched(t + self.rnd.uniform(0.06, 0.14), "duty_park", i)
            return
        if self.agents[i] == C3 and not getattr(self, "duty_cycle", False):
            self._set(i, C2)
        if r.first and self.t0 is not None:
            if self.hold:
                self._life_done(t)
            else:
                self._susp_request(t, i)    # one-shot burst: every agent suspends right after its reply

    def _ev_duty_wake(self, t, i):
        if self.phase != "running" or self.rate <= 0 or self.agents[i] != C1:
            if self.agents[i] == C1 and self.phase != "waking":
                self._set(i, C0)
            return
        self._set(i, C2)
        self._sched(t + self.rnd.uniform(0.08, 0.16), "duty_send", (i, True))

    def _ev_duty_send(self, t, payload):
        i, suspend_after = payload
        if self.phase != "running" or self.rate <= 0:
            if self.agents[i] in (C1, C2, C3) and self.phase != "suspending":
                self._set(i, C0)
            return
        if suspend_after and self.agents[i] not in (C1, C2):
            return
        if suspend_after:
            self._set(i, C3)
        r = self._new_req(i, True, t, suspend_after=suspend_after)
        self.tr["sent"] += 1
        self.tr["inflight"] += 1
        if self.pressure > 0.02:
            w = self._queue_wait_ms(r.band)
            self.band_queue[r.band] += 1
            if suspend_after:
                # Don't block the sandbox's own ~1s wake-request-suspend visual cycle on a 1.1s best-effort queue wait:
                # schedule its park after a realistic ~0.45s turn while the request finishes through the EPP queue.
                r.suspend_after = False
                self._sched(t + self.rnd.uniform(0.35, 0.55), "duty_park", i)
            self._sched(t + w / 1000.0, "dispatch", r)
        else:
            self._dispatch(t, r, self.rnd.uniform(0.2, 3.0))

    def _ev_duty_park(self, t, i):
        if self.phase != "running":
            return
        if self.agents[i] in (C1, C2, C3):
            self._set(i, C4)
            self._sched(t + self.rnd.uniform(0.12, 0.22), "duty_susp_done", i)

    def _ev_duty_susp_done(self, t, i):
        if self.phase == "suspending":
            return
        if self.agents[i] == C4:
            self._set(i, C0)
            if self.t0 is not None:
                bisect.insort(self.susp_ts, self._ms(t))
            if self._up() <= 115:
                self.duty_Settled = True

    def _pick_suspended(self):
        rnd, a = self.rnd, self.agents
        for _ in range(64):
            i = rnd.randrange(TOTAL)
            if a[i] == C0:
                return i
        if self.cnt[C0] == 0:
            return None
        j = a.find(b"0", rnd.randrange(TOTAL))
        if j < 0:
            j = a.find(b"0")
        return j if j >= 0 else None

    def _pick_active_for_req(self):
        rnd, a = self.rnd, self.agents
        for _ in range(64):
            i = rnd.randrange(TOTAL)
            if a[i] in (C2, C3):
                return i
        for ch in (b"3", b"2"):
            j = a.find(ch, rnd.randrange(TOTAL))
            if j < 0:
                j = a.find(ch)
            if j >= 0:
                return j
        return None

    def _fail(self, t, r):
        self.totals["requests"] += 1
        self.totals["failed"] += 1
        self.note = "%s: mock LLM failure: no inference pod is up" % agent_name(r.agent)
        if HARNESS == "hermes":
            self.m_calls += 1
            self.m_call_fail += 1
        if r.steady:
            self.tr["inflight"] -= 1
            self.tr["failed"] += 1
        self._req_finished(t, r)

    def _ev_dispatch(self, t, r):
        self.band_queue[r.band] -= 1
        self._dispatch(t, r, (t - r.t_arrive) * 1000.0)

    def _hermes_turn(self, t, r):
        """One Hermes Agent turn: teach the codename on the first turn, ask for it on every later one."""
        i, rnd, now = r.agent, self.rnd, mono()
        cn = self.codename[i]
        fill = {"cn": cn, "Cn": cn[:1].upper() + cn[1:], "Sp": (cn[:1].upper() + cn[1:]).replace("-", " "),
                "joke": r.joke}
        self.m_calls += 1
        self.m_turns[i] += 1
        if not self.m_taught[i]:
            self.m_taught[i] = True
            self.m_taught_n += 1
            self.m_rests[i] = 0
            self.m_taught_at[i] = now
            self.m_awake_at_teach[i] = self._awake_s(i, now)
            mem = {"memory": "taught", "codename": cn}
            r.text = rnd.choice(TAUGHT_FMT).format(**fill)
        else:
            surv = self.m_rests[i]
            if rnd.random() < FORGET_RATE:
                self.m_fail[i] += 1
                self.m_fail_n += 1
                mem = {"memory": "forgot", "codename": cn, "survived": surv}
                r.text = rnd.choice(FORGOT_FMT).format(**fill)
            else:
                self.m_ok[i] += 1
                self.m_ok_n += 1
                self.m_survived[i] = surv
                if surv >= 1 and not self.m_recalled[i]:
                    self.m_recalled[i] = True
                    self.m_recalled_n += 1
                self.m_max_surv = max(self.m_max_surv, surv)
                mem = {"memory": "recalled", "codename": cn, "survived": surv}
                r.text = (rnd.choice(RECALL_FMT).format(**fill) + rnd.choice(CLOSERS)).strip()
        # the Hermes session grows every turn; the system prompt and history are prefix-cached
        r.prompt = HERMES_SYS_TOKENS + HERMES_TURN_TOKENS * self.m_turns[i] + rnd.randint(0, 40)
        r.completion = max(20, min(60, int(len(r.text) / 4.0) + rnd.randint(6, 14)))
        self.m_prompt[i] = r.prompt
        self.m_cached[i] = (r.prompt - rnd.randint(30, 80)) if self.m_turns[i] > 1 else rnd.randint(900, HERMES_SYS_TOKENS)
        self.m_last[i].append(r.text)
        return mem

    def _ev_reply(self, t, r):
        mem = self._hermes_turn(t, r) if HARNESS == "hermes" else None
        self.pod_inflight[r.pod] -= 1
        self.pod_total[r.pod] += 1
        tot = self.totals
        tot["requests"] += 1
        tot["replies"] += 1
        tot["prompt_tokens"] += r.prompt
        tot["completion_tokens"] += r.completion
        self.unique.add(" ".join(r.text.lower().split()))
        if r.steady:
            self.tr["inflight"] -= 1
            self.tr["replies"] += 1
        self.done_log.append((t, r.pod, r.prompt, r.completion, r.e2e, r.ttft, r.vq))
        self.seq += 1
        if r.mode == "steer8020":
            tag = PODS[r.pod]
        elif r.mode == "priority":
            tag = BANDS[r.band][1]
        else:
            tag = ""
        entry = {
            "seq": self.seq, "agent": agent_name(r.agent), "text": r.text,
            "latency_ms": int(round(r.e2e + r.queue_ms)), "t_ms": int(round(self._ms(t))), "tag": tag,
        }
        if mem:
            entry.update(mem)
        self.ticker.append(entry)
        if r.first and self.t0 is not None:
            bisect.insort(self.done_recs, (self._ms(t), r.prompt + r.completion))
        self._req_finished(t, r)

    def _pick_idle(self):
        rnd, a = self.rnd, self.agents
        for _ in range(64):
            i = rnd.randrange(TOTAL)
            if a[i] == C2:
                return i
        if self.cnt[C2] == 0:
            return None
        j = a.find(b"2", rnd.randrange(TOTAL))
        if j < 0:
            j = a.find(b"2")
        return j if j >= 0 else None

    def _steady(self, t):
        if getattr(self, "duty_cycle", False):
            i = self._pick_active_for_req()
            if i is None:
                self.tr["skipped"] += 1
                return
            r = self._new_req(i, True, t, suspend_after=False)
            self.tr["sent"] += 1
            self.tr["inflight"] += 1
            if self.pressure > 0.02:
                w = self._queue_wait_ms(r.band)
                self.band_queue[r.band] += 1
                self._sched(t + w / 1000.0, "dispatch", r)
            else:
                self._dispatch(t, r, self.rnd.uniform(0.2, 3.0))
            return
        i = self._pick_idle()
        if i is None:
            self.tr["skipped"] += 1
            return
        self._set(i, C3)
        r = self._new_req(i, True, t)
        self.tr["sent"] += 1
        self.tr["inflight"] += 1
        if self.pressure > 0.02:
            w = self._queue_wait_ms(r.band)
            self.band_queue[r.band] += 1
            self._sched(t + w / 1000.0, "dispatch", r)
        else:
            self._dispatch(t, r, self.rnd.uniform(0.2, 3.0))

    # ------------------------------------------------------------------- tick
    def tick(self):
        with self.lock:
            now = mono()
            dt = max(0.0, min(0.25, now - self.last_tick))
            self.last_tick = now
            if self.strategy == "steer8020":
                tgt = 0.8
            else:
                # Oct 8/10: configured TPU/GPU split (api/sim) = share of agents, a constant like the live 50/50
                tgt = max(0.0, min(1.0, self.sim["split_tpu_pct"] / 100.0))
            self.p1 += (tgt - self.p1) * (1.0 - math.exp(-dt / 0.35))
            want = 1.0 if (self.phase == "running" and self.rate >= SATURATE_AT_RPS) else 0.0   # spike in both flow modes (Oct 9)
            self.pressure += (want - self.pressure) * (1.0 - math.exp(-dt / 0.9))
            if self.phase == "running" and self.rate > 0 and dt > 0:
                if getattr(self, "duty_cycle", False):
                    # Maintain ~100 active agents (~90% idle rate) waking from C0 -> C1 -> C2 -> C3 -> C4 -> C0
                    up = self._up()
                    if up < 140:
                        target_active = 100.0
                        err = target_active - up
                        wake_rate = max(15.0, min(190.0, 108.0 + err * 5.0))
                        self.duty_acc = min(self.duty_acc + wake_rate * dt, 40.0)
                        nw = int(self.duty_acc)
                        if nw:
                            self.duty_acc -= nw
                            for k in range(nw):
                                idx = self._pick_suspended()
                                if idx is not None:
                                    wt = now - dt + dt * (k + 1) / nw
                                    self._set(idx, C1)
                                    self._sched(wt + self.rnd.uniform(0.06, 0.11), "duty_wake", idx)
                    else:
                        self.duty_acc = 0.0
                        nw = 0
                    # Any extra req/s above the duty wake rate (e.g. 300 req/s in priority mode)
                    extra_rate = max(0.0, float(self.rate) - 100.0)
                    self.arr_acc = min(self.arr_acc + extra_rate * dt, max(1.0, extra_rate))
                    n = int(self.arr_acc)
                    if n:
                        self.arr_acc -= n
                        for k in range(n):
                            self._steady(now - dt + dt * (k + 1) / n)
                else:
                    self.arr_acc = min(self.arr_acc + self.rate * dt, float(self.rate))  # <= 1 s of backlog
                    n = int(self.arr_acc)
                    if n:
                        self.arr_acc -= n
                        for k in range(n):
                            self._steady(now - dt + dt * (k + 1) / n)
            else:
                self.arr_acc = 0.0
                self.duty_acc = 0.0
            ev = self.events
            while ev and ev[0][0] <= now:
                t, _, kind, p = heapq.heappop(ev)
                getattr(self, "_ev_" + kind)(t, p)
            if (getattr(self, "duty_cycle", False) and self.phase == "running"
                    and self.next_duty_ramp is not None and now >= self.next_duty_ramp
                    and len(self.ramp_duty_pts) < 12):
                self.next_duty_ramp += 1.0
                base_n = len(self.ramp_wake_pts or [])
                step_t = (base_n + len(self.ramp_duty_pts) + 1) * 1000
                self.ramp_duty_pts.append({
                    "t_ms": step_t,
                    "running": self._up(),
                    "woke": self.woke,
                    "replies": self.totals["replies"],
                    "tokens": self.totals["prompt_tokens"] + self.totals["completion_tokens"],
                })
            if now >= self.next_metrics:
                self.next_metrics = now + METRICS_EVERY_S
                self._traj_step(now)
                self._metrics(now)
            if now >= self.next_hist:
                self.next_hist = max(self.next_hist + HIST_EVERY_S, now - HIST_EVERY_S)
                c = self.metrics.get("compare", {})
                # Oct 10: the timeline shows the dynamic path (ramp after wake, KV climb after the toggle), not the
                # settled comparison; ab = the A/B pool's current sample, cluster tok/s = the sum over both pools
                ab = self.ab_hist or {}
                f_cur = c.get("flow", {}).get(self.flow_mode) or {}
                active_tr = self.phase in ("running", "waking") and self.rate > 0 and bool(ab)
                self.history.append({
                    "t": unix_ms(),
                    "pods": [
                        {"req_s": p["req_s"], "gen_tok_s": p["gen_tok_s"], "e2e_ms": p["e2e_ms"]}
                        for p in self.metrics["pods"]
                    ],
                    "routing_mode": self.routing_mode,
                    "cluster_tok_s": ab.get("cluster_tok_s") if active_tr else None,
                    "cache_hit_pct": ab.get("cache_hit_pct") if active_tr else None,
                    "ttft_p50_ms": ab.get("ttft_p50_ms") if active_tr else None,
                    "ttft_p95_ms": ab.get("ttft_p95_ms") if active_tr else None,
                    "e2e_p50_ms": ab.get("e2e_p50_ms") if active_tr else None,
                    "e2e_p95_ms": ab.get("e2e_p95_ms") if active_tr else None,
                    "flow_mode": self.flow_mode,
                    "interactive_wait_ms": f_cur.get("interactive_ms") if active_tr else None,
                    "background_wait_ms": f_cur.get("background_ms") if active_tr else None,
                })

    def _metrics(self, now):
        lo = now - WINDOW_S
        dl, cl = self.dispatch_log, self.done_log
        while dl and dl[0][0] < lo:
            dl.popleft()
        while cl and cl[0][0] < lo:
            cl.popleft()
        W = WINDOW_S
        n, pt, ct = [0, 0], [0, 0], [0, 0]
        e2e, ttft, vq = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
        for (_, pod, p, c, e, f, q) in cl:
            n[pod] += 1
            pt[pod] += p
            ct[pod] += c
            e2e[pod] += e
            ttft[pod] += f
            vq[pod] += q
        dn, bn, bw = [0, 0], [0, 0, 0], [0.0, 0.0, 0.0]
        for (_, pod, band, qms) in dl:
            dn[pod] += 1
            bn[band] += 1
            bw[band] += qms
        tot_d = dn[0] + dn[1]
        rnd = self.rnd
        active_tr = self.phase in ("running", "waking") and (self.rate > 0 or tot_d > 0)
        # ---- Oct 10: calibrated pool dynamics (CALIB). Both pools are sampled from the model every tick; the substrate
        # above only contributes queue_ms / requests_total / the request split.
        self._dyn_step(now, active_tr and self.rate > 0)
        both_up = self.pod_up[0] and self.pod_up[1]
        share = [self.p1, 1.0 - self.p1] if both_up else [1.0 if self.pod_up[0] else 0.0, 1.0 if self.pod_up[1] else 0.0]
        n_agents = self._pool_agents(share)
        pv = {}
        for p, key in enumerate(POOL_KEYS):
            pv[key] = self._pool_sample(key, n_agents[p]) if (active_tr and share[p] > 0.0) else None
        self.pool_now = {k: ({"tok_s": v["tok_s"], "ttft_p50_ms": v["ttft_p50_ms"], "decode_ms": v["decode_ms"],
                              "cache_hit_pct": v["cache_hit_pct"]} if v else None) for k, v in pv.items()}
        pods = []
        for p, key in enumerate(POOL_KEYS):
            v = pv[key]
            k = n[p]
            on = v is not None and v["tok_s"] > 0.0
            running = int(round(n_agents[p] * self.dyn["load"])) if on else 0
            req_s = v["tok_s"] / float(CALIB["out_tok"]) if on else 0.0
            kv_pct = round(CALIB["kv_usage_pct"][key] * min(1.0, running / float(max(1, n_agents[p]))), 1) if on else 0
            pods.append({
                "name": PODS[p], "up": self.pod_up[p],
                "req_s": round(req_s, 1),
                "prompt_tok_s": int(round(req_s * (CALIB["prefix_tok"] + CALIB["turn_tok"] * (CALIB["turns"] - 1) / 2.0))) if on else 0,
                "gen_tok_s": int(round(v["tok_s"])) if on else 0,
                "e2e_ms": int(round(v["e2e_p50_ms"])) if on else None,
                "ttft_ms": int(round(v["ttft_p50_ms"])) if on else None,
                "queue_ms": round(vq[p] / k, 1) if k else None,
                "running": running,
                "waiting": 0,
                "cache_hit_pct": round(v["cache_hit_pct"], 1) if on else None,
                "kv_usage_pct": kv_pct,
                "requests_total": self.pod_total[p],
            })
        split = [round(100.0 * dn[0] / tot_d, 1), round(100.0 * dn[1] / tot_d, 1)] if tot_d else [50.0, 50.0]
        raw = min(0.78, (tot_d / W) / SAT_RPS)
        raw = raw * (1.0 - self.pressure) + (0.9 + rnd.gauss(0, 0.008)) * self.pressure
        self.sat += (raw - self.sat) * 0.35
        if self.sat < 0.001:
            self.sat = 0.0
        bands = [{"priority": prio, "name": name, "queue": max(0, self.band_queue[j]),
                  "wait_ms": int(round(bw[j] / bn[j])) if bn[j] else None, "req_s": round(bn[j] / W, 1)}
                 for j, (prio, name) in enumerate(BANDS)]

        ab_pool = self.sim["hardware"].get("ab_pool") or "gpu"
        ab_idx = POOL_KEYS.index(ab_pool) if ab_pool in POOL_KEYS else 1
        dy = self.dyn
        if active_tr:
            # compare.routing (Oct 10): the SETTLED calibrated numbers of the A/B pool for the mode that is running,
            # published like a measurement window (first after publish_warm_s, then every publish_every_s with a
            # window-to-window jitter of a few tenths of a percent); the other mode keeps its last window as the
            # 'vs round-robin' side. Running the router with no round-robin window yet seeds that side from CALIB.
            mode = self.routing_mode
            other = "without" if mode == "with" else "with"
            if mode == "with" and self.comp_routing["without"] is None:
                self.comp_routing["without"] = dict(settled(ab_pool, "without", n_agents[ab_idx]), live=False)
            if now - dy["mode_since"] >= CALIB["dyn"]["publish_warm_s"] and now >= dy["pub_t"]:
                dy["pub_t"] = now + CALIB["dyn"]["publish_every_s"]
                jit = {k: self._ar1("pub." + k, CALIB["noise"]["pub"])
                       for k in ("tok_s", "cache_hit_pct", "ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms", "e2e_p95_ms")}
                self.comp_routing[mode] = dict(settled(ab_pool, mode, n_agents[ab_idx], jit), live=True)
            elif self.comp_routing[mode]:
                self.comp_routing[mode]["live"] = True
            if self.comp_routing[other]:
                self.comp_routing[other]["live"] = False

            # compare.flow + bands (Oct 10): the measured lane waits (CALIB flow) with a few % of jitter; band req/s is
            # the cluster request rate split by tier mix, queue depth by Little's law (req/s x wait)
            fl = CALIB["flow"]
            cur_f = fl[self.flow_mode]
            fj = lambda k, v: int(round(max(1.0, v * (1.0 + dy["noise"].get("flow." + k, 0.0)))))
            pro_ms, std_ms, reg_ms = fj("int", cur_f["interactive_ms"]), fj("std", cur_f["standard_ms"]), fj("bg", cur_f["background_ms"])
            f_other = "without" if self.flow_mode == "with" else "with"
            if self.flow_mode == "with" and self.comp_flow["without"] is None:
                self.comp_flow["without"] = {"interactive_ms": int(round(fl["without"]["interactive_ms"])),
                                             "background_ms": int(round(fl["without"]["background_ms"])), "live": False}
            self.comp_flow[self.flow_mode] = {"interactive_ms": pro_ms, "background_ms": reg_ms, "live": True}
            if self.comp_flow[f_other]:
                self.comp_flow[f_other]["live"] = False
            cluster_req = sum(p["req_s"] for p in pods)
            bands = []
            for j, ((prio, name), mix, wait) in enumerate(zip(BANDS, CALIB["tier_mix"], (pro_ms, std_ms, reg_ms))):
                rs = cluster_req * mix * (1.0 + rnd.gauss(0, 0.03))
                bands.append({"priority": prio, "name": name,
                              "queue": max(0, int(round(rs * wait / 1000.0 + rnd.uniform(-0.5, 0.5)))) if cluster_req > 0 else 0,
                              "wait_ms": wait if cluster_req > 0 else None, "req_s": round(rs, 1)})
            self.sat = round(min(0.96, max(0.84, (0.90 if self.flow_mode == "with" else 0.89) + rnd.gauss(0, 0.01))), 3) if cluster_req > 0 else self.sat
        else:
            for k in ("without", "with"):
                if self.comp_routing[k]:
                    self.comp_routing[k]["live"] = False
                if self.comp_flow[k]:
                    self.comp_flow[k]["live"] = False

        self.sample_unix = unix_ms()
        # ---- Oct 8/10: per-pool numbers (pod 0 = TPU pool, pod 1 = GPU pool) from the calibrated dynamics: the share of
        # agents follows the configured split, every figure is this second's sample of the pool model (ramp after the
        # wake, KV climb after the toggle), the A/B pool's round-robin steady state is its baseline while the router
        # runs; the TPU pool is not part of the A/B and has no baseline.
        by_pool = None
        self.ab_hist = None
        if active_tr:
            by_pool = {}
            for p, key in enumerate(POOL_KEYS):
                v = pv[key]
                if v is None:
                    by_pool[key] = {"up": self.pod_up[p], "split_pct": 0.0, "req_s": 0.0, "tok_s": None, "ttft_p50_ms": None, "cache_hit_pct": None, "baseline": None,
                                    "pods": self._mock_pods(key, None, now, False)}
                    continue
                hit = round(v["cache_hit_pct"], 1)
                entry = {
                    "up": self.pod_up[p],
                    "split_pct": round(100.0 * share[p], 1),
                    "req_s": round(v["tok_s"] / float(CALIB["out_tok"]), 1),
                    "tok_s": int(round(v["tok_s"])),
                    "ttft_p50_ms": int(round(v["ttft_p50_ms"])),
                    "cache_hit_pct": hit,
                }
                # Oct 8/10: per-chip utilisation 0..1 for the Accelerators row (shades of green): the pool's load relative
                # to its calibrated steady state (chip_util_steady there) plus a slow per-chip random walk, so the chips
                # read as busy, not uniform and not blinking.
                n_chips = max(1, int(self.sim["hardware"].get(key + "_chips") or 1))
                drift = dy["chip"].get(key) or []
                rel = v["tok_s"] / v["tok_steady"] if v["tok_steady"] > 0 else 0.0
                entry["chip_util"] = [round(max(0.05, min(1.0, CALIB["chip_util_steady"] * rel + (drift[c] if c < len(drift) else 0.0))), 2)
                                      for c in range(n_chips)]
                # Oct 9/10: per-pod KV cards behind this pool (tpu_pods / gpu_pods simulated pods)
                entry["pods"] = self._mock_pods(key, hit, now, True, agents=n_agents[p] * dy["load"], h=dy["h"])
                is_ab = pool_calib(key, "without") != pool_calib(key, "with")
                if is_ab and self.routing_mode == "with":
                    base = settled(key, "without", n_agents[p])
                    entry["baseline"] = {"tok_s": base["tok_s"], "ttft_p50_ms": base["ttft_p50_ms"]}
                else:
                    entry["baseline"] = None
                cap = CALIB_KV["kv_cap_bytes"].get(key)   # Oct 13: KV capacity in HBM; the tiered pool reports its demo cap
                if cap is None and CALIB_KV["pool"] == key:
                    cap = CALIB_KV["hbm_cap_bytes"] * max(1, int(self.sim["hardware"].get(key + "_chips") or 1))
                if cap:
                    entry["kv_cap_bytes"] = int(cap)
                by_pool[key] = entry
            self.by_pool_base[self.routing_mode] = {k: dict(v) for k, v in by_pool.items()}
            ab = pv.get(ab_pool)
            if ab is not None:              # what history[] records: the A/B pool's path, cluster tok/s over both pools
                self.ab_hist = {"cluster_tok_s": int(round(sum(v["tok_s"] for v in pv.values() if v))),
                                "cache_hit_pct": round(ab["cache_hit_pct"], 1),
                                "ttft_p50_ms": int(round(ab["ttft_p50_ms"])), "ttft_p95_ms": int(round(ab["ttft_p95_ms"])),
                                "e2e_p50_ms": int(round(ab["e2e_p50_ms"])), "e2e_p95_ms": int(round(ab["e2e_p95_ms"]))}
        elif self.by_pool_base.get(self.routing_mode):
            by_pool = {k: dict(v, baseline=None) for k, v in self.by_pool_base[self.routing_mode].items()}
            for key in POOL_KEYS:           # not running: the cards keep the last hit %, nothing in flight
                if by_pool.get(key) is not None:
                    by_pool[key]["pods"] = self._mock_pods(key, by_pool[key].get("cache_hit_pct"), now, False)
        kv_by_host = {
            KV_HOSTS["tpu"]: (by_pool["tpu"]["cache_hit_pct"] if by_pool and by_pool["tpu"]["cache_hit_pct"] is not None else 0.0),
            KV_HOSTS["gpu"]: (by_pool["gpu"]["cache_hit_pct"] if by_pool and by_pool["gpu"]["cache_hit_pct"] is not None else
                              (self.comp_routing.get(self.routing_mode) or {}).get("cache_hit_pct", 0.0)),
        }
        # Oct 9: raw per-pod scrape shape of the live bench (compare.kv_by_pod), host -> [{ip, cache_hit_pct, running}]
        kv_by_pod = {}
        for key in POOL_KEYS:
            # (named kv_pods: `pods` is the per-pod metrics list that self.metrics["pods"] / history read below)
            kv_pods = (by_pool or {}).get(key, {}).get("pods") if by_pool else None
            if kv_pods is None:
                kv_pods = self._mock_pods(key, None, now, False)
            kv_by_pod[KV_HOSTS[key]] = [{"cache_hit_pct": q["cache_hit_pct"], "running": q["running"]} for q in kv_pods]
        # Oct 10: KV cache tiering on the GPU pool (screen 4). Capacity follows the sim's pool shape (see CALIB_KV).
        if hasattr(self, "kv_tier"):
            hw = self.sim["hardware"]
            self.kv_tier.set_shape(hw.get("gpu_chips") or 1, hw.get("gpu_pods") or 1)
            self.kv_tier.step(now, active_tr)
        self.metrics = {
            "window_s": int(W), "pods": pods, "split_pct": split,
            "flow": {"saturation": round(self.sat, 3), "bands": bands},
            "compare": {
                "routing_mode": self.routing_mode,
                "flow_mode": self.flow_mode,
                # per-Service KV hit (the live bench scrapes both pools)
                "kv_by_host": kv_by_host,
                # Oct 9: per-pod values behind each Service (the live bench's kv_by_pod); UI fallback when by_pool is absent
                "kv_by_pod": kv_by_pod,
                # Oct 8: simulation-only per-pool view (absent on the live contract until server.py fills it)
                "by_pool": by_pool,
                "routing": {
                    "without": self._routing_json("without"),
                    "with": self._routing_json("with"),
                    # mock: the router measured during the flow-control segment (closing summary col 3)
                    "with_fc": self._routing_json("with") if self.flow_mode == "with" else None,
                },
                "flow": {
                    "without": dict(self.comp_flow["without"]) if self.comp_flow["without"] else None,
                    "with": dict(self.comp_flow["with"]) if self.comp_flow["with"] else None,
                },
                # Oct 10: KV cache tiering (screen 4): enabled, pool, mode, capacity config, live path, settled legs
                "kv_tier": self.kv_tier.state() if hasattr(self, "kv_tier") else None,
            },
        }

    # --------------------------------------------------------------- snapshot
    def _ramp(self, now):
        """The driver's rampLocked(1000, rampEnd, 120): values at the END of each step, cumulative except running."""
        if getattr(self, "duty_cycle", False) and self.ramp_wake_pts is not None:
            return list(self.ramp_wake_pts) + list(self.ramp_duty_pts)
        if self.ramp_frozen is not None:
            return self.ramp_frozen
        if self.first_done_ms is not None:
            end = self.first_done_ms
        elif getattr(self, "wake_only", False) and not getattr(self, "traffic_started", True) and self.wake_done_ms is not None:
            end = self.wake_done_ms
        else:
            end = (now - self.t0) * 1000.0
        if end <= 0:
            return []
        npts = min(max(1, int(math.ceil(end / 1000.0))), RAMP_MAX_PTS)
        done_ms = [d[0] for d in self.done_recs]
        pref = [0]
        for d in self.done_recs:
            pref.append(pref[-1] + d[1])
        pts = []
        for k in range(npts):
            t = (k + 1) * 1000
            tt = min(t, end)
            woke = bisect.bisect_right(self.run_ts, tt)
            j = bisect.bisect_right(done_ms, tt)
            pts.append({"t_ms": t, "running": woke - bisect.bisect_right(self.susp_ts, tt), "woke": woke,
                        "replies": j, "tokens": pref[j]})
        if self.first_done_ms is not None:
            self.ramp_frozen = pts
        return pts

    def _agent_awake_pct(self, i, now):
        if self.m_taught[i]:
            w = now - self.m_taught_at[i]
            a = self._awake_s(i, now) - self.m_awake_at_teach[i]
        else:
            w = now - self.t_start
            a = self._awake_s(i, now)
        return round(100.0 * a / w, 1) if w > 0.5 else 0.0

    def _memory_json(self, now):
        acc, n, surv, sn = 0.0, 0, 0, 0
        for i in range(TOTAL):
            if self.m_taught[i] and now - self.m_taught_at[i] >= 5.0:
                acc += self._agent_awake_pct(i, now)
                n += 1
            if self.m_recalled[i]:
                surv += self.m_survived[i]
                sn += 1
        return {"enabled": True, "gen": self.gen, "taught": self.m_taught_n, "recall_ok": self.m_ok_n,
                "recall_fail": self.m_fail_n, "agents_recalled": self.m_recalled_n, "max_survived": self.m_max_surv,
                "mean_survived": round(surv / float(sn), 2) if sn else 0.0,
                "awake_pct": round(acc / n, 1) if n else round(100.0 * self._up() / TOTAL, 1),
                "proxy_calls": self.m_calls, "proxy_failures": self.m_call_fail}

    def agent_json(self, n):
        """GET api/agent?i=N (N = 1..TOTAL)."""
        with self.lock:
            i, now = n - 1, mono()
            out = {"agent": agent_name(i), "harness": HARNESS, "state": chr(self.agents[i]), "wakes": self.wakes[i],
                   "awake_pct": self._agent_awake_pct(i, now)}
            if HARNESS == "hermes":
                out.update({
                    "codename": self.codename[i], "session": "%s-g%d" % (agent_name(i), self.gen),
                    "taught": self.m_taught[i], "recall_ok": self.m_ok[i], "recall_fail": self.m_fail[i],
                    "rests_since_taught": self.m_rests[i], "survived": self.m_survived[i],
                    "last": list(self.m_last[i]),
                    "last_prompt_tokens": self.m_prompt[i] or None, "last_cached_tokens": self.m_cached[i] or None,
                })
            return out

    def snapshot(self):
        with self.lock:
            now = mono()
            b = {"id": "", "t0_unix_ms": 0, "elapsed_ms": 0, "running": self._up(), "idle_rate_pct": self._idle_pct(),
                 "peak_running": 0, "woke": 0, "wake_failed": 0, "all_running_ms": None, "suspend_elapsed_ms": None,
                 "all_suspended_ms": None, "milestones_ms": {}, "wake_ms": percentiles([]), "ramp": []}
            if self.t0 is not None:
                if self.all_running_ms is not None:
                    el = self.all_running_ms
                elif self.wake_done_ms is not None:
                    el = self.wake_done_ms
                else:
                    el = (now - self.t0) * 1000.0
                b.update({
                    "id": self.bid, "wake_only": getattr(self, "wake_only", False),
                    "traffic_started": getattr(self, "traffic_started", True),
                    "duty_cycle": getattr(self, "duty_cycle", False),
                    "idle_rate_pct": self._idle_pct(),
                    "t0_unix_ms": self.t0_unix, "elapsed_ms": int(round(el)),
                    "peak_running": self.peak, "woke": self.woke, "wake_failed": self.wake_failed,
                    "milestones_ms": {str(k): (None if self.milestones[k] is None else int(round(self.milestones[k])))
                                      for k in self.ms_keys},
                    "wake_ms": percentiles(self.wake_lat), "ramp": list(self._ramp(now)),
                })
                if self.all_running_ms is not None:
                    b["all_running_ms"] = int(round(self.all_running_ms))
                if self.susp_t0 is not None:
                    if self.phase == "suspending" and self.all_susp_ms is None:
                        b["suspend_elapsed_ms"] = int(round((now - self.susp_t0) * 1000))
                    elif self.all_susp_ms is not None:
                        b["suspend_elapsed_ms"] = int(round(self.all_susp_ms))
                    if self.all_susp_ms is not None:
                        b["all_suspended_ms"] = int(round(self.all_susp_ms))
            cfg = {"total_agents": TOTAL, "model": MODEL, "pods": list(PODS),
                   "accelerator": ACCEL, "max_tokens": MAX_TOKENS, "harness": HARNESS,
                   "build": (os.environ.get("DASHBOARD_BUILD") or (str(int(os.path.getmtime(INDEX_HTML))) if os.path.exists(INDEX_HTML) else "")).rsplit("/", 1)[-1].rsplit(":", 1)[-1],
                   "simulated": True,   # Oct 8: the dashboard shows SIMULATION instead of LIVE; the real driver never sets this
                   "sim": {"split_tpu_pct": self.sim["split_tpu_pct"]},   # Oct 8: current simulation settings (api/sim)
                   "hardware": dict(self.sim["hardware"]),
                   "live_available": False,
                   "live_reason": "Standalone simulation server — no live cluster connected"}
            if SEND_CLUSTER:
                cfg["cluster"] = dict(CLUSTERS[HARNESS])
            state = {
                "server_unix_ms": unix_ms(),
                "config": cfg,
                "phase": self.phase,
                "burst": b,
                "agents": self.agents.decode("ascii"),
                "totals": dict(self.totals, unique_replies=len(self.unique)),
                "ticker": list(self.ticker),
                # Oct 9: agent trajectory view, newest session first (same shape as the live server)
                "trajectories": self._traj_json(),
                "traffic": dict({"rate": self.rate, "strategy": self.strategy}, **self.tr),
                "llmd": dict(self.metrics, sample_unix_ms=self.sample_unix, history=list(self.history)),
            }
            if HARNESS == "hermes":
                state["memory"] = self._memory_json(now)
            if self.note:
                state["note"] = self.note
        return json.dumps(state, separators=(",", ":"), ensure_ascii=False)


SIM = Sim()


def sim_loop():
    while True:
        try:
            SIM.tick()
        except Exception as e:  # keep the mock alive no matter what
            print("[mock] tick error: %r" % (e,), file=sys.stderr, flush=True)
        time.sleep(TICK_S)


class Handler(BaseHTTPRequestHandler):
    server_version = "llmd-mock/1.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the 4 Hz polling quiet
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path.endswith("/api/state"):
            self._send(200, SIM.snapshot().encode("utf-8"), "application/json; charset=utf-8")
        elif path.endswith("/api/agent"):
            q = dict(kv.partition("=")[::2] for kv in query.split("&") if kv)
            try:
                n = int(q.get("i", ""))
            except ValueError:
                n = 0
            if not 1 <= n <= TOTAL:
                return self._json(400, {"ok": False, "error": "i must be 1..%d" % TOTAL})
            self._json(200, SIM.agent_json(n))
        elif path.endswith("/") or path.endswith("/index.html"):
            try:
                with open(INDEX_HTML, "rb") as f:
                    body = f.read()
            except OSError as e:
                return self._json(500, {"ok": False, "error": "cannot read index.html: %s" % e})
            self._send(200, body, "text/html; charset=utf-8")
        elif path.endswith("/favicon.ico"):
            self._send(204, b"", "image/x-icon")
        elif "/api/" not in path and "." not in path.rsplit("/", 1)[-1]:
            self._send(301, b"", "text/plain", {"Location": path + "/"})  # /prefix -> /prefix/
        else:
            self._json(404, {"ok": False, "error": "not found"})

    do_HEAD = do_GET

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            return self._json(400, {"ok": False, "error": "bad JSON"})
        if not isinstance(body, dict):
            return self._json(400, {"ok": False, "error": "bad JSON: expected an object"})
        i = path.rfind("/api/")
        name = path[i + 5:] if i >= 0 else ""
        fn = {"burst": SIM.start_burst, "simulate_traffic": SIM.simulate_traffic,
              "traffic": SIM.set_traffic, "strategy": SIM.set_strategy,
              "suspend": SIM.suspend, "reconcile": SIM.reconcile, "mock": SIM.mock_control,
              "sim": SIM.set_sim,
              "kv_tier": SIM.set_kv_tier,   # Oct 10: screen 4 toggle
              "memory/reset": SIM.memory_reset}.get(name)
        if fn is None:
            return self._json(404, {"ok": False, "error": "unknown endpoint: %s" % path})
        ok, res = fn(body)
        print("[mock] POST api/%s %s -> %s" % (name, json.dumps(body), "ok" if ok else "error: %s" % res),
              file=sys.stderr, flush=True)
        if not ok:
            return self._json(409, {"ok": False, "error": res})
        out = {"ok": True}
        if res is not None:
            out["result"] = res
        self._json(200, out)


def warm_to_hermes_live():
    """Fast-forwards SIM at startup so the UI immediately has active traffic if --warm is requested."""
    with SIM.lock:
        now = mono()
        SIM.burst_no = 1
        SIM.bid = "b-%s-0001" % RUN_TAG
        SIM._new_burst(SIM.bid, now - 14.0, TOTAL, True, wake_only=False)
        SIM.phase = "running"
        SIM.busy = False
        SIM.woke = TOTAL
        SIM.peak = TOTAL
        SIM.all_running_ms = 284.0
        SIM.wake_done_ms = 284.0
        SIM.first_done_ms = 920.0
        SIM.milestones = {100: 68.0, 250: 112.0, 500: 142.0, 750: 208.0, 1000: 284.0}
        SIM.wake_lat = [random.uniform(80.0, 284.0) for _ in range(TOTAL)]
        SIM.wake_lat[TOTAL // 2] = 142.0
        SIM.run_ts = sorted(SIM.wake_lat)
        SIM.ramp_wake_pts = [
            {"t_ms": 1000, "running": 1000, "woke": 1000, "replies": 1000, "tokens": 380000},
            {"t_ms": 2000, "running": 420, "woke": 1000, "replies": 1480, "tokens": 580000},
            {"t_ms": 3000, "running": 102, "woke": 1000, "replies": 1920, "tokens": 760000},
        ]
        SIM.ramp_duty_pts = [
            {"t_ms": 4000 + k * 1000, "running": 100 + (k % 3) * 4, "woke": 1000,
             "replies": 2200 + k * 210, "tokens": 880000 + k * 75000}
            for k in range(9)
        ]
        SIM.duty_cycle = True
        SIM.traffic_started = True
        SIM.duty_Settled = True
        SIM.traffic_t0 = now - 12.0
        SIM.next_duty_ramp = now + 1.0
        SIM.strategy = "priority"
        SIM.rate = 300
        SIM.pressure = 1.0
        SIM.sat = 0.94

        # Set 100 active agents (states C2/C3) and 900 suspended (C0) -> 90% Fleet Idle Rate
        for i in range(TOTAL):
            SIM.m_taught[i] = True
            SIM.m_taught_at[i] = now - 30.0
            SIM.m_awake_at_teach[i] = 0.0
            SIM.awake_acc[i] = 2.94
            SIM.wakes[i] = random.randint(3, 7)
            SIM.m_rests[i] = random.randint(2, 6)
            SIM.m_survived[i] = SIM.m_rests[i]
            SIM.m_recalled[i] = i < 984
            SIM.m_ok[i] = random.randint(2, 4)
            SIM.m_turns[i] = SIM.m_ok[i] + 1
            cn = SIM.codename[i]
            joke = SIM.rnd.choice(JOKES)
            txt = f"{cn} › {joke}"
            SIM.m_last[i].append(txt)
            SIM.m_prompt[i] = 1280
            SIM.m_cached[i] = 1190
            if i % 10 == 0:
                SIM._set(i, C3 if (i % 20 == 0) else C2)
                SIM.awake_since[i] = now - 0.2
            else:
                SIM._set(i, C0)

        SIM.m_taught_n = TOTAL
        SIM.m_ok_n = 2840
        SIM.m_fail_n = 0
        SIM.m_recalled_n = 984
        SIM.m_max_surv = 6
        SIM.m_calls = 3892
        SIM.totals = {
            "requests": 3892,
            "replies": 3892,
            "failed": 0,
            "prompt_tokens": 1342600,
            "completion_tokens": 140000,
        }
        SIM.pod_total = [2024, 1868]
        for idx in (881, 107, 420):
            cn = SIM.codename[idx]
            joke = SIM.rnd.choice(JOKES)
            SIM.seq += 1
            SIM.ticker.append({
                "seq": SIM.seq,
                "agent": agent_name(idx),
                "text": f"{cn} › {joke}",
                "latency_ms": random.randint(140, 175),
                "t_ms": SIM.seq * 320,
                "tag": "premium" if idx != 881 else "standard",
                "memory": "recalled",
                "codename": cn,
                "survived": SIM.m_survived[idx],
            })


def live_gateway_loop(gateway_url: str, model: str):
    """Sends real requests to llm-d-gateway when reachable and injects live Gemma 4 responses."""
    import urllib.request
    if not gateway_url:
        return
    url = gateway_url if gateway_url.endswith("/v1/chat/completions") else gateway_url.rstrip("/") + "/v1/chat/completions"
    idx = 0
    while True:
        try:
            with SIM.lock:
                active = SIM.phase == "running" and SIM.rate > 0
                i = (idx * 37 + 11) % TOTAL
                cn = SIM.codename[i]
                strat = SIM.strategy
            if not active:
                time.sleep(0.5)
                continue
            workload = "multi-turn" if (idx % 2 == 0) else "one-shot"
            objective = "agentic-multi-turn" if workload == "multi-turn" else "oneshot-fast"
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": f"You are {agent_name(i)} with codename {cn}. Reply with 1 short witty sentence starting with 'Codename {cn} here!' about AI agents or llm-d."},
                    {"role": "user", "content": "Give a 1-sentence joke about distributed systems, neural networks, or KV cache."},
                ],
                "max_tokens": 36,
                "temperature": 0.7,
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-llm-d-workload": workload,
                    "x-llm-d-inference-fairness-id": f"hermes-actor-{i:04d}",
                    "x-llm-d-inference-objective": objective,
                },
                method="POST",
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                lat_ms = int(round((time.perf_counter() - t0) * 1000.0))
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                txt = (msg.get("content") or msg.get("reasoning_content") or "").strip()
                if txt:
                    if cn.lower() not in txt.lower():
                        txt = f"Codename {cn} here! {txt}"
                    with SIM.lock:
                        SIM.seq += 1
                        SIM.m_last[i].append(txt)
                        SIM.ticker.append({
                            "seq": SIM.seq,
                            "agent": agent_name(i),
                            "text": txt,
                            "latency_ms": lat_ms,
                            "t_ms": int(round(SIM._ms(mono()))),
                            "tag": "premium" if strat == "priority" else PODS[idx % 2],
                            "memory": "recalled",
                            "codename": cn,
                            "survived": max(1, SIM.m_survived[i]),
                        })
            idx += 1
            time.sleep(0.8)
        except Exception:
            time.sleep(2.0)


def main():
    global HARNESS, SEND_CLUSTER
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8085"))
    gateway_url = os.environ.get("GATEWAY_URL", "http://llm-d-gateway.llm-d-serving.svc.cluster.local/v1/chat/completions")
    model = os.environ.get("MODEL_NAME", "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic")
    auto_warm = False
    args = sys.argv[1:]
    for j, a in enumerate(args):
        if a.isdigit():
            port = int(a)
        elif a.startswith("--port="):
            port = int(a.split("=", 1)[1])
        elif a == "--port" and j + 1 < len(args):
            port = int(args[j + 1])
        elif a.startswith("--host="):
            host = a.split("=", 1)[1]
        elif a == "--host" and j + 1 < len(args):
            host = args[j + 1]
        elif a.startswith("--gateway-url="):
            gateway_url = a.split("=", 1)[1]
        elif a == "--gateway-url" and j + 1 < len(args):
            gateway_url = args[j + 1]
        elif a.startswith("--model="):
            model = a.split("=", 1)[1]
        elif a == "--model" and j + 1 < len(args):
            model = args[j + 1]
        elif a == "--no-cluster":
            SEND_CLUSTER = False
        elif a == "--idle":
            auto_warm = False
        elif a == "--warm":
            auto_warm = True
        elif a.startswith("--harness="):
            HARNESS = a.split("=", 1)[1]
        elif a == "--harness" and j + 1 < len(args):
            HARNESS = args[j + 1]
    if HARNESS not in ("sandbox", "hermes"):
        sys.exit("--harness must be sandbox or hermes")
    if auto_warm:
        warm_to_hermes_live()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=sim_loop, daemon=True).start()
    threading.Thread(target=live_gateway_loop, args=(gateway_url, model), daemon=True).start()
    print("benchmark live driver (harness %s): http://%s:%d/  (gateway=%s)" % (HARNESS, host, port, gateway_url), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
