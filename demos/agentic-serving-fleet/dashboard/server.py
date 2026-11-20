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

"""Live fleet-driver backend for the "Agent Substrate x llm-d" observability dashboard.

Python 3 stdlib only. Serves ./index.html at "/" and simulates the JSON API:

  GET  api/state                                  full dashboard state (page polls at 4 Hz)
  POST api/burst     {"hold": true}               wake all agents at once
                     (+ optional "agents": N, "concurrency": C, like the real driver)
  POST api/traffic   {"rate": 100}                steady agent traffic in req/s (0 = off)
  POST api/strategy  {"mode": "balanced" | "steer8020" | "priority"}
  POST api/suspend   {}                           suspend all running agents (scale to zero)
  POST api/reconcile {}                           heal every agent back to suspended (phase "reconciling")

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
import socket
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "index.html")

TOTAL = 1000
PODS = ("Pod 1", "Pod 2")
MODEL = "google/gemma-4-26B-A4B-it"
ACCEL = "Pod 1 + Pod 2"
# Real hardware behind the pool (0.1 in the Oct 6 review): the deploy script derives these from the
# cluster's node labels (cloud.google.com/gke-tpu-accelerator / gke-accelerator) and the decode
# Deployments' ready replicas, so the UI never claims hardware that is not actually serving.
HARDWARE = {
    "tpu": os.environ.get("HW_TPU", "TPU"),
    "gpu": os.environ.get("HW_GPU", "GPU"),
    "tpu_pods": int(os.environ.get("HW_TPU_PODS", "1") or 0),
    "gpu_pods": int(os.environ.get("HW_GPU_PODS", "2") or 0),
    "engine": os.environ.get("HW_ENGINE", "vLLM"),
    # What vLLM / SGLang runs on in each pool (TPU backend on Pool A, CUDA on Pool B).
    "tpu_backend": os.environ.get("HW_TPU_BACKEND", "TPU"),
    "gpu_backend": os.environ.get("HW_GPU_BACKEND", "CUDA"),
    "model_label": os.environ.get("HW_MODEL_LABEL", "Gemma 4 26B-A4B · FP8"),
    # Oct 8: accelerator detail for the stack view's Accelerators row. The deploy script fills these from the
    # nodes' gke-tpu-accelerator / gke-tpu-topology / gke-accelerator labels and the pods' device limits.
    # Empty/0 = unknown; the UI then falls back to the pool name and 8 generic glyphs.
    "tpu_chip": os.environ.get("HW_TPU_CHIP", ""),
    "tpu_detail": os.environ.get("HW_TPU_DETAIL", ""),
    "tpu_topology": os.environ.get("HW_TPU_TOPOLOGY", ""),
    "tpu_chips": int(os.environ.get("HW_TPU_CHIPS", "0") or 0),
    "gpu_chip": os.environ.get("HW_GPU_CHIP", ""),
    "gpu_detail": os.environ.get("HW_GPU_DETAIL", ""),
    "gpu_chips": int(os.environ.get("HW_GPU_CHIPS", "0") or 0),
    # Which accelerator pool the A/B (routing pane) runs on: "gpu" | "tpu" (B2). Both sides target that pool.
    "ab_pool": (os.environ.get("AB_POOL", "") or ("tpu" if os.environ.get("AB_WORKLOAD", "one-shot") == "multi-turn" else "gpu")).lower(),
}
MAX_TOKENS = 50
WINDOW_S = 3.0
TICK_S = 0.01
METRICS_EVERY_S = 0.1
HIST_EVERY_S = 0.5
HIST_KEEP = 120
TICKER_KEEP = 60
TRAJ_KEEP = 12              # agent trajectories kept for the UI (newest first in api/state.trajectories)
TRAJ_TIERS = ("premium", "standard", "best-effort")
RAMP_MAX_PTS = 120          # the driver sends at most 120 ramp entries (1,000 ms steps)
MAX_SEQS = 128              # vLLM max_num_seqs per pod (mock): beyond this requests wait inside vLLM
SAT_RPS = 400.0             # request rate that would fully saturate the pool (mock)
SATURATE_AT_RPS = 300       # priority mode forms flow-control queues at/above this steady rate
BANDS = ((100, "premium"), (0, "standard"), (-10, "best-effort"))
BAND_WAIT_MS = (70.0, 400.0, 1100.0)
# Without flow control every tier sits in ONE shared FIFO under the spike: same queueing work, no lanes, so each
# tier waits about the load-weighted mean of the lanes above (20 % Pro / 60 % Plus / 20 % Free -> ~470 ms).
SHARED_WAIT_MS = 470.0
STANDARD = 1
MILESTONE_FRACS = (0.10, 0.25, 0.50, 0.75, 1.0)
SUSPEND_CAP = 200           # the driver caps suspend calls in flight at 200 ...
SUSPEND_S = (0.35, 0.95)    # ... and one suspend call takes this long (mock), so 1,000 agents take ~3 s
DRAIN_MAX_S = 15.0          # suspend waits (at most this long) for in-flight traffic before its clock starts
PREFLIGHT_S = (0.2, 0.4)    # one ListActors round trip (mock); every burst and reconcile starts with one
GATE_S = 0.3                # the driver parks all agent goroutines for 300 ms before it starts the clock
RECREATE_S = 1.2            # delete + create of one crashed agent (mock)
RECONCILE_MIN_S = 1.5       # a reconcile always shows up for at least this long (mock)

# Coding traces (7.1 in the Oct 6 review). JOKES keeps its name because the Sim/mock paths reference it;
# the content is now short coding one-liners an audience can read in ~1 s.
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
# Shared, byte-identical system prefix for every live agent request: the KV-cache-aware router can only
# win if there is a common prefix to hit, so keep this stable and put per-agent text after it.
# Shared by every agent and placed FIRST in the system prompt. It is deliberately long (~650 tokens): the
# TPU pod (vLLM on TPU) uses a 256-token KV block (`vllm:cache_config_info{block_size="256"}`), and a
# prefix only hits when a whole block is identical. With an ~80-token preamble followed by the per-agent
# codename nothing ever shared a full block, and the TPU pool measured 0 prefix-cache hits (2026-10-08).
_REPO_CONVENTIONS = (
    "Repository conventions (read before answering):\n"
    "1. Tensors: create on the accelerator device directly; move with .to(device, dtype=torch.bfloat16, non_blocking=True); "
    "never call .cuda() or .cpu() in library code; keep dtype explicit in every factory call.\n"
    "2. Autograd: zero grads with opt.zero_grad(set_to_none=True); use torch.no_grad() for evaluation, torch.inference_mode() "
    "for serving; detach before converting to Python numbers; avoid in-place ops on leaf tensors that require grad.\n"
    "3. Compilation: wrap the whole model with torch.compile(model, mode='max-autotune') once; keep shapes static where "
    "possible; mark dynamic dims with torch._dynamo.mark_dynamic; never compile inside a loop.\n"
    "4. Mixed precision: use torch.autocast(device_type, dtype=torch.bfloat16); GradScaler only for float16; keep the "
    "loss in float32; cast logits to float32 before softmax or cross-entropy when accuracy matters.\n"
    "5. Data: DataLoader with num_workers>0, pin_memory=True, persistent_workers=True, drop_last for training; use a "
    "sampler for distributed runs; seed with torch.manual_seed and a worker_init_fn.\n"
    "6. Distributed: initialise with torch.distributed.init_process_group(backend); prefer FSDP2 (fully_shard) for large "
    "models and DDP for small ones; all_reduce metrics before logging; destroy the process group on exit.\n"
    "7. Checkpoints: save state_dict, not the module; use safetensors where possible; include optimizer and scheduler "
    "state; write atomically (tmp file + os.replace); load with map_location and strict=True.\n"
    "8. Attention: use torch.nn.functional.scaled_dot_product_attention; pass is_causal=True instead of a mask when "
    "possible; keep q, k, v in (batch, heads, seq, dim) layout; use GQA by repeating k and v heads.\n"
    "9. Inference: batch requests; reuse the KV cache across turns; prefer paged attention through the serving engine; "
    "set max_new_tokens explicitly; stream tokens; measure TTFT and tokens per second separately.\n"
    "10. Style: one statement per line; no print debugging in committed code; type hints on public functions; "
    "torch 2.x idioms only; no deprecated torch.autograd.Variable; prefer torch.einsum for clarity over manual permutes.\n"
    "11. Testing: compare against a float32 reference with torch.testing.assert_close(rtol, atol); test one tiny shape "
    "on CPU and the real shape on the accelerator; seed every test; never assert exact equality of bfloat16 values.\n"
    "12. Performance: profile with torch.profiler before optimising; avoid host-device syncs (.item(), .cpu()) in hot "
    "loops; preallocate buffers; use channels_last for convolutions; fuse small ops with torch.compile.\n"
)
CODE_SYSTEM_PREFIX = (
    _REPO_CONVENTIONS +
    "\nYou are a Hermes coding agent working in this shared ML repository. Answer with exactly one line of "
    "Python, no prose, no markdown fences, no comments. Prefer modern tensor idioms, bfloat16 on accelerators, "
    "and set_to_none=True when zeroing grads. If a variable is not given, use conventional names "
    "(model, x, logits, opt, loader)."
)
CODE_TASKS = (
    "Move tensor x to the accelerator in bfloat16.",
    "Compute softmax probabilities of logits along the last dim.",
    "Build a DataLoader over ds with batch size 32 and shuffling.",
    "Save model's state_dict to ckpt.pt.",
    "Wrap model with torch.compile.",
    "Count trainable parameters of model.",
    "Seed torch and CUDA with 0.",
    "Flatten a 4D tensor x to 2D keeping the batch dim.",
    "Run model(x) with gradients disabled.",
    "Create an AdamW optimizer for model with lr 3e-4.",
    "One training step: backward, optimizer step, zero grads.",
    "Batched matmul of a and b with einsum.",
    "Causal attention mask of size T x T as a bool tensor.",
    "Causal scaled dot-product attention over q, k, v.",
    "Load ckpt.pt into model on CPU.",
    "Run model(x) under bfloat16 autocast on CUDA.",
    "Clip model's gradient norm to 1.0.",
    "Cosine LR schedule over steps for opt.",
    "Accuracy of pred vs target as a Python float.",
    "Allocate an FP8 KV cache of shape B,H,T,D.",
    "Greedy next-token ids from logits.",
    "Take the last-position logits and divide by temperature.",
    "Generate 48 new tokens greedily from ids with model.",
    "Mask x where mask is False with the dtype's minimum.",
    "All-reduce sum tensor t across ranks.",
)
OPENERS = ("",)              # code one-liners: no chatty openers/closers
CLOSERS = ("",)

# ---- Hermes Agent mode (--harness hermes)
HARNESS = "hermes"           # set from argv in main()
# config.cluster per harness; --no-cluster omits it (emulates an older backend without the field)
CLUSTERS = {"sandbox": {"nodes": 25, "node_type": "c4-standard-4", "vcpus_per_node": 4},
            "hermes": {"nodes": 25, "node_type": "e2-standard-4", "vcpus_per_node": 4}}
SEND_CLUSTER = True
AB = None                    # abbench.ABBench, created in main() when a gateway/baseline URL is configured
AGENTS_BY_POOL = {}          # {"tpu": n, "gpu": n} live agent workers per pool (set in main; drives by_pool split)
AGENT_THINK_S = float(os.environ.get("AGENT_THINK_S", "0.8") or 0.8)   # Oct 9: one turn per agent per this many s
# Oct 9 (dual-mode app): /sim/... is proxied to a mock_server.py sidecar so the same page can switch Live <-> Simulation.
SIM_PROXY = os.environ.get("SIM_PROXY", "http://127.0.0.1:8086").rstrip("/")
# Oct 9: build stamp -> config.build; the page reloads itself when it changes (no more stale tabs after a redeploy).
try:
    _BUILD_FALLBACK = str(int(os.path.getmtime(INDEX_HTML)))
except OSError:
    _BUILD_FALLBACK = str(int(time.time()))
BUILD = (os.environ.get("DASHBOARD_BUILD") or _BUILD_FALLBACK).rsplit("/", 1)[-1].rsplit(":", 1)[-1]   # tag only: no registry/project in the API
PACING = {"think_s": AGENT_THINK_S}   # runtime copy: POST api/agents {"think_s": 0} = closed loop (faster pool -> more turns)
CODE_ADJ = ("tensor", "gradient", "sparse", "fused", "quantized", "latent", "eager", "sharded", "cached", "vector",
            "stochastic", "compiled", "async", "scalar", "ternary", "dropout")
CODE_NOUN = ("otter", "falcon", "panda", "gecko", "badger", "lynx", "heron", "koala", "marmot", "orca", "puffin",
             "wombat", "tapir", "ibis", "newt", "yak")
FORGET_RATE = 0.01
TAUGHT_FMT = ("{cn} › {joke}", "{cn} › {joke}", "{cn} (noted) › {joke}", "{cn} › {joke}")
RECALL_FMT = ("{cn} › {joke}", "{cn} › {joke}", "{cn} › {joke}", "{cn} › {joke}",
              "{cn} › {joke}", "{cn} › {joke}")
FORGOT_FMT = ("(no codename) › {joke}", "(no codename) › {joke}")
HERMES_SYS_TOKENS = 1150     # Hermes system prompt + tool schemas (mock), cached after the first turn
HERMES_TURN_TOKENS = 64      # each turn adds to the session history (mock)
UPSET = (ord("1"), ord("2"), ord("3"), ord("4"))

C0, C1, C2, C3, C4, C9 = (ord(c) for c in "012349")


def mono():
    return time.monotonic()


_LIVE_STATUS = {"available": None, "reason": "", "checked_at": 0}

def get_live_status():
    now = time.monotonic()
    if _LIVE_STATUS["available"] is not None and now - _LIVE_STATUS["checked_at"] < 30.0:
        return _LIVE_STATUS["available"], _LIVE_STATUS["reason"]

    env_live = os.environ.get("LIVE_AVAILABLE")
    if env_live is not None:
        avail = env_live.lower() in ("1", "true", "yes")
        reason = os.environ.get("LIVE_REASON") or ("Live backend enabled via LIVE_AVAILABLE" if avail else "Live backend disabled via LIVE_AVAILABLE environment variable")
        _LIVE_STATUS.update({"available": avail, "reason": reason, "checked_at": now})
        return avail, reason

    gw = os.environ.get("GATEWAY_URL", "http://llm-d-gateway.llm-d-serving.svc.cluster.local/v1/chat/completions")
    try:
        host = urllib.parse.urlparse(gw).hostname
        if not host:
            avail, reason = False, "No GATEWAY_URL host configured"
        else:
            socket.getaddrinfo(host, None)
            avail, reason = True, f"Gateway reachable ({host})"
    except Exception:
        avail, reason = False, f"No VPC configured: GKE cluster backend '{host}' is unreachable"

    _LIVE_STATUS.update({"available": avail, "reason": reason, "checked_at": now})
    return avail, reason


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


# ---- Oct 10: KV cache tiering (screen 4, api/state compare.kv_tier, POST api/kv_tier).
# When KV_TIER_ENABLED=1 on the model pool, ABBench scrapes real vllm:external_prefix_cache_* + kv_offload_total_bytes_total
# counters. When KV_TIER_ENABLED=0 (default on the uncapped 4xGB200 pool so Screen 2's +76% routing benchmark is unaffected),
# Sim runs the same PR #3-calibrated KVTier model in-process (like Screen 3's flow scheduler) so Screen 4 is always live.
CALIB_KV = {
    "pool": HARDWARE["ab_pool"],
    "hbm_cap_bytes": 8 * 2 ** 30,
    "host_bytes": 64 * 2 ** 30,
    "storage_bytes": 1024 * 2 ** 30,
    "bytes_per_token": 46 * 1024,
    "prefix_tokens": 8100,
    "without": {"ttft_return_p50_ms": 1059.0, "ttft_return_p95_ms": 1804.0, "tok_s": 345.0, "req_s": 16.5,
                "hbm_hit_pct": 5.0, "host_hit_pct": 0.0, "storage_hit_pct": 0.0,
                "restore_gbps": 0.0, "offload_gbps": 0.0, "storage_read_gbps": 0.0, "storage_write_gbps": 0.0,
                "hbm_used_pct": 96.0, "host_used_pct": 0.0, "storage_used_pct": 0.0, "prefix_tok_s": 5245262.0 / 38.9},
    "with": {"ttft_return_p50_ms": 303.0, "ttft_return_p95_ms": 540.0, "tok_s": 931.0, "req_s": 44.6,
             "hbm_hit_pct": 5.0, "host_hit_pct": 74.0, "storage_hit_pct": 15.0,
             "restore_gbps": 15.1, "offload_gbps": 1.9, "storage_read_gbps": 4.8, "storage_write_gbps": 1.6,
             "hbm_used_pct": 97.0, "host_used_pct": 70.0, "storage_used_pct": 18.0, "prefix_tok_s": 5245184.0 / 14.3},
    "tau_up_s": 8.0,
    "tau_down_s": 4.0,
    "tau_offload_s": 2.5,
    "tau_host_fill_s": 10.0,
    "tau_host_drain_s": 40.0,
    "settled_frac": 0.9,
    "noise_tau_s": 1.6,
    "noise_pct": 0.035,
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
    """In-process HBM -> host-DRAM -> shared-storage KV tier model calibrated to PR #3/#5 (used when KV_TIER_ENABLED=0)."""

    def __init__(self, rnd):
        self.rnd = rnd
        self.mode = "without"
        self.enabled = os.environ.get("KV_TIER_ENABLED", os.environ.get("ENABLE_KV_TIERING", "0")) in ("1", "true", "yes")
        self.storage_enabled = os.environ.get("ENABLE_KV_STORAGE_TIER", "0") in ("1", "true", "yes")
        self.x = 0.0
        self.xo = 0.0
        self.xh = 0.0
        self.n1 = 0.0
        self.n2 = 0.0
        self.n3 = 0.0
        self.leg_t0 = None
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
        self.xh = self._relax(self.xh, 1.0 if on else (self.xh if self.enabled and active else 0.0), dt,
                              c["tau_host_fill_s"] if on else c["tau_host_drain_s"])
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
        self.live_samples = deque()        # Oct 9: (t_mono, ttft_ms, e2e_ms, completion_tokens) of the agents' live gateway replies
        # Oct 9: recent agent sessions for the trajectory view (filled by live_gateway_loop, cleared with the ticker)
        self.trajectories = deque(maxlen=TRAJ_KEEP)
        self.traj_seq = 0
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
        self.strategy = "balanced"          # one shared queue until the flow-control toggle turns priority lanes on (B3)
        self.routing_mode = "without"       # Pane 2: 'without' (round-robin) | 'with' (llm-d EPP)
        self.flow_mode = "without"          # Pane 3: 'without' | 'with' (flow control)
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
        self.kv_tier = KVTier(self.rnd)
        self._mem_reset(1)
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
        if getattr(self, "trajectories", None) is not None:
            self.trajectories.clear()       # old codenames would be stale in the trajectory view

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
            self.rate = 300 if self.strategy == "priority" else 100
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

    def set_strategy(self, body):
        # Pane 2/3 toggles: {"routing": "without"|"with"} and {"flow": "without"|"with"}
        routing = body.get("routing")
        flow = body.get("flow")
        if routing is not None or flow is not None:
            if routing is not None and routing not in ("without", "with"):
                return False, "routing must be without or with"
            if flow is not None and flow not in ("without", "with"):
                return False, "flow must be without or with"
            if routing is not None and AB is not None:
                AB.set_kv_ring(False)
                if getattr(AB, "kv_tier_enabled", False):
                    AB.set_kv_mode("without")
            with self.lock:
                if routing is not None:
                    self.routing_mode = routing
                if flow is not None:
                    # B3 (Oct 7): the flow toggle must drive the scheduler, not just a label. Priority lanes
                    # only form under strategy == "priority"; "without flow control" = one shared queue.
                    self.flow_mode = flow
                    self.strategy = "priority" if flow == "with" else "balanced"
                    if self.phase == "running" and self.rate > 0:
                        if self.rate < SATURATE_AT_RPS:
                            # Oct 9: whatever was measured before the spike (1-2 ms waits at the idle rate) is not the
                            # baseline for this screen; both columns are re-measured under the spike.
                            self.flow_measured = {"without": None, "with": None}
                        self.rate = SATURATE_AT_RPS    # saturate the pool so the queue story is visible either way
                return True, {"routing": self.routing_mode, "flow": self.flow_mode}
        mode = body.get("mode")
        if mode not in ("balanced", "steer8020", "priority"):
            return False, "mode must be balanced, steer8020 or priority"
        with self.lock:
            self.strategy = mode
            if self.phase == "running" and self.rate > 0:
                self.rate = 300 if mode == "priority" else 100
            return True, mode

    def set_kv_tier(self, body):
        """POST api/kv_tier {"mode": "without"|"with"|"storage"} (Oct 10, screen 4; Oct 14: Tier 3)."""
        mode = body.get("mode")
        if AB is not None and getattr(AB, "kv_tier_enabled", False):
            with self.lock:
                self.routing_mode = "with"
            if not mode:
                return True, {"mode": AB.kv_tier_mode, "enabled": True}
            AB.set_kv_ring(mode in ("with", "storage"))
            return AB.set_kv_mode(mode)
        with self.lock:
            if not mode:
                return True, {"mode": self.kv_tier.mode, "enabled": True}
            ok, res = self.kv_tier.set_mode(mode)
            if not ok:
                return False, res
            return True, {"mode": res, "enabled": True}

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
        self._reset_counters()
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
                    self.rate = 300 if self.strategy == "priority" else 100

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
                        self.rate = 300 if self.strategy == "priority" else 100
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
        self._reset_counters(fresh_modes=True)

    def _reset_counters(self, fresh_modes=False):
        """Oct 8: a suspended fleet shows a blank dashboard. Every counter the stack / panes display is cleared
        here (and again at the next wake) so a suspend -> resume cycle never carries stale numbers: request and
        token totals, the reply ticker, per-pod request counts, the throughput history, the queue-wait EMAs of the
        flow pane and the live A/B measurements. The burst record (wake / suspend timings) is kept for the agents
        screen. fresh_modes also puts both toggles back to "without" so the next take starts at the beginning."""
        self.totals = {"requests": 0, "replies": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.unique = set()
        self.ticker.clear()
        self.trajectories.clear()
        self.pod_total = [0, 0]
        self.tr = {"inflight": self.tr.get("inflight", 0), "sent": 0, "replies": 0, "failed": 0, "skipped": 0}
        self.dispatch_log.clear()
        self.done_log.clear()
        self.history.clear()
        self.flow_measured = {"without": None, "with": None}
        self.pressure = 0.0
        self.sat = 0.0
        self.note = ""
        if fresh_modes:
            self.routing_mode = "without"
            self.flow_mode = "without"
            self.strategy = "balanced"
        if hasattr(self, "kv_tier"):
            if fresh_modes:
                self.kv_tier.reset()
            else:
                self.kv_tier.legs = {"without": None, "with": None}
                self.kv_tier.tok_restored = 0.0
                self.kv_tier.gb_restored = 0.0
                self.kv_tier.gb_offloaded = 0.0
        self.live_samples.clear()
        if AB is not None and AB.enabled():
            AB.reset()

    def _ev_recon_done(self, t, p):
        rest, dead = p
        for i in range(TOTAL):
            if self.agents[i] != C0:
                self._set(i, C0)
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
        both = self.pod_up[0] and self.pod_up[1]
        share = (self.p1 if pod == 0 else 1.0 - self.p1) if both else 1.0
        load = self.rate * share if self.phase == "running" else 0.0
        rnd = self.rnd
        ttft = max(18.0, 40.0 + 0.45 * load + rnd.gauss(0, 5))
        tpot = max(4.0, 6.3 + 0.018 * load + rnd.gauss(0, 0.25))
        return ttft, max(80.0, ttft + completion * tpot)

    def _queue_wait_ms(self, band):
        """Queue wait for a request under spike pressure: its priority lane with flow control, the shared FIFO without."""
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
                tgt = 0.5 + 0.02 * math.sin(now / 3.1) + 0.012 * math.sin(now / 1.27)
            self.p1 += (tgt - self.p1) * (1.0 - math.exp(-dt / 0.35))
            # Oct 9: the spike builds queue pressure in BOTH flow modes (it used to form only under 'priority', so
            # 'without flow control' showed Pro waiting 1 ms and the pane read as flow control making Pro worse).
            want = 1.0 if (self.phase == "running" and self.rate >= SATURATE_AT_RPS) else 0.0
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
                self._metrics(now)
            if now >= self.next_hist:
                self.next_hist = max(self.next_hist + HIST_EVERY_S, now - HIST_EVERY_S)
                entry = {"t": unix_ms(), "pods": [
                    {"req_s": p["req_s"], "gen_tok_s": p["gen_tok_s"], "e2e_ms": p["e2e_ms"]}
                    for p in self.metrics["pods"]]}
                if AB is not None and AB.enabled():
                    entry.update(AB.point())
                    entry["flow_mode"] = self.flow_mode
                self.history.append(entry)

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
        up_frac = self._up() / float(TOTAL)
        fleet_active_scale = 1.0 if (getattr(self, "duty_cycle", False) and self.rate > 0) else up_frac
        rnd = self.rnd
        pods = []
        for p in range(2):
            inflight = self.pod_inflight[p]
            running = min(inflight, MAX_SEQS)
            k = n[p]
            pod_share = (dn[p] / float(tot_d)) if tot_d else 0.5
            kv_pct = (
                round(max(1.0, min(97.0, 18.0 * fleet_active_scale + 36.0 * fleet_active_scale * pod_share + running * 0.55 + rnd.gauss(0, 0.4))), 1)
                if (inflight or (k and fleet_active_scale > 0))
                else 0
            )
            hit_pct = round(min(99.0, 88.5 + 3.5 * pod_share + rnd.gauss(0, 0.6)), 1) if k else None
            if AB is not None and AB.enabled():
                pkey = "tpu" if p == 0 else "gpu"
                real_hit = None
                real_u = None
                for h_name, h_val in getattr(AB, "kv_by_host", {}).items():
                    if pkey in h_name.lower() and h_val is not None:
                        real_hit = h_val
                for h_name, t_val in getattr(AB, "kv_tier_by_host", {}).items():
                    if pkey in h_name.lower() and isinstance(t_val, dict) and t_val.get("hbm_used_pct") is not None:
                        real_u = t_val.get("hbm_used_pct")
                hit_pct = real_hit if (real_hit is not None and k) else None
                if real_u is not None:
                    kv_pct = real_u
            pods.append({
                "name": PODS[p], "up": self.pod_up[p],
                "req_s": round(k / W, 1),
                "prompt_tok_s": int(round(pt[p] / W)),
                "gen_tok_s": int(round(ct[p] / W)),
                "e2e_ms": int(round(e2e[p] / k)) if k else None,
                "ttft_ms": int(round(ttft[p] / k)) if k else None,
                "queue_ms": round(vq[p] / k, 1) if k else None,
                "running": running,
                "waiting": max(0, inflight - MAX_SEQS),
                "cache_hit_pct": hit_pct,
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
        self.sample_unix = unix_ms()
        if hasattr(self, "kv_tier"):
            pool_k = HARDWARE.get("ab_pool") or "gpu"
            self.kv_tier.set_shape(HARDWARE.get(pool_k + "_chips") or 4, HARDWARE.get(pool_k + "_pods") or 4)
            self.kv_tier.step(now, self.phase in ("running", "waking") and self.rate > 0)
        self.metrics = {"window_s": int(W), "pods": pods, "split_pct": split,
                        "flow": {"saturation": round(self.sat, 3), "bands": bands}}
        # Pane 3 comparison (open issue 4, Oct 6): the queue waits per tier under the current flow mode, as an
        # EMA over the ticks spent in that mode. These come from the same scheduler model as the bands table
        # above (the agents' queue is in this process; the engines behind it are the real pools).
        if self.phase == "running" and self.rate > 0:
            byname = {b["name"]: b["wait_ms"] for b in bands}
            cur = {"interactive_ms": byname.get("premium"), "standard_ms": byname.get("standard"),
                   "background_ms": byname.get("best-effort")}
            if all(v is not None for v in cur.values()):
                fm = getattr(self, "flow_measured", None)
                if fm is None:
                    fm = self.flow_measured = {"without": None, "with": None}
                prev = fm.get(self.flow_mode)
                if prev is None:
                    fm[self.flow_mode] = dict(cur, n=1)
                else:
                    a = 0.25
                    fm[self.flow_mode] = {k: int(round(prev[k] + a * (cur[k] - prev[k]))) for k in cur}
                    fm[self.flow_mode]["n"] = prev.get("n", 0) + 1

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
            live_avail, live_reason = get_live_status()
            cfg = {"total_agents": TOTAL, "model": MODEL, "pods": list(PODS),
                   "accelerator": ACCEL, "max_tokens": MAX_TOKENS, "harness": HARNESS,
                   "hardware": {k: ("" if k in ("tpu_chip", "tpu_detail", "gpu_chip", "gpu_detail") else v) for k, v in HARDWARE.items()},   # no chip SKUs / arch names leave the server (Oct 9)
                   "agent_think_s": PACING["think_s"], "sim_proxy": bool(SIM_PROXY), "build": BUILD,
                   "live_available": live_avail, "live_reason": live_reason}
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
                # Oct 9: agent trajectory view, newest session first; copied because the worker threads keep
                # appending turns to the live entries after we leave the lock
                "trajectories": [dict(tr, turns=list(tr["turns"])) for tr in reversed(self.trajectories)],
                "traffic": dict({"rate": self.rate, "strategy": self.strategy}, **self.tr),
                "llmd": dict(self.metrics, sample_unix_ms=self.sample_unix, history=list(self.history),
                             compare=(AB.compare(self.flow_mode) if (AB is not None and AB.enabled()) else
                                      {"routing_mode": self.routing_mode, "flow_mode": self.flow_mode,
                                       "source": "none", "routing": {"without": None, "with": None},
                                       "flow": {"without": None, "with": None}})),
            }
            fm = getattr(self, "flow_measured", None) or {}
            cmpd = state["llmd"].get("compare") or {}
            # Oct 9: per-pool numbers on the live backend (the mock emits the same shape): the A/B pool from the bench,
            # the other pool from the agents' gateway traffic, KV + in-flight load per pool from the pod scrape.
            if AB is not None and AB.enabled():
                cmpd.setdefault("kv_by_pod", {})      # raw per-pod scrape (AB.compare fills it); the no-by_pool fallback
                try:
                    cmpd["by_pool"] = AB.by_pool(
                        HARDWARE["ab_pool"], list(self.live_samples),
                        {"tpu": HARDWARE.get("tpu_chips") or 0, "gpu": HARDWARE.get("gpu_chips") or 0},
                        active=(self.phase == "running" and self.rate > 0), agents=AGENTS_BY_POOL)
                except Exception as e:
                    cmpd["by_pool"] = None
                    self.note = "by_pool error: %r" % (e,)
            if AB is not None and AB.enabled():
                pool_k = HARDWARE["ab_pool"]
                cmpd["kv_tier"] = AB.kv_tier_state(
                    pool=pool_k, active=(self.phase == "running" and self.rate > 0),
                    accelerators=HARDWARE.get(pool_k + "_chips") or 4,
                    replicas=HARDWARE.get(pool_k + "_pods") or 1)
            else:
                cmpd["kv_tier"] = self.kv_tier.state()
            cmpd["flow"] = {m: (dict(fm[m], live=(m == self.flow_mode and self.phase == "running")) if fm.get(m) else None)
                            for m in ("without", "with")}
            state["llmd"]["compare"] = cmpd
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

    def _proxy_sim(self, raw=None):
        """Forward /sim/<rest> to the simulation sidecar (mock_server.py) and relay its reply verbatim."""
        i = self.path.find("/sim/")
        url = SIM_PROXY + self.path[i + 4:]
        req = urllib.request.Request(url, data=raw, method=self.command, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as r:
                body = r.read()
                if self.path.split("?", 1)[0].endswith("/api/state"):
                    try:
                        js = json.loads(body.decode("utf-8"))
                        if isinstance(js, dict) and isinstance(js.get("config"), dict):
                            live_avail, live_reason = get_live_status()
                            js["config"]["live_available"] = live_avail
                            js["config"]["live_reason"] = live_reason
                            js["config"]["sim_proxy"] = True
                            body = json.dumps(js).encode("utf-8")
                    except Exception:
                        pass
                self._send(r.status, body, r.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read(), e.headers.get("Content-Type", "application/json"))
        except Exception as e:
            self._json(502, {"ok": False, "error": "simulation backend unavailable: %s" % e})

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if "/sim/" in path and SIM_PROXY:
            return self._proxy_sim()
        if path.endswith("/api/raw"):          # Oct 9: raw vLLM /metrics of one pod, for the "behind the scenes" panel
            q = dict(kv.partition("=")[::2] for kv in query.split("&") if kv)
            pool = (q.get("pool") or "gpu").lower()
            try:
                i = max(1, int(q.get("i", "1")))
            except ValueError:
                i = 1
            pods = []
            if AB is not None:
                for h, v in (AB.kv_by_pod or {}).items():
                    if pool in h.lower():
                        pods = v or []
                        break
                if not pods and pool == HARDWARE["ab_pool"]:
                    for h in (AB.kv_hosts or ()):
                        if h in (AB.kv_by_pod or {}):
                            pods = AB.kv_by_pod[h] or []
                            break
            if i > len(pods) or not pods[i - 1].get("ip"):
                return self._json(404, {"ok": False, "error": "no such pod (pool=%s, i=%d, known=%d)" % (pool, i, len(pods))})
            ip = pods[i - 1]["ip"]
            p_port = pods[i - 1].get("port") or "8000"
            try:
                with urllib.request.urlopen("http://%s:%s/metrics" % (ip, p_port), timeout=2.0) as r:
                    txt = r.read().decode("utf-8", errors="replace")
            except Exception as e:
                return self._json(502, {"ok": False, "error": "scrape failed: %s" % e})
            # Traffic counters only (no cache_config_info / process start / addresses): nothing that describes the hardware or the project.
            keep = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
                    "vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_queries_total",
                    "vllm:kv_offload_total_bytes_total", "vllm:kv_offload_tiering_", "vllm:num_requests_running",
                    "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "vllm:generation_tokens_total",
                    "vllm:prompt_tokens_total", "vllm:request_success_total")
            lines = [l for l in txt.splitlines() if l.startswith(keep)]
            label = "Accelerator A" if pool == "tpu" else "Accelerator B"
            body = ("# %s pool · pod-%d · %s\n" % (label, i, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())) + "\n".join(lines) + "\n").encode("utf-8")
            return self._send(200, body, "text/plain; charset=utf-8")
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
        if "/sim/" in path and SIM_PROXY:
            return self._proxy_sim(raw)
        if path.endswith("/api/agents"):      # Oct 9: agent pacing. think_s 0 = closed loop, 0.8 = equal rate per pool
            try:
                v = float(body.get("think_s", PACING["think_s"]))
            except (TypeError, ValueError):
                return self._json(400, {"ok": False, "error": "think_s must be a number"})
            PACING["think_s"] = max(0.0, min(5.0, v))
            print("[agents] think_s -> %.2f" % PACING["think_s"], file=sys.stderr, flush=True)
            return self._json(200, {"ok": True, "result": {"think_s": PACING["think_s"]}})
        i = path.rfind("/api/")
        name = path[i + 5:] if i >= 0 else ""
        fn = {"burst": SIM.start_burst, "simulate_traffic": SIM.simulate_traffic,
              "traffic": SIM.set_traffic, "strategy": SIM.set_strategy,
              "suspend": SIM.suspend, "reconcile": SIM.reconcile, "mock": SIM.mock_control,
              "kv_tier": SIM.set_kv_tier,
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


def storage_ring_loop(model: str, worker: int = 0, nworkers: int = 1):
    """Oct 14: the Tier 3 ring. KV_STORAGE_WORKERS copies run a fixed ring of KV_STORAGE_RING sessions against
    the scale-out replicas (llmd-gpu-storage) while Tier 3 is selected. Every turn goes to the next replica,
    so the prefix a returning session needs was computed by another replica and can only come from the shared
    storage tier. Three turns in four allow the restore (leg "restore"), the fourth forces a recompute with the
    per-request opt-out (leg "recompute"), so both numbers are always live. The session set never changes, so
    the chunks on storage stay bounded. Nothing here touches the A/B pool or the routing numbers.
    """
    from abbench import stream_chat, _Session, COHORTS, MAX_TOKENS
    ring_n = max(1, int(os.environ.get("KV_STORAGE_RING", "5") or 5))
    think_s = max(0.0, float(os.environ.get("KV_STORAGE_THINK_S", "0.5") or 0.5))
    rnd = random.Random(hash((time.time_ns(), worker, os.getpid(), "storage")))
    ring = [None] * ring_n
    pos = 0
    while True:
        try:
            ab = AB if (AB is not None and AB.enabled()) else None
            with SIM.lock:
                active = SIM.phase == "running" and SIM.rate > 0
            if ab is None or not active or ab.kv_tier_mode != "storage":
                time.sleep(0.5)
                continue
            slot = pos
            pos = (pos + 1) % ring_n
            i = (worker * ring_n + slot + 7) % TOTAL
            if ring[slot] is None:
                s = _Session(agent_name(i), i % COHORTS, rnd)
                s.msgs[0]["content"] += f"\n{CODE_SYSTEM_PREFIX}\nYour codename is {SIM.codename[i]}."
                ring[slot] = s
            sess = ring[slot]
            sess.msgs = [sess.msgs[0]]
            if sess.turn == 0:
                sess.turn = 1
                user = CODE_TASKS[(worker * 7 + i) % len(CODE_TASKS)]
            else:
                user = sess.next_user(rnd)          # increments sess.turn
            sess.msgs.append({"role": "user", "content": f"Turn {sess.turn}: {user}"})
            url = ab.storage_pod_url(worker * ring_n + slot + sess.turn)
            if not url:
                time.sleep(1.0)
                continue
            leg = "recompute" if sess.turn > 1 and sess.turn % 4 == 0 else "restore"
            req_body = {
                "model": model,
                "messages": sess.msgs,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.2,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if leg == "recompute":
                req_body["kv_transfer_params"] = {"max_load_tokens": 0, "max_offload_tokens": 0}
            res = stream_chat(url, json.dumps(req_body).encode("utf-8"), {"Content-Type": "application/json"}, timeout=30.0)
            if res is None:
                ab.add_storage_error()
                time.sleep(1.0)
                continue
            ttft_ms, e2e_ms, out_tok, prompt_tok, txt = res
            sess.msgs.append({"role": "assistant", "content": (txt or "ok")[:400]})
            ab.add_storage_sample(leg, ttft_ms, e2e_ms, out_tok, prompt_tok, turn=sess.turn, url=url)
            if think_s:
                time.sleep(rnd.uniform(0.5 * think_s, 1.5 * think_s))
        except Exception as exc:
            print("[kv-storage-%d] %s" % (worker, exc), file=sys.stderr, flush=True)
            time.sleep(1.0)


def live_gateway_loop(gateway_url: str, model: str, worker: int = 0, nworkers: int = 1):
    """Real Hermes agent traffic against BOTH pools (Oct 9) and the live ticker.

    LIVE_GATEWAY_WORKERS copies run in parallel. Even workers drive the agents' pool (TPU when AB_POOL=gpu)
    through the llm-d gateway; odd workers drive the A/B pool and follow the Routing toggle exactly like the
    old synthetic bench did: 'without' = the pool's plain k8s Service (kube-proxy round-robin, no EPP),
    'with' = the gateway (KV-cache-aware EPP). Every finished turn on the A/B pool is recorded with
    AB.add_sample(), every turn on the other pool in SIM.live_samples -> compare.by_pool shows the same
    workload on both cards and the "Without vs With llm-d" numbers are the agents' own.

    Each agent runs a short session (abbench.TURNS turns) that starts with its own ~1.6k-token workspace
    prefix (unique per session, so a prefix is only ever cached on the pod(s) that served that session),
    turn 1 is a coding task from CODE_TASKS, later turns short follow-ups; then the worker moves on to the
    next agent. Without an A/B bench (local runs) every worker goes through the gateway as before.
    """
    if not gateway_url:
        return
    from abbench import stream_chat, _Session, COHORTS, TURNS, MAX_TOKENS
    gw = gateway_url if gateway_url.endswith("/v1/chat/completions") else gateway_url.rstrip("/") + "/v1/chat/completions"
    # worker w takes idx = w, w+n, w+2n, ... so codenames don't collide across workers.
    idx = worker
    # seeded from the clock: a deterministic seed replayed the same workspace prefixes after every redeploy, and the
    # pods still had them cached -> round-robin opened at ~98 % KV hit (seen 02:35Z). Prefixes must be new per process.
    rnd = random.Random(hash((time.time_ns(), worker, os.getpid())))
    KV_RING_SIZE = 5
    ring = [None] * KV_RING_SIZE
    ring_traj = [None] * KV_RING_SIZE
    ring_ttft1 = [None] * KV_RING_SIZE
    ring_pos = 0
    sess = None
    sess_i = None
    sess_mode = None
    traj = None                  # trajectory entry of the current session (SIM.trajectories)
    traj_ttft1 = None            # turn-1 TTFT of the current session (cache_hit heuristic)
    while True:
        try:
            t_turn = mono()
            with SIM.lock:
                active = SIM.phase == "running" and SIM.rate > 0
                i = (idx * 37 + 11) % TOTAL
                cn = SIM.codename[i]
                strat = SIM.strategy
                routing = SIM.routing_mode
                flow = SIM.flow_mode
            if not active:
                time.sleep(0.5)
                continue
            ab = AB if (AB is not None and AB.enabled()) else None
            on_ab_pool = ab is not None and nworkers > 1 and (worker % 2 == 1)
            ring_mode = on_ab_pool and ab is not None and getattr(ab, "kv_tier_enabled", False) and getattr(ab, "kv_ring_active", False)
            if ring_mode:
                gpu_w = worker // 2
                mode = "with"
                url = ab.pod_chat_url(gpu_w)
                headers = ab.headers(mode, agent_name(i))
                pool = HARDWARE["ab_pool"]
                slot = ring_pos
                ring_pos = (ring_pos + 1) % KV_RING_SIZE
                i = (gpu_w * KV_RING_SIZE + slot + 1) % TOTAL
                cn = SIM.codename[i]
                if ring[slot] is None or (slot == 0 and ring[slot].turn >= 8):
                    s = _Session(agent_name(i), i % COHORTS, rnd)
                    s.msgs[0]["content"] += f"\n{CODE_SYSTEM_PREFIX}\nYour codename is {cn}."
                    ring[slot] = s
                    with SIM.lock:
                        SIM.traj_seq += 1
                        ring_traj[slot] = {
                            "id": "%s-g%d-%d" % (agent_name(i), SIM.gen, SIM.traj_seq),
                            "agent": agent_name(i), "codename": cn,
                            "pool": pool, "mode": mode,
                            "tier": TRAJ_TIERS[i % 3],
                            "started_ms": int(round(SIM._ms(mono()))),
                            "done": False,
                            "turns": [],
                        }
                        SIM.trajectories.append(ring_traj[slot])
                    ring_ttft1[slot] = None
                sess = ring[slot]
                traj = ring_traj[slot]
                traj_ttft1 = ring_ttft1[slot]
                sess.msgs = [sess.msgs[0]]
                if sess.turn == 0:
                    sess.turn = 1
                    user = CODE_TASKS[(idx * 7 + i) % len(CODE_TASKS)]
                else:
                    sess.turn += 1
                    user = sess.next_user(rnd)
                sess.msgs.append({"role": "user", "content": f"Turn {sess.turn}: {user}"})
            elif on_ab_pool:
                mode = routing
                url = ab.target(mode)
                headers = ab.headers(mode, agent_name(i))
                pool = HARDWARE["ab_pool"]
                mode_switched = sess_mode != mode
                if mode_switched:
                    sess = None
                    sess_mode = mode
                if sess is None or sess_i != i or sess.turn >= TURNS:
                    sess = _Session(agent_name(i), i % COHORTS, rnd)
                    sess.msgs[0]["content"] += f"\n{CODE_SYSTEM_PREFIX}\nYour codename is {cn}."
                    sess_i = i
                    with SIM.lock:
                        SIM.traj_seq += 1
                        traj = {
                            "id": "%s-g%d-%d" % (agent_name(i), SIM.gen, SIM.traj_seq),
                            "agent": agent_name(i), "codename": cn,
                            "pool": pool, "mode": mode,
                            "tier": TRAJ_TIERS[i % 3],
                            "started_ms": int(round(SIM._ms(mono()))),
                            "done": False,
                            "turns": [],
                        }
                        SIM.trajectories.append(traj)
                    traj_ttft1 = None
                if sess.turn == 0:
                    sess.turn = 1
                    user = CODE_TASKS[(idx * 7 + i) % len(CODE_TASKS)]
                else:
                    user = sess.next_user(rnd)
                sess.msgs.append({"role": "user", "content": user})
            else:
                mode = routing if ab is not None else None
                workload = "one-shot" if HARDWARE["ab_pool"] == "tpu" else "multi-turn"
                objective = "agentic-multi-turn" if workload == "multi-turn" else "oneshot-fast"
                other_svc = (
                    "http://llmd-gpu-service.llm-d-gpu.svc.cluster.local:8000/v1/chat/completions"
                    if HARDWARE["ab_pool"] == "tpu"
                    else "http://llmd-tpu-service.llm-d-serving.svc.cluster.local:8000/v1/chat/completions"
                )
                url = (os.environ.get("OTHER_BASELINE_URL", other_svc) if mode == "without" else gw)
                headers = {"Content-Type": "application/json"}
                if mode != "without":
                    headers.update({
                        "x-llm-d-workload": workload,
                        "x-llm-d-inference-fairness-id": agent_name(i),
                        "x-llm-d-inference-objective": objective,
                    })
                pool = "tpu" if HARDWARE["ab_pool"] == "gpu" else "gpu"
                mode_switched = sess_mode != mode
                if mode_switched:
                    sess = None
                    sess_mode = mode
                if sess is None or sess_i != i or sess.turn >= TURNS:
                    sess = _Session(agent_name(i), i % COHORTS, rnd)
                    sess.msgs[0]["content"] += f"\n{CODE_SYSTEM_PREFIX}\nYour codename is {cn}."
                    sess_i = i
                    with SIM.lock:
                        SIM.traj_seq += 1
                        traj = {
                            "id": "%s-g%d-%d" % (agent_name(i), SIM.gen, SIM.traj_seq),
                            "agent": agent_name(i), "codename": cn,
                            "pool": pool, "mode": mode,
                            "tier": TRAJ_TIERS[i % 3],
                            "started_ms": int(round(SIM._ms(mono()))),
                            "done": False,
                            "turns": [],
                        }
                        SIM.trajectories.append(traj)
                    traj_ttft1 = None
                if sess.turn == 0:
                    sess.turn = 1
                    user = CODE_TASKS[(idx * 7 + i) % len(CODE_TASKS)]
                else:
                    user = sess.next_user(rnd)
                sess.msgs.append({"role": "user", "content": user})
            if pool == "tpu":
                headers["X-data-parallel-rank"] = str((((i // 2) & 2) + rnd.randrange(2)) if mode == "without" else ((i // 2) % 4))
            if not url:
                time.sleep(1.0)
                continue
            req_body = {
                "model": model,
                "messages": sess.msgs,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.2,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            kv_m = ab.kv_tier_mode if (on_ab_pool and ab is not None) else "without"
            if on_ab_pool and ab is not None and ab.kv_tier_enabled and kv_m == "without":
                req_body["kv_transfer_params"] = {"max_load_tokens": 0, "max_offload_tokens": 0}
            payload = json.dumps(req_body).encode("utf-8")
            res = stream_chat(url, payload, headers, timeout=20.0)
            if res is None:
                if ring_mode:
                    ring[slot] = None
                else:
                    sess = None              # start this agent over with a fresh session after the back-off
                if on_ab_pool:
                    ab.add_error(mode)
                raise RuntimeError("request failed")
            ttft_ms, e2e_ms, out_tok, prompt_tok, txt = res
            sess.msgs.append({"role": "assistant", "content": (txt or "ok")[:400]})
            lat_ms = int(round(e2e_ms))
            # trajectory turn: cache_hit cannot be attributed per request on the live path; a later turn whose TTFT
            # is under half of this session's turn-1 TTFT is marked as a hit, anything else stays null.
            if traj_ttft1 is None:
                traj_ttft1 = ttft_ms
                if ring_mode:
                    ring_ttft1[slot] = ttft_ms
                hit = None
            else:
                hit = True if ttft_ms < 0.5 * traj_ttft1 else None
            reply = txt or ""
            if reply.startswith("```"):
                reply = "\n".join(l for l in reply.splitlines() if not l.strip().startswith("```")).strip()
            with SIM.lock:
                if len(traj["turns"]) < TURNS:
                    traj["turns"].append({
                        "turn": sess.turn, "user": user[:90], "reply": reply[:140],
                        "ttft_ms": int(round(ttft_ms)), "e2e_ms": lat_ms,
                        "out_tok": int(out_tok), "prompt_tok": int(prompt_tok),
                        "t_ms": int(round(SIM._ms(mono()))), "cache_hit": hit,
                    })
                    if len(traj["turns"]) >= TURNS:
                        traj["done"] = True
            if on_ab_pool:
                ab.add_sample(mode, ttft_ms, e2e_ms, out_tok, prompt_tok, fc=(mode == "with" and flow == "with"),
                              turn=sess.turn, kv_mode=kv_m)
            else:
                with SIM.lock:
                    SIM.live_samples.append((mono(), ttft_ms, e2e_ms, out_tok))
                    cutoff = mono() - 60.0
                    while SIM.live_samples and SIM.live_samples[0][0] < cutoff:
                        SIM.live_samples.popleft()
            if txt:
                # Strip markdown fences if the model added them; keep the first code line(s) readable.
                if txt.startswith("```"):
                    txt = "\n".join(l for l in txt.splitlines() if not l.strip().startswith("```")).strip()
                txt = txt[:220]
                if cn.lower() not in txt.lower():
                    txt = f"{cn} › {txt}"
                with SIM.lock:
                    SIM.seq += 1
                    SIM.m_last[i].append(txt)
                    SIM.ticker.append({
                        "seq": SIM.seq,
                        "agent": agent_name(i),
                        "text": txt,
                        "latency_ms": lat_ms,
                        "t_ms": int(round(SIM._ms(mono()))),
                        # PODS[0] is the TPU card, PODS[1] the GPU card in the stack view
                        "tag": "premium" if strat == "priority" else PODS[0 if pool == "tpu" else 1],
                        "memory": "recalled",
                        "codename": cn,
                        "survived": max(1, SIM.m_survived[i]),
                    })
            if not ring_mode and sess.turn >= TURNS:
                idx += nworkers          # session done -> next agent
            # Fixed think time per agent (Oct 9): every agent issues one turn per AGENT_THINK_S regardless of how fast
            # its pool answered, so both pools see the same request rate and neither reads as "getting more traffic".
            if PACING["think_s"] <= 0 and mode == "with":
                time.sleep(0.005)
            else:
                time.sleep(max(0.05, PACING["think_s"] - (mono() - t_turn)) + rnd.uniform(0.0, 0.05))
        except Exception:
            time.sleep(2.0)


def main():
    global HARNESS, SEND_CLUSTER, AB, MODEL
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8085"))
    gateway_url = os.environ.get("GATEWAY_URL", "http://llm-d-gateway.llm-d-serving.svc.cluster.local/v1/chat/completions")
    model = os.environ.get("MODEL_NAME", "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic")
    # "Without llm-d" baseline: comma-separated URLs round-robined by the bench (pod port-forwards locally,
    # or a single k8s Service URL in-cluster, which kube-proxy round-robins).
    baseline_urls = os.environ.get("BASELINE_URLS", "")
    ab_concurrency = int(os.environ.get("AB_CONCURRENCY", "0") or 0)   # Oct 9: 0 = agents drive the A/B pool
    pod_metrics = os.environ.get("POD_METRICS_HOSTS", "") or ",".join(
        h for h in (os.environ.get("TPU_POD1_HOST"), os.environ.get("TPU_POD2_HOST"), os.environ.get("GPU_POD1_HOST")) if h)
    # The pool the A/B drives (B2, Oct 7): KV-aware routing needs >= 2 endpoints to show a delta. AB_KV_HOSTS is the
    # subset of POD_METRICS_HOSTS that pool is scraped from; AB_WORKLOAD picks the gateway route to that pool.
    ab_kv_hosts = os.environ.get("AB_KV_HOSTS", "")
    # Oct 14: scale-out replicas with the shared-storage tier (Tier 3 panel). AB_KV_STORAGE_HOSTS is their headless
    # Service (per-pod scrape and per-pod targets), AB_KV_STORAGE_URLS a fallback chat endpoint.
    ab_storage_hosts = os.environ.get("AB_KV_STORAGE_HOSTS", "")
    ab_storage_urls = os.environ.get("AB_KV_STORAGE_URLS", "")
    ab_workload = os.environ.get("AB_WORKLOAD", "multi-turn" if HARDWARE["ab_pool"] == "tpu" else "one-shot")
    ab_objective = os.environ.get("AB_OBJECTIVE", "agentic-multi-turn" if ab_workload == "multi-turn" else "oneshot-fast")
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
        elif a.startswith("--baseline-url="):
            baseline_urls = a.split("=", 1)[1]
        elif a == "--baseline-url" and j + 1 < len(args):
            baseline_urls = args[j + 1]
        elif a.startswith("--ab-concurrency="):
            ab_concurrency = int(a.split("=", 1)[1])
        elif a.startswith("--pod-metrics="):
            pod_metrics = a.split("=", 1)[1]
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
    # Keep the UI's config.model in sync with the model we actually send requests for.
    MODEL = model
    os.environ["GATEWAY_URL"] = gateway_url
    if auto_warm:
        warm_to_hermes_live()

    # Live A/B bench (Pane 2: Without llm-d vs With llm-d) — only when we have somewhere to send traffic.
    try:
        from abbench import ABBench
        AB = ABBench(
            with_url=gateway_url,
            without_urls=[u for u in baseline_urls.split(",") if u.strip()],
            model=model,
            concurrency=ab_concurrency,
            metrics_hosts=[h.strip() for h in pod_metrics.split(",") if h.strip()],
            kv_hosts=[h.strip() for h in ab_kv_hosts.split(",") if h.strip()],
            storage_hosts=[h.strip() for h in ab_storage_hosts.split(",") if h.strip()],
            storage_urls=[u.strip() for u in ab_storage_urls.split(",") if u.strip()],
            workload=ab_workload, objective=ab_objective,
            is_active=lambda: SIM.phase == "running" and SIM.rate > 0,
            get_mode=lambda: SIM.routing_mode,
            get_flow=lambda: SIM.flow_mode,
        )
        if AB.enabled():
            AB.start()
    except Exception as exc:  # never let the bench take the dashboard down
        print("[ab] disabled: %s" % exc, file=sys.stderr, flush=True)
        AB = None

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=sim_loop, daemon=True).start()
    # Real Hermes agent traffic on both pools (even workers: agents' pool via the gateway; odd: the A/B pool).
    live_workers = max(1, int(os.environ.get("LIVE_GATEWAY_WORKERS", "16") or 16))
    other_pool = "tpu" if HARDWARE["ab_pool"] == "gpu" else "gpu"
    if AB is not None and AB.enabled() and live_workers > 1:
        AGENTS_BY_POOL.update({other_pool: (live_workers + 1) // 2, HARDWARE["ab_pool"]: live_workers // 2})
    else:
        AGENTS_BY_POOL.update({other_pool: live_workers, HARDWARE["ab_pool"]: 0})
    for w in range(live_workers):
        threading.Thread(target=live_gateway_loop, args=(gateway_url, model, w, live_workers), daemon=True,
                         name=f"live-gw-{w}").start()
    # Oct 14: the storage ring (Tier 3 panel) only runs when scale-out replicas are configured.
    if AB is not None and getattr(AB, "kv_storage_replicas", False):
        st_workers = max(1, int(os.environ.get("KV_STORAGE_WORKERS", "4") or 4))
        for w in range(st_workers):
            threading.Thread(target=storage_ring_loop, args=(model, w, st_workers), daemon=True,
                             name=f"kv-storage-{w}").start()
    print("benchmark live driver (harness %s): http://%s:%d/  (gateway=%s, baseline=%s, ab_conc=%d, ab_workload=%s, pod_metrics=%s, ab_kv_hosts=%s, agents=%s, think=%.2fs)"
          % (HARNESS, host, port, gateway_url, baseline_urls or "-", ab_concurrency, ab_workload, pod_metrics or "-", ab_kv_hosts or "-", AGENTS_BY_POOL, AGENT_THINK_S), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
