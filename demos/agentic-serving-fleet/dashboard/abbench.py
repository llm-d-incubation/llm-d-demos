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
"""
abbench.py — live "Without llm-d vs With llm-d" A/B measurement for the llm-d dashboard.

Drives the SAME model pool two ways and measures what the user actually feels:

  * without  -> round-robin across the vLLM replicas, bypassing the llm-d EPP
                (a list of pod/service URLs; client-side RR, or a k8s Service URL
                which kube-proxy round-robins in-cluster)
  * with     -> through the llm-d gateway (KV-cache-aware + load-aware EPP)

Every request is a streaming chat completion so we capture, per request:
  TTFT (first content token), E2E, output tokens (from stream usage).

Traffic shape mirrors the Hermes multi-turn agents: N cohorts share a long
"workspace" system prefix, and each session grows over TURNS turns, so prefix
cache locality is exactly what cache-aware routing can exploit.

Oct 9: in the cluster deployment there are NO synthetic sessions any more
(AB_CONCURRENCY=0). The server's Hermes agents are split across both pools; the
ones on the A/B pool follow the routing toggle and report every finished turn
through add_sample(), so "Without llm-d vs With llm-d" is measured on the exact
same agent workload the other pool carries. The worker threads below are kept
for local runs / other deployments (AB_CONCURRENCY > 0).

Aggregates (rolling AGG_WINDOW_S) per routing mode:
  tok_s, req_s, ttft_p50/p95, e2e_p50/p95, cache_hit_pct (from vLLM /metrics if reachable)

Exposes:
  AB.compare()  -> dict for state.llmd.compare   (routing_mode, routing.{without,with})
  AB.point()    -> dict merged into each state.llmd.history entry
"""

import json
import os
import random
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import deque

AGG_WINDOW_S = 20.0      # window for tok/s + percentiles
ST_WINDOW_S = 20.0       # window for the scale-out replicas' hit split and Lustre read/write rates
MIN_AGG_N = 6            # B1: don't publish an aggregate from fewer samples than this ...
MIN_AGG_SPAN_S = 6.0     # ... or a shorter span (mode switches otherwise spike/dip the throughput chart)
HIST_KEEP = 180          # 1-s points kept for the time series
# Oct 9: turns per session = number of replicas on the A/B pool (4): round-robin then lands every turn of a session on a
# different pod (0 prefix hits), the KV-aware router hits turns 2..N. Env-overridable (AB_TURNS).
TURNS = int(os.environ.get("AB_TURNS", "4") or 4)
COHORTS = 8              # shared-prefix cohorts (workspaces)
# Oct 9: env-overridable workload shape. A heavier prefill share (56 paras / 32 out) was measured and did NOT widen the
# A/B (prefill is cheap on GB200; TTFT is scheduling/decode-bound) — the structural lever is the pod count, not the prefix.
PREFIX_PARAS = int(os.environ.get("AB_PREFIX_PARAS", "160") or 160)   # ≈9.5k-token workspace prefix per session (Oct 9 19:30Z: +76% tok/s vs round-robin on 4 replicas)
REQ_TIMEOUT_S = 60.0
MAX_TOKENS = int(os.environ.get("AB_MAX_TOKENS", "32") or 32)
LOAD_FULL_REQS = 4.0     # Oct 9: requests in flight per pool that light every chip glyph fully (Accelerators row);
                         # 8 paced agents per pool keep ~1.5-4 in flight, so 4 = "busy"

_LOCK = threading.Lock()


def _env_int(name, default):
    """Integer env var with a fallback. A value that does not parse (e.g. an unexpanded template placeholder)
    must degrade to the default instead of raising, which would disable the whole bench."""
    raw = (os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        print("[ab] ignoring %s=%r (not a number), using %r" % (name, raw, default), flush=True)
        return default


def _now():
    return time.monotonic()


def _unix_ms():
    return int(time.time() * 1000)


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return int(round(s[min(len(s) - 1, int(q * len(s)))]))


_DNS_CACHE = {}
_DNS_LOCK = threading.Lock()


def _resolve_url(url, headers):
    """Rewrite hostname in url to its cached IPv4 address (TTL 60s) and preserve Host header.
    Avoids per-request glibc getaddrinfo UDP storms across ndots:5 search domains under 48 threads."""
    try:
        import socket
        p = urllib.parse.urlsplit(url)
        host = p.hostname
        if not host or host[0].isdigit():
            return url, headers
        now = _now()
        with _DNS_LOCK:
            ent = _DNS_CACHE.get(host)
        if not ent or now - ent[1] > 60.0:
            ip = socket.gethostbyname(host)
            with _DNS_LOCK:
                _DNS_CACHE[host] = (ip, now)
        else:
            ip = ent[0]
        netloc = f"{ip}:{p.port}" if p.port else ip
        hdrs = dict(headers) if headers else {}
        hdrs.setdefault("Host", p.netloc)
        return urllib.parse.urlunsplit((p.scheme, netloc, p.path, p.query, p.fragment)), hdrs
    except Exception:
        return url, headers


def stream_chat(url, payload, headers, timeout=REQ_TIMEOUT_S):
    """POST a streaming chat completion; returns (ttft_ms, e2e_ms, completion_tokens, prompt_tokens, text) or None.

    Shared by the A/B bench and the agents' live gateway loop (Oct 9) so both pools are measured the same way.
    """
    r_url, r_headers = _resolve_url(url, headers)
    req = urllib.request.Request(r_url, data=payload, headers=r_headers, method="POST")
    t0 = _now()
    ttft = None
    chunks = 0
    usage = None
    text = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    c = d.get("content")
                    if c:
                        if ttft is None:
                            ttft = (_now() - t0) * 1000.0
                        chunks += 1
                        text.append(c)
    except Exception:
        return None
    e2e = (_now() - t0) * 1000.0
    if ttft is None:
        ttft = e2e
    out_tok = int((usage or {}).get("completion_tokens") or chunks)
    prompt_tok = int((usage or {}).get("prompt_tokens") or 0)
    return ttft, e2e, out_tok, prompt_tok, "".join(text).strip()


_COHORT_BODY_CACHE = {}


def _workspace_prefix(cohort, rnd, sid="", epoch=0):
    """Long system prefix (~1.6k tokens) for one agent session.

    The FIRST line is unique per session+epoch (agent id + nonce), so the prefix is only ever cached on the
    pod(s) that served *this* session. That is what makes the A/B honest: with only a handful of shared
    cohort prefixes every pod ends up caching all of them and round-robin hits ~98% too (measured
    2026-10-08 on 2x GB200: 875 vs 889 tok/s, "1.0x"). Real agents each carry their own workspace context.
    """
    key = (cohort, PREFIX_PARAS)
    body = _COHORT_BODY_CACHE.get(key)
    if body is None:
        topics = ("tensor parallelism", "KV cache paging", "speculative decoding", "FP8 quantization",
                  "expert parallelism", "prefix caching", "continuous batching", "chunked prefill",
                  "disaggregated serving", "torch.compile", "flash attention", "custom kernels")
        paras = []
        for k in range(PREFIX_PARAS):
            t = topics[(cohort * 7 + k) % len(topics)]
            paras.append(
                f"Workspace W{cohort:02d} note {k + 1}: the team documented {t} behaviour under agentic load. "
                f"Observed that {t} interacts with request fan-out, retry budgets, and session affinity; "
                f"record findings as short bullets, prefer measured numbers, cite the run id W{cohort:02d}-{k * 13 % 97:02d}."
            )
        body = "\n".join(paras)
        _COHORT_BODY_CACHE[key] = body
    nonce = "%08x" % rnd.getrandbits(32)
    return (f"Agent session {sid} epoch {epoch} nonce {nonce}: you are a Hermes coding agent working in "
            f"workspace W{cohort:02d}. Answer in one short sentence.\n\n" + body)


class _Session:
    __slots__ = ("sid", "cohort", "turn", "msgs", "prefix", "rnd", "epoch")

    def __init__(self, sid, cohort, rnd):
        self.sid = sid
        self.cohort = cohort
        self.rnd = rnd
        self.epoch = 0
        self.reset()

    def reset(self):
        # a new session = a new agent context: fresh unique prefix, nothing of it is cached anywhere yet
        self.epoch += 1
        self.prefix = _workspace_prefix(self.cohort, self.rnd, self.sid, self.epoch)
        self.turn = 0
        self.msgs = [{"role": "system", "content": self.prefix}]

    def next_user(self, rnd):
        self.turn += 1
        qs = ("Summarize the most important note so far in one sentence.",
              "Which note mentions the highest run id? One sentence.",
              "Give one concrete next step for this workspace.",
              "What is the main risk you see? One sentence.",
              "Name one metric we should track. One sentence.")
        return qs[(self.turn - 1) % len(qs)]


class ABBench:
    def __init__(self, with_url, without_urls, model, concurrency=12, metrics_hosts=(), is_active=None, get_mode=None,
                 get_flow=None, kv_hosts=(), workload="multi-turn", objective="agentic-multi-turn",
                 storage_hosts=(), storage_urls=()):
        self.with_url = self._chat_url(with_url) if with_url else None
        self.without_urls = [self._chat_url(u) for u in without_urls if u]
        self.model = model
        # Oct 9: concurrency 0 = no synthetic sessions at all; the server's Hermes agents feed samples in through
        # add_sample()/add_error() instead, so the A/B pool carries the same workload as the other pool.
        self.concurrency = max(0, int(concurrency))
        self.metrics_hosts = [h for h in metrics_hosts if h]       # every vLLM Service we scrape (per-pool KV hit)
        self.kv_hosts = [h for h in kv_hosts if h] or list(self.metrics_hosts[:1])   # the pool the A/B drives (aggregate)
        self.workload = workload                                     # x-llm-d-workload header -> which EPP/pool the gateway picks
        self.objective = objective
        self.kv_by_host = {}                                         # host -> hit % (last scrape window)
        self.load_by_host = {}                                       # host -> vllm:num_requests_running summed over its pods
        # Oct 9: per-pod view of the same scrape, host -> [{"ip", "cache_hit_pct", "running"}, ...] sorted by ip
        self.kv_by_pod = {}
        self.is_active = is_active or (lambda: True)
        self.get_mode = get_mode or (lambda: "without")
        self.get_flow = get_flow or (lambda: "without")
        self.samples = deque()                 # (t_mono, mode, ttft_ms, e2e_ms, out_tok, prompt_tok, fc)
        self.history = deque(maxlen=HIST_KEEP)
        # "with_fc": the llm-d router measured while the flow-control segment is on (closing summary, col 3).
        self.measured = {"without": None, "with": None, "with_fc": None}
        self.errors = {"without": 0, "with": 0}
        self.rr = 0
        self.kv_prev = {}                      # host -> (hits, queries)
        self.kv_hit_pct = None
        self.kv_hbm_hit_pct = None
        # Oct 10: KV cache tiering (HBM -> host DRAM, screen 4) scraped from the same /metrics payload
        _ab_pool = (os.environ.get("AB_POOL", "") or ("tpu" if os.environ.get("AB_WORKLOAD", "one-shot") == "multi-turn" else "gpu")).strip().lower()
        _pool_tier_env = "ENABLE_KV_OFFLOAD" if _ab_pool == "tpu" else "ENABLE_KV_TIERING"
        self.kv_tier_enabled = (
            os.environ.get("KV_TIER_ENABLED", os.environ.get(_pool_tier_env, "0"))
            == "1"
        )
        # Oct 14: shared-storage tier (Lustre). Either the A/B pool itself has the fs tier (ENABLE_KV_STORAGE_TIER=1)
        # or a separate set of scale-out replicas does (AB_KV_STORAGE_HOSTS, see storage_* below).
        self.kv_storage_enabled = os.environ.get("ENABLE_KV_STORAGE_TIER", "0") == "1"
        self.kv_tier_hbm_cap = _env_int("KV_TIER_HBM_CAP_BYTES", 0) or None
        self.kv_tier_host_bytes = _env_int("KV_TIER_CPU_BYTES", 68719476736) or 68719476736
        self.kv_storage_bytes = _env_int("KV_TIER_STORAGE_BYTES", _env_int("KV_STORAGE_BYTES", 1024 * 2 ** 30)) or (1024 * 2 ** 30)
        # Oct 14: both node pools can run a host tier (GPU pool: ENABLE_KV_TIERING, TPU pool: ENABLE_KV_OFFLOAD). The
        # Memory panel shows a live Tier 2 row for every pool that has one, not only for the A/B pool, so each pool
        # gets its own flag and (optional) host pool size; a pool whose /metrics show offload traffic counts as tiered.
        self.kv_tier_pools = {"gpu": os.environ.get("ENABLE_KV_TIERING", "0") == "1",
                              "tpu": os.environ.get("ENABLE_KV_OFFLOAD", "0") == "1"}
        if self.kv_tier_enabled:
            self.kv_tier_pools[_ab_pool] = True
        self.kv_host_bytes_by_pool = {p: _env_int("KV_TIER_CPU_BYTES_" + p.upper(), 0) or self.kv_tier_host_bytes for p in ("gpu", "tpu")}
        self.kv_host_win = {}                  # host -> deque of (t_mono, dh, dq, dext_h, dext_q, db_restore, db_offload)
        self.kv_pod_win = {}                   # (host, ip) -> deque of (t_mono, dh, dq, dext_h)
        self._last_routing_mode = None
        self.mode_since = 0.0
        self.kv_host_cum = {}                  # host -> [gb_offloaded, gb_restored] since start (or since reset)
        self.kv_tier_by_host = {}              # host -> live host-tier stats (compare.by_pool.<pool>.kv_tier)
        self.kv_tier_mode = "without"
        self.kv_ring_active = False
        self.kv_pod_ips = []
        self.kv_scrape_win = deque()
        self.kv_ext_prev = {}                  # (host, ip) -> (t_mono, ext_h, ext_q, b_restore, b_offload, p_h, fs_h, fs_r, fs_w)
        self.kv_fs_base = {}                   # (host, ip) -> (fs_r_base, fs_w_base, h_base, q_base, ext_h_base, p_h_base, fs_h_base)
        self.kv_boot_t = None
        self.kv_pod_seen = set()
        self.kv_pod_new_since = {}             # (host, ip) -> t_mono when scaled-out or restarted pod appeared
        self.kv_tier_pods = []
        self.kv_ext_hit_pct = None
        self.kv_storage_hit_pct = None
        self.kv_restore_gbps = 0.0
        self.kv_offload_gbps = 0.0
        self.kv_storage_read_gbps = 0.0
        self.kv_storage_write_gbps = 0.0
        self.kv_hbm_used_pct = 0.0
        self.kv_tok_restored = 0.0
        self.kv_gb_restored = 0.0
        self.kv_gb_offloaded = 0.0
        self.kv_gb_storage_read = 0.0
        self.kv_gb_storage_written = 0.0
        self.kv_restore_hist = deque([0.0] * 60, maxlen=60)
        self.kv_samples = deque()              # (t_mono, kv_mode, turn, ttft_ms, out_tok, prompt_tok)
        self.kv_measured = {"without": None, "with": None}
        # Oct 14: scale-out replicas with the shared-storage tier (llmd-gpu-storage), driven by the server's
        # storage ring and scraped separately from the A/B pool. Empty when the dashboard has no such replicas.
        self.st_hosts = [h for h in (storage_hosts or ()) if h]
        self.st_urls = [self._chat_url(u) for u in (storage_urls or ()) if u]
        self.kv_storage_replicas = bool(self.st_hosts or self.st_urls)
        if self.kv_storage_replicas:
            self.kv_storage_enabled = True
        self.st_hbm_cap = int(os.environ.get("KV_STORAGE_REPLICA_HBM_CAP_BYTES", "8589934592") or 8589934592)
        self.st_host_bytes = int(os.environ.get("KV_STORAGE_REPLICA_CPU_BYTES", "8589934592") or 8589934592)
        self.st_pod_ips = []
        self.st_prev = {}                      # ip:port -> (t, h, q, ext_h, b_restore, b_offload, fs_r, fs_w)
        self.st_base = {}                      # ip:port -> (ext_h, fs_r, fs_w) at the last reset
        self.st_seen = set()
        self.st_new_since = {}
        self.st_boot_t = None
        self.st_win = deque()                  # (t, dh, dq, dext_h, db_r, dfs_r, dfs_w, db_o)
        self.st_pod_win = {}                   # ip -> deque of (t, dh, dq, dext_h)
        self.st_pods = []
        self.st_live = {}
        self.st_tok_restored = 0.0
        self.st_samples = deque()              # (t_mono, leg, turn, ttft_ms, e2e_ms, out_tok, prompt_tok)
        self.st_measured = {"restore": None, "recompute": None}
        self.st_errors = 0
        self.last_point = {}
        self.rnd = random.Random(7)
        self._threads = []

    @staticmethod
    def _chat_url(u):
        u = u.strip()
        return u if u.endswith("/v1/chat/completions") else u.rstrip("/") + "/v1/chat/completions"

    def set_kv_ring(self, active):
        with _LOCK:
            self.kv_ring_active = bool(active)

    def pod_chat_url(self, slot):
        with _LOCK:
            ips = list(self.kv_pod_ips)
        if ips:
            ip, port = ips[slot % len(ips)]
            return f"http://{ip}:{port}/v1/chat/completions"
        return self._target("with")

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if not self.with_url and not self.without_urls:
            return
        for i in range(self.concurrency):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True, name=f"ab-worker-{i}")
            t.start()
            self._threads.append(t)
        threading.Thread(target=self._agg_loop, daemon=True, name="ab-agg").start()

    def enabled(self):
        return bool(self.with_url or self.without_urls)

    # ------------------------------------------------------------------ external samples (Oct 9)
    def _check_mode_switch_locked(self, cur_rmode, now):
        if self._last_routing_mode != cur_rmode:
            self._last_routing_mode = cur_rmode
            self.mode_since = now
            self.samples = deque(s for s in self.samples if s[1] != cur_rmode)
            self.kv_host_win.clear()
            self.kv_pod_win.clear()
            self.kv_scrape_win.clear()

    def add_sample(self, mode, ttft_ms, e2e_ms, out_tok, prompt_tok=0, fc=False, turn=1, kv_mode=None):
        """Record one finished request made by an outside driver (the server's Hermes agents) against the A/B
        pool in routing mode `mode`. It joins the same window/aggregation as the synthetic workers would."""
        now = _now()
        cur_rmode = self.get_mode()
        with _LOCK:
            self._check_mode_switch_locked(cur_rmode, now)
            self.samples.append((now, mode, float(ttft_ms), float(e2e_ms), int(out_tok), int(prompt_tok), bool(fc)))
            km = kv_mode or self.kv_tier_mode
            self.kv_samples.append((now, "with" if km == "storage" else km, int(turn), float(ttft_ms), int(out_tok), int(prompt_tok)))

    def add_error(self, mode):
        with _LOCK:
            if mode in self.errors:
                self.errors[mode] += 1

    def target(self, mode):
        """Chat-completions URL for routing mode `mode` ('with' = gateway, 'without' = baseline round-robin)."""
        return self._target(mode)

    def headers(self, mode, fairness_id):
        h = {"Content-Type": "application/json"}
        if mode == "with":
            h.update({
                "x-llm-d-workload": self.workload,
                "x-llm-d-inference-fairness-id": fairness_id,
                "x-llm-d-inference-objective": self.objective,
            })
        return h

    # ------------------------------------------------------------------ workers
    def _target(self, mode):
        if mode == "with":
            return self.with_url or (self.without_urls[0] if self.without_urls else None)
        if self.without_urls:
            with _LOCK:
                u = self.without_urls[self.rr % len(self.without_urls)]
                self.rr += 1
            return u
        return self.with_url

    def _worker(self, idx):
        rnd = random.Random(hash((time.time_ns(), idx)))   # never replay prefixes across restarts (see server.py)
        sess = _Session(f"ab-{idx:03d}", idx % COHORTS, rnd)
        sess_mode = None
        while True:
            if not self.is_active():
                time.sleep(0.4)
                continue
            mode = self.get_mode()
            url = self._target(mode)
            if not url:
                time.sleep(1.0)
                continue
            if sess_mode != mode or sess.turn >= TURNS:
                sess.reset()
                sess_mode = mode
            sess.msgs.append({"role": "user", "content": sess.next_user(rnd)})
            headers = {"Content-Type": "application/json"}
            if mode == "with":
                headers.update({
                    "x-llm-d-workload": self.workload,
                    "x-llm-d-inference-fairness-id": sess.sid,
                    "x-llm-d-inference-objective": self.objective,
                })
            if self.workload == "multi-turn" and mode != "without":
                headers["X-data-parallel-rank"] = str(idx % 4)
            payload = json.dumps({
                "model": self.model,
                "messages": sess.msgs,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.2,
                "stream": True,
                "stream_options": {"include_usage": True},
            }).encode("utf-8")
            res = self._stream_once(url, payload, headers)
            if res is None:
                with _LOCK:
                    self.errors[mode] += 1
                sess.msgs.pop()
                time.sleep(1.0)
                continue
            ttft, e2e, out_tok, prompt_tok, text = res
            sess.msgs.append({"role": "assistant", "content": text or "ok"})
            fc = (mode == "with" and self.get_flow() == "with")
            now = _now()
            cur_rmode = self.get_mode()
            with _LOCK:
                self._check_mode_switch_locked(cur_rmode, now)
                self.samples.append((now, mode, ttft, e2e, out_tok, prompt_tok, fc))
            time.sleep(rnd.uniform(0.05, 0.25))

    def _stream_once(self, url, payload, headers):
        """Returns (ttft_ms, e2e_ms, completion_tokens, prompt_tokens, text) or None on error."""
        return stream_chat(url, payload, headers)

    # ------------------------------------------------------------------ KV hit from vLLM /metrics
    def _scrape_kv(self):
        """Per-pod prefix-cache hit rate for the last scrape window.

        Each entry of metrics_hosts is a Service host:port. We resolve it to ALL its A records (point it at a
        headless Service so that is one record per pod) and scrape every pod separately with its own counter
        history. Scraping a ClusterIP Service directly lands on a random pod per request, which produced
        impossible deltas (156 %) on a 2-pod pool. A pool with no queries in the window reports None ("—"),
        never a stale number; counter resets (pod restart) drop one window instead of going negative.
        """
        import socket
        if not self.metrics_hosts:
            return
        cur_rmode = self.get_mode()
        now = _now()
        with _LOCK:
            self._check_mode_switch_locked(cur_rmode, now)
        hits_d = queries_d = 0.0
        ext_hits_d = ext_queries_d = 0.0
        restore_bps_sum = offload_bps_sum = 0.0
        fs_read_bps_sum = fs_write_bps_sum = 0.0
        db_r_sum = db_o_sum = 0.0
        dp_h_sum = dfs_h_sum = dfs_r_sum = dfs_w_sum = 0.0
        cum_fs_r_sum = cum_fs_w_sum = 0.0
        cum_ph_sum = cum_fsh_sum = 0.0
        seen_fs = False
        hbm_u_vals = []
        got = False
        by_host = {}
        load_by_host = {}
        by_pod = {}
        kv_pod_ips = []
        tier_pods_raw = []
        tier_by_host = {}
        for host in self.metrics_hosts:
            name, _, port = host.partition(":")
            port = port or "8000"
            try:
                ips = sorted({ai[4][0] for ai in socket.getaddrinfo(name, int(port), socket.AF_INET, socket.SOCK_STREAM)})
            except Exception:
                ips = []
            if not ips:
                by_host[host] = None
                by_pod[host] = []
                continue
            if host in self.kv_hosts:
                kv_pod_ips.extend((ip, port) for ip in ips)
            dh_sum = dq_sum = dext_h_sum = 0.0
            running_sum = 0.0
            hd_ext_h = hd_ext_q = hd_br = hd_bo = 0.0      # this pool's host-tier deltas over all its pods
            h_hbm_vals = []
            h_cap_bytes = 0
            pods = []
            for ip in ips:
                try:
                    with urllib.request.urlopen(f"http://{ip}:{port}/metrics", timeout=0.9) as r:
                        txt = r.read().decode("utf-8", errors="replace")
                except Exception:
                    continue
                h = q = running = 0.0
                ext_h = ext_q = b_restore = b_offload = 0.0
                p_h = fs_h = fs_r = fs_w = 0.0
                dext_h = dext_q = db_r = db_o = 0.0
                dp_h = dfs_h = dfs_r = dfs_w = 0.0
                hbm_u = None
                for line in txt.splitlines():
                    if line.startswith("vllm:prefix_cache_hits_total"):
                        h += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith("vllm:prefix_cache_queries_total"):
                        q += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith("vllm:num_requests_running"):
                        running += float(line.rsplit(" ", 1)[-1])     # requests in flight on this pod right now
                    elif line.startswith("vllm:external_prefix_cache_hits_total"):
                        ext_h += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith("vllm:external_prefix_cache_queries_total"):
                        ext_q += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith("vllm:kv_offload_total_bytes_total"):
                        v = float(line.rsplit(" ", 1)[-1])
                        if "CPU_to_GPU" in line:
                            b_restore += v
                        elif "GPU_to_CPU" in line:
                            b_offload += v
                    elif line.startswith("vllm:kv_offload_tiering_"):
                        if 'tier="1:fs"' in line:
                            seen_fs = True
                        if line.startswith("vllm:kv_offload_tiering_chunk_hits_total"):
                            v = float(line.rsplit(" ", 1)[-1])
                            if 'tier="0:primary"' in line:
                                p_h += v
                            elif 'tier="1:fs"' in line:
                                fs_h += v
                        elif line.startswith("vllm:kv_offload_tiering_read_bytes_total") and 'tier="1:fs"' in line:
                            fs_r += float(line.rsplit(" ", 1)[-1])
                        elif line.startswith("vllm:kv_offload_tiering_write_bytes_total") and 'tier="1:fs"' in line:
                            fs_w += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith(("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")):
                        hbm_u = float(line.rsplit(" ", 1)[-1]) * 100.0
                    elif line.startswith("vllm:cache_config_info{"):
                        m_cap = re.search(r'kv_cache_memory_bytes="(\d+)"', line)
                        if m_cap:
                            h_cap_bytes += int(m_cap.group(1))
                        else:
                            m_tok = re.search(r'kv_cache_size_tokens="(\d+)"', line)
                            if m_tok:
                                h_cap_bytes += int(m_tok.group(1)) * 46 * 1024
                running_sum += running
                key = (host, ip)
                eprev = self.kv_ext_prev.get(key)
                self.kv_ext_prev[key] = (now, ext_h, ext_q, b_restore, b_offload, p_h, fs_h, fs_r, fs_w)
                if hbm_u is not None:
                    h_hbm_vals.append(hbm_u)
                # Offload counters are read for every pool (both node pools can run a host tier); the A/B pool
                # additionally feeds the tier legs and the storage-tier figures below.
                mono = bool(eprev and ext_h >= eprev[1] and ext_q >= eprev[2] and b_restore >= eprev[3] and b_offload >= eprev[4] and (len(eprev) < 9 or (p_h >= eprev[5] and fs_h >= eprev[6] and fs_r >= eprev[7] and fs_w >= eprev[8])))
                dt = max(0.1, now - eprev[0]) if eprev else 1.0
                if mono:
                    dext_h, dext_q = ext_h - eprev[1], ext_q - eprev[2]
                    db_r, db_o = b_restore - eprev[3], b_offload - eprev[4]
                    if len(eprev) >= 9:
                        dp_h = max(0.0, p_h - eprev[5])
                        dfs_h = max(0.0, fs_h - eprev[6])
                        dfs_r = max(0.0, fs_r - eprev[7])
                        dfs_w = max(0.0, fs_w - eprev[8])
                    hd_ext_h += dext_h
                    hd_ext_q += dext_q
                    hd_br += db_r
                    hd_bo += db_o
                prev = self.kv_prev.get(key)
                self.kv_prev[key] = (h, q)
                if self.kv_boot_t is not None and key not in self.kv_pod_seen:
                    self.kv_pod_new_since[key] = now
                elif prev and (h < prev[0] or q < prev[1]):
                    self.kv_pod_new_since[key] = now
                    self.kv_fs_base.pop(key, None)
                self.kv_pod_seen.add(key)
                base = self.kv_fs_base.get(key) or (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                cum_r = max(0.0, fs_r)
                cum_w = max(0.0, fs_w)
                cum_h = max(0.0, h - base[2])
                cum_q = max(0.0, q - base[3])
                cum_eh = max(0.0, ext_h - base[4])
                cum_ph = max(0.0, p_h)
                cum_fsh = max(0.0, fs_h)
                if host in self.kv_hosts:
                    cum_fs_r_sum += cum_r
                    cum_fs_w_sum += cum_w
                    cum_ph_sum += cum_ph
                    cum_fsh_sum += cum_fsh
                    if hbm_u is not None:
                        hbm_u_vals.append(hbm_u)
                    if mono:
                        ext_hits_d += dext_h
                        ext_queries_d += dext_q
                        db_r_sum += db_r
                        db_o_sum += db_o
                        dp_h_sum += dp_h
                        dfs_h_sum += dfs_h
                        dfs_r_sum += dfs_r
                        dfs_w_sum += dfs_w
                        restore_bps_sum += db_r / dt
                        offload_bps_sum += db_o / dt
                        fs_read_bps_sum += dfs_r / dt
                        fs_write_bps_sum += dfs_w / dt
                        with _LOCK:
                            self.kv_tok_restored += dext_h
                            self.kv_gb_restored += db_r / 1e9
                            self.kv_gb_offloaded += db_o / 1e9
                dh = dq = 0.0
                win_hit = None
                if prev and h >= prev[0] and q >= prev[1]:
                    dh, dq = h - prev[0], q - prev[1]
                    if dext_q > 0 and dq > dh + dext_q:
                        dq = dh + dext_q
                    dh_sum += dh
                    dq_sum += dq
                    dext_h_sum += dext_h
                pwin = self.kv_pod_win.setdefault(key, deque())
                pwin.append((now, dh, dq, dext_h))
                while pwin and pwin[0][0] < now - 8.0:
                    pwin.popleft()
                pw_h = sum(x[1] for x in pwin)
                pw_q = sum(x[2] for x in pwin)
                pw_eh = sum(x[3] for x in pwin)
                if pw_q > 0:
                    win_hit = round(min(100.0, 100.0 * (pw_h + pw_eh) / pw_q), 1)
                pods.append({"ip": ip, "port": port, "running": int(round(running)), "cache_hit_pct": win_hit})
                if host in self.kv_hosts:
                    cum_hit = round(min(100.0, 100.0 * (cum_h + cum_eh) / cum_q), 1) if cum_q > 0 else None
                    pod_hit = win_hit if win_hit is not None else cum_hit
                    if (dp_h + dfs_h) > 0 and dq > 0:
                        pod_st_pct = round(min(100.0, 100.0 * dext_h / dq) * (dfs_h / (dp_h + dfs_h)), 1)
                    elif db_r > 0 and dfs_r > 0 and dq > 0:
                        pod_st_pct = round(min(100.0, 100.0 * dext_h / dq) * min(1.0, dfs_r / db_r), 1)
                    elif (cum_ph + cum_fsh) > 0 and cum_q > 0:
                        pod_st_pct = round(min(100.0, 100.0 * cum_eh / cum_q) * (cum_fsh / (cum_ph + cum_fsh)), 1)
                    elif cum_r > 0 and cum_q > 0:
                        pod_st_pct = round(min(100.0, 100.0 * cum_eh / cum_q) * min(1.0, cum_r / max(cum_r, b_restore)), 1)
                    else:
                        pod_st_pct = 0.0 if cum_q > 0 else None
                    tier_pods_raw.append({
                        "key": key,
                        "cache_hit_pct": pod_hit,
                        "storage_hit_pct": pod_st_pct,
                        "storage_read_gb": round(cum_r / 1e9, 2),
                        "storage_write_gb": round(cum_w / 1e9, 1),
                        "running": int(round(running)),
                    })
            load_by_host[host] = running_sum
            by_pod[host] = pods                # already in ip order (ips is sorted) -> stable pod-1..N numbering
            # Oct 14: live host-tier figures per pool over the last ~8 s (the Memory panel shows both node pools).
            win = self.kv_host_win.setdefault(host, deque())
            win.append((now, dh_sum, dq_sum, hd_ext_h, hd_ext_q, hd_br, hd_bo))
            while win and win[0][0] < now - 8.0:
                win.popleft()
            w_h, w_q, w_eh, w_eq, w_br, w_bo = (sum(x[i] for x in win) for i in range(1, 7))
            by_host[host] = round(min(100.0, 100.0 * (w_h + w_eh) / w_q), 1) if w_q > 0 else None
            w_span = max(1.0, now - win[0][0] + 1.0)
            cum = self.kv_host_cum.setdefault(host, [0.0, 0.0])
            cum[0] += hd_bo / 1e9
            cum[1] += hd_br / 1e9
            hbm_hit = round(min(100.0, 100.0 * w_h / w_q), 1) if w_q > 0 else None
            tier_by_host[host] = {
                "hbm_hit_pct": hbm_hit,
                "host_hit_pct": round(min(100.0 - (hbm_hit or 0.0), 100.0 * w_eh / w_q), 1) if w_q > 0 else None,
                "offload_gbps": round(w_bo / w_span / 1e9, 2),
                "restore_gbps": round(w_br / w_span / 1e9, 2),
                "hbm_used_pct": round(sum(h_hbm_vals) / len(h_hbm_vals), 1) if h_hbm_vals else None,
                "kv_cap_bytes": h_cap_bytes or None,
                "gb_offloaded_total": round(cum[0], 1),
                "gb_restored_total": round(cum[1], 1),
                "active": bool(cum[0] > 0 or cum[1] > 0 or w_eh > 0),     # an OffloadingConnector is doing work here
            }
            if host in self.kv_hosts:          # aggregate = the pool the A/B actually drives
                hits_d += dh_sum
                queries_d += dq_sum
                got = True
        if self.kv_boot_t is None and got:
            self.kv_boot_t = now
        self.kv_by_host = by_host
        self.load_by_host = load_by_host
        self.kv_by_pod = by_pod
        self.kv_tier_by_host = tier_by_host
        if kv_pod_ips:
            with _LOCK:
                self.kv_pod_ips = kv_pod_ips
        if got:
            if seen_fs:
                self.kv_storage_enabled = True
            max_w = max((p["storage_write_gb"] for p in tier_pods_raw), default=0.0)
            tier_pods = []
            for j, p in enumerate(tier_pods_raw):
                new_t = self.kv_pod_new_since.get(p["key"])
                is_new = bool(
                    (new_t is not None and (now - new_t) < 180.0)
                    or (p["storage_read_gb"] >= 0.5 and max_w >= 5.0 and p["storage_write_gb"] < 0.35 * max_w)
                )
                tier_pods.append({
                    "name": f"pod-{j + 1}",
                    "cache_hit_pct": p["cache_hit_pct"],
                    "storage_hit_pct": p["storage_hit_pct"],
                    "storage_read_gb": p["storage_read_gb"],
                    "storage_write_gb": p["storage_write_gb"],
                    "running": p["running"],
                    "is_new": is_new,
                })
            self.kv_scrape_win.append((
                now, hits_d, queries_d, ext_hits_d, ext_queries_d,
                db_r_sum, db_o_sum, dp_h_sum, dfs_h_sum, dfs_r_sum, dfs_w_sum,
            ))
            while self.kv_scrape_win and self.kv_scrape_win[0][0] < now - 8.0:
                self.kv_scrape_win.popleft()
            w_h = sum(x[1] for x in self.kv_scrape_win)
            w_q = sum(x[2] for x in self.kv_scrape_win)
            w_eh = sum(x[3] for x in self.kv_scrape_win)
            w_eq = sum(x[4] for x in self.kv_scrape_win)
            w_br = sum(x[5] for x in self.kv_scrape_win)
            w_bo = sum(x[6] for x in self.kv_scrape_win)
            w_ph = sum(x[7] for x in self.kv_scrape_win)
            w_fsh = sum(x[8] for x in self.kv_scrape_win)
            w_fsr = sum(x[9] for x in self.kv_scrape_win)
            w_fsw = sum(x[10] for x in self.kv_scrape_win)
            w_span = max(1.0, now - self.kv_scrape_win[0][0] + 1.0) if self.kv_scrape_win else 1.0
            self.kv_hbm_hit_pct = round(min(100.0, 100.0 * w_h / w_q), 1) if w_q > 0 else None
            self.kv_hit_pct = round(min(100.0, 100.0 * (w_h + w_eh) / w_q), 1) if w_q > 0 else None
            ext_tot = round(min(100.0 - (self.kv_hbm_hit_pct or 0.0), 100.0 * w_eh / w_q), 1) if w_q > 0 else (
                round(min(100.0, 100.0 * w_eh / w_eq), 1) if w_eq > 0 else 0.0
            )
            if self.kv_storage_enabled and (w_ph + w_fsh) > 0 and w_fsh > 0:
                st_pct = round(ext_tot * (w_fsh / (w_ph + w_fsh)), 1)
                self.kv_storage_hit_pct = st_pct
                self.kv_ext_hit_pct = round(max(0.0, ext_tot - st_pct), 1)
            elif self.kv_storage_enabled and w_br > 0 and w_fsr > 0:
                st_pct = round(ext_tot * min(1.0, w_fsr / w_br), 1)
                self.kv_storage_hit_pct = st_pct
                self.kv_ext_hit_pct = round(max(0.0, ext_tot - st_pct), 1)
            elif self.kv_storage_enabled and (cum_ph_sum + cum_fsh_sum) > 0 and cum_fsh_sum > 0:
                st_pct = round(ext_tot * (cum_fsh_sum / (cum_ph_sum + cum_fsh_sum)), 1)
                self.kv_storage_hit_pct = st_pct
                self.kv_ext_hit_pct = round(max(0.0, ext_tot - st_pct), 1)
            elif self.kv_storage_enabled and cum_fs_r_sum > 0:
                st_pct = round(ext_tot * min(1.0, cum_fs_r_sum / max(cum_fs_r_sum, self.kv_gb_restored * 1e9)), 1)
                self.kv_storage_hit_pct = st_pct
                self.kv_ext_hit_pct = round(max(0.0, ext_tot - st_pct), 1)
            else:
                self.kv_storage_hit_pct = 0.0 if self.kv_storage_enabled else None
                self.kv_ext_hit_pct = ext_tot
            inst_r = restore_bps_sum / 1e9
            inst_o = offload_bps_sum / 1e9
            inst_fs_r = fs_read_bps_sum / 1e9
            inst_fs_w = fs_write_bps_sum / 1e9
            self.kv_restore_gbps = 0.0 if self.kv_tier_mode == "without" else round(inst_r if inst_r > 0 else (w_br / w_span / 1e9), 2)
            self.kv_offload_gbps = 0.0 if self.kv_tier_mode == "without" else round(inst_o if inst_o > 0 else (w_bo / w_span / 1e9), 2)
            self.kv_storage_read_gbps = 0.0 if self.kv_tier_mode == "without" else round(inst_fs_r if inst_fs_r > 0 else (w_fsr / w_span / 1e9), 2)
            self.kv_storage_write_gbps = 0.0 if self.kv_tier_mode == "without" else round(inst_fs_w if inst_fs_w > 0 else (w_fsw / w_span / 1e9), 2)
            with _LOCK:
                self.kv_gb_storage_read = cum_fs_r_sum / 1e9
                self.kv_gb_storage_written = cum_fs_w_sum / 1e9
                self.kv_tier_pods = tier_pods
            if hbm_u_vals:
                self.kv_hbm_used_pct = round(sum(hbm_u_vals) / len(hbm_u_vals), 1)
            self.kv_restore_hist.append(self.kv_restore_gbps)

    # ------------------------------------------------------------------ aggregation
    # ------------------------------------------------------------------ storage replicas (Oct 14)
    def storage_pod_url(self, idx):
        """Chat URL of scale-out replica number idx (round-robin over the resolved pod IPs)."""
        with _LOCK:
            ips = list(self.st_pod_ips)
        if ips:
            ip, port = ips[idx % len(ips)]
            return f"http://{ip}:{port}/v1/chat/completions"
        if self.st_urls:
            return self.st_urls[idx % len(self.st_urls)]
        return None

    def storage_replica_count(self):
        with _LOCK:
            return len(self.st_pod_ips) or len(self.st_urls)

    def add_storage_sample(self, leg, ttft_ms, e2e_ms, out_tok, prompt_tok=0, turn=1, url=None):
        pod = urllib.parse.urlparse(url).netloc if url else None
        with _LOCK:
            self.st_samples.append((_now(), leg, int(turn), float(ttft_ms), float(e2e_ms), int(out_tok), int(prompt_tok), pod))

    def add_storage_error(self):
        with _LOCK:
            self.st_errors += 1

    def _scrape_storage(self):
        """Per-pod counters of the scale-out replicas: prefix hits, host restores and the fs tier's read/write bytes.

        vLLM does not count chunk hits for the fs tier, so the storage share of a replica's prefix reuse is the
        share of restored bytes that came from the fs tier (fs read bytes / CPU->GPU bytes) in the window.
        """
        import socket
        if not self.st_hosts:
            return
        now = _now()
        pod_ips = []
        rows = []
        for host in self.st_hosts:
            name, _, port = host.partition(":")
            port = port or "8000"
            try:
                ips = sorted({ai[4][0] for ai in socket.getaddrinfo(name, int(port), socket.AF_INET, socket.SOCK_STREAM)})
            except Exception:
                ips = []
            for ip in ips:
                try:
                    with urllib.request.urlopen(f"http://{ip}:{port}/metrics", timeout=0.9) as r:
                        txt = r.read().decode("utf-8", errors="replace")
                except Exception:
                    continue
                pod_ips.append((ip, port))
                key = f"{ip}:{port}"
                h = q = ext_h = b_restore = b_offload = fs_r = fs_w = running = 0.0
                hbm_u = None
                for line in txt.splitlines():
                    if line.startswith("vllm:prefix_cache_hits_total"):
                        h += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith("vllm:prefix_cache_queries_total"):
                        q += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith("vllm:external_prefix_cache_hits_total"):
                        ext_h += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith("vllm:num_requests_running"):
                        running += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith("vllm:kv_offload_total_bytes_total"):
                        v = float(line.rsplit(" ", 1)[-1])
                        if "CPU_to_GPU" in line:
                            b_restore += v
                        elif "GPU_to_CPU" in line:
                            b_offload += v
                    elif line.startswith("vllm:kv_offload_tiering_read_bytes_total") and 'tier="1:fs"' in line:
                        fs_r += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith("vllm:kv_offload_tiering_write_bytes_total") and 'tier="1:fs"' in line:
                        fs_w += float(line.rsplit(" ", 1)[-1])
                    elif line.startswith(("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")):
                        hbm_u = float(line.rsplit(" ", 1)[-1]) * 100.0
                rows.append((key, now, h, q, ext_h, b_restore, b_offload, fs_r, fs_w, running, hbm_u))
        if not rows:
            return
        if self.st_boot_t is None:
            self.st_boot_t = now
        # The control turns (leg "recompute") opt out of the tier, so vLLM counts their prompt tokens as queries
        # with no hit. Take them out of the denominator so the hit split describes the turns that may reuse KV.
        ctrl = {}
        with _LOCK:
            for (t, lg, _turn, _f, _e, _o, ptok, pod) in self.st_samples:
                if lg == "recompute" and t >= now - ST_WINDOW_S:
                    ctrl[pod] = ctrl.get(pod, 0.0) + ptok
        dh = dq = deh = dbr = dfr = dfw = dbo = 0.0
        pods_raw = []
        hbm_vals = []
        for (key, t, h, q, ext_h, b_restore, b_offload, fs_r, fs_w, running, hbm_u) in rows:
            prev = self.st_prev.get(key)
            self.st_prev[key] = (t, h, q, ext_h, b_restore, b_offload, fs_r, fs_w)
            if key not in self.st_seen and now - self.st_boot_t > 5.0:
                self.st_new_since[key] = now
            if prev and (q < prev[2] or fs_w < prev[7]):
                self.st_new_since[key] = now         # counters went backwards: the pod restarted
                self.st_base.pop(key, None)
                self.st_pod_win.pop(key, None)
            self.st_seen.add(key)
            base = self.st_base.get(key) or (0.0, 0.0, 0.0)
            win_hit = win_ext = None
            if prev and h >= prev[1] and q >= prev[2] and ext_h >= prev[3] and b_restore >= prev[4] and fs_r >= prev[6]:
                p_dh, p_dq, p_deh = h - prev[1], q - prev[2], ext_h - prev[3]
                p_dbr, p_dfr, p_dfw, p_dbo = b_restore - prev[4], fs_r - prev[6], fs_w - prev[7], b_offload - prev[5]
                dh += p_dh; dq += p_dq; deh += p_deh; dbr += p_dbr; dfr += p_dfr; dfw += p_dfw; dbo += p_dbo
                with _LOCK:
                    self.st_tok_restored += p_deh
                pw = self.st_pod_win.setdefault(key, deque())
                pw.append((now, p_dh, p_dq, p_deh))
                while pw and pw[0][0] < now - ST_WINDOW_S:
                    pw.popleft()
                pw_h = sum(x[1] + x[3] for x in pw)
                pw_q = max(pw_h, sum(x[2] for x in pw) - ctrl.get(key, 0.0))
                if pw_q > 0:
                    win_hit = round(min(100.0, 100.0 * pw_h / pw_q), 1)
                    win_ext = round(min(100.0, 100.0 * sum(x[3] for x in pw) / pw_q), 1)
            if hbm_u is not None:
                hbm_vals.append(hbm_u)
            pods_raw.append({
                "ip": key,
                "cache_hit_pct": win_hit,
                "ext_hit_pct": win_ext,
                "storage_read_gb": round(max(0.0, fs_r - base[1]) / 1e9, 2),
                "storage_write_gb": round(max(0.0, fs_w - base[2]) / 1e9, 1),
                "running": int(round(running)),
                "is_new": bool(key in self.st_new_since and (now - self.st_new_since[key]) < 180.0),
            })
        self.st_win.append((now, dh, dq, deh, dbr, dfr, dfw, dbo))
        while self.st_win and self.st_win[0][0] < now - ST_WINDOW_S:
            self.st_win.popleft()
        span = max(1.0, now - self.st_win[0][0] + 1.0)
        w_h = sum(x[1] for x in self.st_win); w_q = sum(x[2] for x in self.st_win); w_eh = sum(x[3] for x in self.st_win)
        w_q = max(w_h + w_eh, w_q - sum(ctrl.values()))
        w_br = sum(x[4] for x in self.st_win); w_fr = sum(x[5] for x in self.st_win); w_fw = sum(x[6] for x in self.st_win)
        w_bo = sum(x[7] for x in self.st_win)
        hbm_pct = round(min(100.0, 100.0 * w_h / w_q), 1) if w_q > 0 else None
        ext_pct = round(min(100.0 - (hbm_pct or 0.0), 100.0 * w_eh / w_q), 1) if w_q > 0 else None
        share = min(1.0, w_fr / w_br) if w_br > 0 else (1.0 if w_fr > 0 else 0.0)
        st_pct = round(ext_pct * share, 1) if ext_pct is not None else None
        host_pct = round(max(0.0, ext_pct - st_pct), 1) if ext_pct is not None else None
        pods = []
        for j, p in enumerate(sorted(pods_raw, key=lambda x: x["ip"])):
            pods.append(dict({k: v for k, v in p.items() if k != "ip"}, name=f"pod-{j + 1}"))
        cum_r = sum(p["storage_read_gb"] for p in pods)
        cum_w = sum(p["storage_write_gb"] for p in pods)
        with _LOCK:
            self.st_pod_ips = sorted(pod_ips)
            self.st_pods = pods
            self.st_live = {
                "hbm_hit_pct": hbm_pct,
                "host_hit_pct": host_pct,
                "storage_hit_pct": st_pct,
                "read_gbps": round(w_fr / span / 1e9, 2),
                "write_gbps": round(w_fw / span / 1e9, 2),
                "restore_gbps": round(w_br / span / 1e9, 2),
                "offload_gbps": round(w_bo / span / 1e9, 2),
                "hbm_used_pct": round(sum(hbm_vals) / len(hbm_vals), 1) if hbm_vals else None,
                "gb_read_total": round(cum_r, 2),
                "gb_written_total": round(cum_w, 1),
                "tokens_restored_total": int(self.st_tok_restored),
            }

    def _agg_storage(self, leg, now):
        lo = now - AGG_WINDOW_S
        ret, e2e, n, first, out = [], [], 0, None, 0
        for (t, lg, turn, f, e, o, _p, _ip) in self.st_samples:
            if t < lo or lg != leg:
                continue
            if first is None:
                first = t
            n += 1
            out += o
            if turn >= 2:
                ret.append(f)
                e2e.append(e)
        if n < 3 or first is None or not ret:
            return None
        span = max(1.0, min(AGG_WINDOW_S, now - first))
        return {
            "ttft_return_p50_ms": _pct(ret, 0.5),
            "ttft_return_p95_ms": _pct(ret, 0.95),
            "e2e_return_p50_ms": _pct(e2e, 0.5),
            "tok_s": int(round(out / span)),
            "req_s": round(n / span, 1),
            "n": n,
        }

    def storage_state(self, active=False):
        with _LOCK:
            return self._storage_state_locked(active)

    def _storage_state_locked(self, active=False):
        """compare.kv_tier.storage: the scale-out replicas (or, without them, the A/B pool's own fs tier).
        Caller holds _LOCK (kv_tier_state calls this from inside its own locked section)."""
        if not self.kv_storage_enabled:
            return None
        if self.kv_storage_replicas:
            live = dict(self.st_live)
            live["pods"] = [dict(p) for p in self.st_pods]
            rep = max(len(self.st_pods), len(self.st_urls), 1)
            legs = {k: (dict(v) if v else None) for k, v in self.st_measured.items()}
            return {
                "enabled": True,
                "source": "replicas",
                "replicas": rep,
                "hbm_cap_bytes": self.st_hbm_cap,
                "host_bytes": self.st_host_bytes,
                "storage_bytes": self.kv_storage_bytes,
                "active": bool(active and self.kv_tier_mode == "storage"),
                "errors": self.st_errors,
                "live": live,
                "legs": legs,
            }
        # the A/B pool itself has the fs tier: same shape, from the pool scrape
        return {
            "enabled": True,
            "source": "pool",
            "replicas": len(self.kv_tier_pods),
            "hbm_cap_bytes": self.kv_tier_hbm_cap,
            "host_bytes": self.kv_tier_host_bytes,
            "storage_bytes": self.kv_storage_bytes,
            "active": bool(active and self.kv_tier_mode != "without"),
            "errors": 0,
            "live": {
                "hbm_hit_pct": self.kv_hbm_hit_pct, "host_hit_pct": self.kv_ext_hit_pct, "storage_hit_pct": self.kv_storage_hit_pct,
                "read_gbps": self.kv_storage_read_gbps, "write_gbps": self.kv_storage_write_gbps,
                "restore_gbps": self.kv_restore_gbps, "offload_gbps": self.kv_offload_gbps,
                "hbm_used_pct": self.kv_hbm_used_pct,
                "gb_read_total": round(self.kv_gb_storage_read, 2), "gb_written_total": round(self.kv_gb_storage_written, 1),
                "tokens_restored_total": int(self.kv_tok_restored),
                "pods": [dict(p) for p in self.kv_tier_pods],
            },
            "legs": {"restore": None, "recompute": None},
        }

    def _agg(self, mode, now, fc_only=False):
        lo = now - AGG_WINDOW_S
        ttft, e2e, out, n = [], [], 0, 0
        first = None
        for (t, m, f, e, o, _p, fc) in self.samples:
            if t < lo or m != mode or (fc_only and not fc):
                continue
            if first is None:
                first = t
            n += 1
            ttft.append(f)
            e2e.append(e)
            out += o
        if n == 0:
            return None
        span = max(1.0, min(AGG_WINDOW_S, now - first)) if first is not None else AGG_WINDOW_S
        # B1 (Oct 7): right after a mode switch the new mode has only a handful of samples over a tiny span,
        # which made tok/s spike and dip. Publish nothing until the window is meaningfully filled; the UI
        # keeps the previous number (status "warming up") instead of drawing the artifact.
        if n < MIN_AGG_N or span < MIN_AGG_SPAN_S:
            return None
        return {
            "tok_s": int(round(out / span)),
            "req_s": round(n / span, 1),
            "ttft_p50_ms": _pct(ttft, 0.5), "ttft_p95_ms": _pct(ttft, 0.95),
            "e2e_p50_ms": _pct(e2e, 0.5), "e2e_p95_ms": _pct(e2e, 0.95),
            "n": n,
        }

    def _agg_kv(self, kv_mode, now):
        lo = now - AGG_WINDOW_S
        ret_ttft, all_ttft, out, ptok, n, first = [], [], 0, 0, 0, None
        for (t, km, turn, f, o, p) in self.kv_samples:
            if t < lo or km != kv_mode:
                continue
            if first is None:
                first = t
            n += 1
            all_ttft.append(f)
            if turn >= 2:
                ret_ttft.append(f)
            out += o
            ptok += p
        if n < MIN_AGG_N or first is None or (now - first) < MIN_AGG_SPAN_S:
            return None
        span = max(1.0, min(AGG_WINDOW_S, now - first))
        use_ttft = ret_ttft if len(ret_ttft) >= 3 else all_ttft
        return {
            "ttft_return_p50_ms": _pct(use_ttft, 0.5),
            "ttft_return_p95_ms": _pct(use_ttft, 0.95),
            "tok_s": int(round(out / span)),
            "req_s": round(n / span, 1),
            "prefix_tok_s": int(round(ptok / span)),
        }

    def _agg_loop(self):
        while True:
            time.sleep(1.0)
            now = _now()
            self._scrape_kv()
            self._scrape_storage()
            with _LOCK:
                lo = now - max(AGG_WINDOW_S, 30.0)
                while self.samples and self.samples[0][0] < lo:
                    self.samples.popleft()
                while self.kv_samples and self.kv_samples[0][0] < lo:
                    self.kv_samples.popleft()
                while self.st_samples and self.st_samples[0][0] < lo:
                    self.st_samples.popleft()
                if self.kv_storage_replicas:
                    for leg in ("restore", "recompute"):
                        cur_leg = self._agg_storage(leg, now)
                        if cur_leg is not None:
                            self.st_measured[leg] = dict(cur_leg, live=True)
                        elif self.st_measured[leg] is not None:
                            self.st_measured[leg] = dict(self.st_measured[leg], live=False)
                mode = self.get_mode()
                self._check_mode_switch_locked(mode, now)
                cur = self._agg(mode, now)
                if cur is not None:
                    cur = dict(cur, cache_hit_pct=self.kv_hit_pct, live=True, errors=self.errors[mode])
                    self.measured[mode] = cur
                other = "with" if mode == "without" else "without"
                if self.measured[other] is not None:
                    self.measured[other] = dict(self.measured[other], live=False)
                # Closing-summary column 3: the router measured while flow control is on.
                if mode == "with" and self.get_flow() == "with":
                    fc_cur = self._agg("with", now, fc_only=True)
                    if fc_cur is not None:
                        self.measured["with_fc"] = dict(fc_cur, cache_hit_pct=self.kv_hit_pct, live=True,
                                                        errors=self.errors["with"])
                elif self.measured["with_fc"] is not None:
                    self.measured["with_fc"] = dict(self.measured["with_fc"], live=False)
                if self.kv_tier_enabled:
                    km = self.kv_pool_mode()
                    kcur = self._agg_kv(km, now)
                    if kcur is not None:
                        self.kv_measured[km] = dict(
                            kcur, hbm_hit_pct=self.kv_hbm_hit_pct if self.kv_hbm_hit_pct is not None else self.kv_hit_pct,
                            host_hit_pct=0.0 if km == "without" else self.kv_ext_hit_pct,
                            storage_hit_pct=0.0 if km == "without" else self.kv_storage_hit_pct,
                            restore_gbps=0.0 if km == "without" else self.kv_restore_gbps,
                            offload_gbps=0.0 if km == "without" else self.kv_offload_gbps,
                            storage_read_gbps=0.0 if km == "without" else self.kv_storage_read_gbps,
                            storage_write_gbps=0.0 if km == "without" else self.kv_storage_write_gbps,
                            tokens_restored=0 if km == "without" else int(self.kv_tok_restored), live=True)
                    kother = "with" if km == "without" else "without"
                    if self.kv_measured[kother] is not None:
                        self.kv_measured[kother] = dict(self.kv_measured[kother], live=False)
                active = self.is_active()
                prev_pt = self.last_point if active else {}
                prev_tok = prev_pt.get("cluster_tok_s")
                if cur and active:
                    pt = {
                        "routing_mode": mode,
                        "cluster_tok_s": cur["tok_s"],
                        "ttft_p50_ms": cur["ttft_p50_ms"],
                        "ttft_p95_ms": cur["ttft_p95_ms"],
                        "e2e_p50_ms": cur["e2e_p50_ms"],
                        "e2e_p95_ms": cur["e2e_p95_ms"],
                        "cache_hit_pct": self.kv_hit_pct if self.kv_hit_pct is not None else prev_pt.get("cache_hit_pct"),
                    }
                elif active and prev_tok is not None:
                    pt = {
                        "routing_mode": mode,
                        "cluster_tok_s": prev_tok,
                        "ttft_p50_ms": prev_pt.get("ttft_p50_ms"),
                        "ttft_p95_ms": prev_pt.get("ttft_p95_ms"),
                        "e2e_p50_ms": prev_pt.get("e2e_p50_ms"),
                        "e2e_p95_ms": prev_pt.get("e2e_p95_ms"),
                        "cache_hit_pct": self.kv_hit_pct if self.kv_hit_pct is not None else prev_pt.get("cache_hit_pct"),
                    }
                else:
                    pt = {
                        "routing_mode": mode,
                        "cluster_tok_s": None,
                        "ttft_p50_ms": None,
                        "ttft_p95_ms": None,
                        "e2e_p50_ms": None,
                        "e2e_p95_ms": None,
                        "cache_hit_pct": None,
                    }
                self.last_point = pt
                self.history.append(dict(pt, t=_unix_ms()))

    # ------------------------------------------------------------------ public
    def kv_pool_mode(self):
        """The A/B pool's leg for the current tier selection: Tier 3 keeps the pool on its host-DRAM leg
        (the storage tier lives on the scale-out replicas), so 'storage' maps to 'with' here."""
        return "with" if self.kv_tier_mode == "storage" else self.kv_tier_mode

    def set_kv_mode(self, mode):
        if mode not in ("without", "with", "storage"):
            return False, "mode must be without, with or storage"
        if mode == "storage" and not self.kv_storage_enabled:
            return False, "no storage tier configured"
        now = _now()
        with _LOCK:
            if self.kv_tier_mode != mode:
                self.kv_scrape_win.clear()
                self.kv_pod_win.clear()
                self.st_win.clear()
                self.st_pod_win.clear()
                if self.get_mode() == "with":
                    cutoff = now - 3.0
                    self.samples = deque(s for s in self.samples if s[1] != "with" or s[0] >= cutoff)
            self.kv_tier_mode = mode
        return True, {"mode": self.kv_tier_mode, "enabled": self.kv_tier_enabled}

    def kv_tier_state(self, pool="gpu", active=False, accelerators=4, replicas=4):
        bpt = 46 * 1024
        ptok = 8100
        acc = max(1, int(accelerators or 1))
        rep = max(1, int(replicas or 1))
        cap = self.kv_tier_hbm_cap or (8 * 2 ** 30)
        host_b = self.kv_tier_host_bytes
        st_b = self.kv_storage_bytes
        with _LOCK:
            kcur = self.kv_measured.get(self.kv_pool_mode())
            hbm_u = self.kv_hbm_used_pct if active else 0.0
            host_u = min(100.0, self.kv_gb_offloaded * 1e9 / max(1, host_b * rep) * 100.0) if self.kv_tier_enabled else 0.0
            st_u = min(100.0, self.kv_gb_storage_written * 1e9 / max(1, st_b) * 100.0) if self.kv_storage_enabled else 0.0
            shbm = int(round((hbm_u / 100.0) * (cap * acc / bpt) / ptok))
            shost = int(round((max(0.0, host_u) / 100.0) * (host_b * rep / bpt) / ptok))
            sst = int(round((self.kv_gb_storage_written * 1e9 / bpt) / ptok)) if self.kv_storage_enabled else 0
            live = {
                "hbm_hit_pct": (self.kv_hbm_hit_pct if self.kv_hbm_hit_pct is not None else self.kv_hit_pct) if active else None,
                "host_hit_pct": (0.0 if self.kv_tier_mode == "without" else self.kv_ext_hit_pct) if active else None,
                "storage_hit_pct": (0.0 if self.kv_tier_mode == "without" else self.kv_storage_hit_pct) if active else None,
                "restore_gbps": (0.0 if self.kv_tier_mode == "without" else self.kv_restore_gbps) if active else 0.0,
                "offload_gbps": (0.0 if self.kv_tier_mode == "without" else self.kv_offload_gbps) if active else 0.0,
                "storage_read_gbps": (0.0 if self.kv_tier_mode == "without" else self.kv_storage_read_gbps) if active else 0.0,
                "storage_write_gbps": (0.0 if self.kv_tier_mode == "without" else self.kv_storage_write_gbps) if active else 0.0,
                "hbm_used_pct": hbm_u,
                "host_used_pct": round(max(0.0, host_u), 1),
                "host_full": bool(self.kv_tier_enabled and self.kv_gb_offloaded * 1e9 >= host_b * rep),
                "storage_used_pct": round(max(0.0, st_u), 1),
                "ttft_return_p50_ms": kcur["ttft_return_p50_ms"] if (kcur and active) else None,
                "ttft_return_p95_ms": kcur["ttft_return_p95_ms"] if (kcur and active) else None,
                "tok_s": kcur["tok_s"] if (kcur and active) else None,
                "req_s": kcur["req_s"] if (kcur and active) else None,
                "prefix_tok_s": kcur.get("prefix_tok_s") if (kcur and active) else None,
                "sessions_hbm": shbm,
                "sessions_host": shost,
                "sessions_storage": sst,
                "tokens_restored_total": int(self.kv_tok_restored),
                "gb_restored_total": round(self.kv_gb_restored, 1),
                "gb_offloaded_total": round(self.kv_gb_offloaded, 1),
                "gb_storage_read_total": round(self.kv_gb_storage_read, 2),
                "gb_storage_written_total": round(self.kv_gb_storage_written, 1),
                "pods": [dict(p) for p in self.kv_tier_pods],
                "restore_gbps_hist": [round(v, 2) for v in self.kv_restore_hist],
            }
            return {
                "enabled": self.kv_tier_enabled,
                "storage_enabled": self.kv_storage_enabled,
                "pool": pool,
                "mode": self.kv_tier_mode,
                "hbm_cap_bytes": self.kv_tier_hbm_cap,
                "host_bytes": host_b,
                "storage_bytes": st_b if self.kv_storage_enabled else None,
                "bytes_per_token": bpt,
                "prefix_tokens": ptok,
                "accelerators": acc,
                "replicas": max(rep, len(self.kv_tier_pods)),
                "live": live,
                "without": dict(self.kv_measured["without"]) if self.kv_measured["without"] else None,
                "with": dict(self.kv_measured["with"]) if self.kv_measured["with"] else None,
                "storage": self._storage_state_locked(active),
            }

    def reset(self):
        """Forget every measurement (Oct 8): called when the fleet has been suspended so the next wake starts
        from a blank stack (no stale tok/s, TTFT, KV %, deltas). Workers keep running; sessions roll over."""
        with _LOCK:
            for key, ep in self.kv_ext_prev.items():
                hp = self.kv_prev.get(key) or (0.0, 0.0)
                self.kv_fs_base[key] = (ep[7], ep[8], hp[0], hp[1], ep[1], ep[5], ep[6])
            self.kv_pod_new_since.clear()
            self.samples.clear()
            self.kv_samples.clear()
            self.history.clear()
            self.measured = {"without": None, "with": None, "with_fc": None}
            self.kv_measured = {"without": None, "with": None}
            self.errors = {"without": 0, "with": 0}
            self.kv_hit_pct = None
            self.kv_hbm_hit_pct = None
            self.kv_ext_hit_pct = None
            self.kv_storage_hit_pct = None
            self.kv_restore_gbps = 0.0
            self.kv_offload_gbps = 0.0
            self.kv_storage_read_gbps = 0.0
            self.kv_storage_write_gbps = 0.0
            self.kv_tok_restored = 0.0
            self.kv_gb_restored = 0.0
            self.kv_gb_offloaded = 0.0
            self.kv_gb_storage_read = 0.0
            self.kv_gb_storage_written = 0.0
            self.kv_host_win.clear()
            self.kv_pod_win.clear()
            self.kv_scrape_win.clear()
            self.kv_host_cum.clear()
            self.kv_tier_by_host = {}
            self.kv_tier_mode = "without"
            self.kv_ring_active = False
            self.kv_restore_hist = deque([0.0] * 60, maxlen=60)
            self.mode_since = 0.0
            self.last_point = {}
            # storage replicas: keep the counters' baseline so cumulative GB start from zero again
            for ip, p in self.st_prev.items():
                self.st_base[ip] = (p[3], p[6], p[7])
            self.st_new_since.clear()
            self.st_win.clear()
            self.st_pod_win.clear()
            self.st_samples.clear()
            self.st_measured = {"restore": None, "recompute": None}
            self.st_tok_restored = 0.0
            self.st_errors = 0
            self.st_live = {}
            self.st_pods = []

    def point(self):
        with _LOCK:
            return dict(self.last_point)

    def compare(self, flow_mode="without"):
        with _LOCK:
            return {
                "routing_mode": self.get_mode(),
                "flow_mode": flow_mode,
                "source": "live-ab",
                "targets": {"with": self.with_url, "without": list(self.without_urls), "workload": self.workload},
                "kv_by_host": dict(self.kv_by_host),
                "kv_by_pod": {h: [{k: x for k, x in p.items() if k not in ("ip", "port")} for p in v] for h, v in self.kv_by_pod.items()},   # no addresses in the public state
                "routing": {"without": self.measured["without"], "with": self.measured["with"],
                            "with_fc": self.measured["with_fc"]},
                "flow": {"without": None, "with": None},
            }

    # ------------------------------------------------------------------ per-pool view (Oct 9)
    def by_pool(self, ab_pool, agent_samples, chips, active=True, agents=None):
        """compare.by_pool for the live backend: {tpu: {...}, gpu: {...}} with the same fields the mock emits.

        The A/B pool's numbers are the measurement for the current routing mode (baseline = the other
        mode, i.e. "vs round-robin on this pool" at the close); since Oct 9 those samples come from the Hermes
        agents assigned to that pool (add_sample), not from synthetic sessions. The other pool carries the rest of
        the agents through the gateway: agent_samples is an iterable of (t_mono, ttft_ms, e2e_ms, completion_tokens)
        collected by the caller over the same window, so both cards show the same workload.
        KV hit per pool comes from the per-pod scrape; chip_util is the pool's
        requests in flight (vllm:num_requests_running summed over its pods) relative to LOAD_FULL_REQS, repeated per
        chip with a small deterministic spread so the glyphs read as one busy pool, not a random blink.
        chips = {"tpu": n, "gpu": n}. agents = {"tpu": n, "gpu": n} agent workers assigned to each pool: the split is
        the share of AGENTS (50/50 by default), not of completed requests, so a faster pool does not read as "getting
        more traffic" (Oct 9, user). Falls back to the request share when agents is not given.
        Returns None until something has been measured.
        """
        now = _now()
        lo = now - AGG_WINDOW_S
        other = "tpu" if ab_pool == "gpu" else "gpu"
        with _LOCK:
            mode = self.get_mode()
            cur = self.measured.get(mode)
            base = self.measured.get("without") if mode == "with" else None
            kv = dict(self.kv_by_host)
            load = dict(self.load_by_host)
            kv_pods = {h: [dict(p) for p in v] for h, v in self.kv_by_pod.items()}
            tier_by_host = {h: dict(v) for h, v in self.kv_tier_by_host.items()}
        # the agents' pool
        ttft, e2e, out, n, first = [], [], 0, 0, None
        for (t, f, e, o) in agent_samples:
            if t < lo:
                continue
            if first is None:
                first = t
            n += 1
            ttft.append(f)
            e2e.append(e)
            out += o
        span = max(1.0, min(AGG_WINDOW_S, now - first)) if first is not None else AGG_WINDOW_S
        agent_m = None          # (Oct 9 fix) was named `agents`, shadowing the agents={"tpu":n,"gpu":n} split argument
        if n >= MIN_AGG_N and span >= MIN_AGG_SPAN_S:
            agent_m = {"tok_s": int(round(out / span)), "req_s": round(n / span, 1), "ttft_p50_ms": _pct(ttft, 0.5),
                       "e2e_p50_ms": _pct(e2e, 0.5)}
        if cur is None and agent_m is None:
            return None

        def host_val(d, key):
            for h, v in d.items():
                if key in h.lower():
                    return v
            if key == ab_pool:
                for h in self.kv_hosts:
                    if h in d:
                        return d[h]
            return None

        def entry(key, m, baseline):
            # in-flight work on the pool: the 1-s gauge sample (vllm:num_requests_running) is spiky at ~10 req/s, so
            # take the larger of it and Little's law (req/s x E2E) over the window; this is what lights the glyphs.
            inflight = float(host_val(load, key) or 0.0)
            if m and m.get("req_s") and m.get("e2e_p50_ms"):
                inflight = max(inflight, float(m["req_s"]) * float(m["e2e_p50_ms"]) / 1000.0)
            u = min(1.0, inflight / float(LOAD_FULL_REQS)) if active else 0.0
            if active and m and m.get("req_s"):
                u = max(u, 0.3)      # traffic is flowing -> never fully dark
            k = max(1, int(chips.get(key) or 8))
            util = [round(max(0.0, min(1.0, u * (0.85 + 0.3 * ((c * 7) % 5) / 4.0))), 2) for c in range(k)] if u > 0 else [0.0] * k
            hit = host_val(kv, key)
            # per-pod KV cards (Oct 9): the pods behind this pool's headless Service, numbered pod-1..N in ip order
            pods = [{"name": "pod-%d" % (j + 1), "cache_hit_pct": p.get("cache_hit_pct"),      # no ip in the public state
                     "running": int(p.get("running") or 0)}
                    for j, p in enumerate(host_val(kv_pods, key) or [])]
            # Oct 14: this pool's host tier (both node pools can have one). Fill is cumulative bytes offloaded since
            # start over the pool size: vLLM exposes no host-pool occupancy, so once that passes 100 % the pool is
            # simply full and the connector evicts least-recently-used blocks (host_full).
            tier = host_val(tier_by_host, key)
            kv_tier = None
            if self.kv_tier_pools.get(key):
                kv_tier = dict(tier or {})
                kv_tier["enabled"] = True
                hb = self.kv_host_bytes_by_pool.get(key)
                reps = max(1, len(pods))
                written = float(kv_tier.get("gb_offloaded_total") or 0.0) * 1e9
                kv_tier["host_bytes"] = hb
                kv_tier["replicas"] = reps
                kv_tier["host_used_pct"] = round(min(100.0, 100.0 * written / (hb * reps)), 1) if hb else None
                kv_tier["host_full"] = bool(hb) and written >= hb * reps
            cap_b = (tier.get("kv_cap_bytes") if tier else None) or (int(self.kv_tier_hbm_cap) if (key == ab_pool and self.kv_tier_enabled and self.kv_tier_hbm_cap) else None)
            return {
                "up": True,
                "req_s": m["req_s"] if m else 0.0,
                "tok_s": m["tok_s"] if (m and active) else None,
                "ttft_p50_ms": m["ttft_p50_ms"] if (m and active) else None,
                "cache_hit_pct": hit if (hit is not None and hit > 0) else None,
                "kv_cap_bytes": cap_b,
                "chip_util": util,
                "pods": pods,
                "kv_tier": kv_tier,
                "baseline": baseline,
                "split_pct": None,
            }
        out_d = {
            ab_pool: entry(ab_pool, cur, ({"tok_s": base["tok_s"], "ttft_p50_ms": base["ttft_p50_ms"]} if base else None)),
            other: entry(other, agent_m, None),
        }
        if agents and sum(int(agents.get(k) or 0) for k in out_d) > 0:
            tot = float(sum(int(agents.get(k) or 0) for k in out_d))
            for k, v in out_d.items():
                v["split_pct"] = round(100.0 * int(agents.get(k) or 0) / tot, 1) if active else None
        else:
            tot = sum((v["req_s"] or 0.0) for v in out_d.values())
            for v in out_d.values():
                v["split_pct"] = round(100.0 * (v["req_s"] or 0.0) / tot, 1) if tot > 0 else None
        return out_d
