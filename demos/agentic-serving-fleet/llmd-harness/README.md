# llm-d Deployment & Benchmark Harness

Deployment and benchmark scripts for the heterogeneous `Gemma-4-26B-A4B` serving stack (`vllm-torchtpu` on TPUs and `vLLM` on NVIDIA GPUs) with `llm-d` routing, priority flow control, and GKE Agent Substrate.

## Quick Start

```bash
git clone https://github.com/llm-d/llm-d.git "${HOME}/llm-d"
export LLMD_REPO="${HOME}/llm-d"
export PROJECT_ID="<your-gcp-project-id>"
export CLUSTER_NAME="llmd-demo-cluster"
export HF_TOKEN="<your-huggingface-token>"
cd llmd-harness
./run_all.sh
```

### Stages

| Stage | Script | Description |
| :---: | :--- | :--- |
| **00** | [`scripts/00_preflight.sh`](scripts/00_preflight.sh) | Checks CLI tools, `HF_TOKEN`, GKE cluster connectivity, and TPU capacity |
| **50** | [`scripts/50_prepare_cluster.sh`](scripts/50_prepare_cluster.sh) | Ensures TPU, GPU, and Agent Substrate CPU node pools exist |
| **60** | [`scripts/60_deploy_llmd_heterogeneous.sh`](scripts/60_deploy_llmd_heterogeneous.sh) | Deploys TPU and GPU model server pools, `llm-d` EPPs, `InferenceObjective`s, and the gateway |
| **62** | [`scripts/62_deploy_kv_storage_replicas.sh`](scripts/62_deploy_kv_storage_replicas.sh) | Optional (`ENABLE_KV_STORAGE_REPLICAS=1`): deploys scale-out GPU replicas with host-DRAM and shared-storage KV tiers |
| **65** | [`scripts/65_deploy_substrate_and_agents.sh`](scripts/65_deploy_substrate_and_agents.sh) | Deploys GKE Agent Substrate `WorkerPool` + `ActorTemplate` (`hermes-agent-actor`) |
| **70** | [`scripts/70_verify_routing.sh`](scripts/70_verify_routing.sh) | Verifies `multi-turn` and `one-shot` routing through `llm-d-gateway` |
| **80** | [`scripts/80_run_benchmark.sh`](scripts/80_run_benchmark.sh) | Runs the 1,000-agent benchmark across both pools |
| **85** | [`scripts/85_run_dashboard.sh`](scripts/85_run_dashboard.sh) | Launches the local dashboard (`http://localhost:8085`) |
| **86** | [`scripts/86_deploy_dashboard_image.sh`](scripts/86_deploy_dashboard_image.sh) | Builds and deploys the in-cluster `llmd-live-dashboard` |
| **90** | [`scripts/90_teardown.sh`](scripts/90_teardown.sh) | Deletes the `llm-d-serving` and `llm-d-gpu` namespaces |

## Configuration ([`config/env.sh`](config/env.sh))

- `MODEL_ID`: Defaults to `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`.
- `TPU_NUM_SLICES` / `TPU_REPLICAS`: Number of `tpu7x-standard-4t` (`2x2x1`) slices and replicas.
- `GPU_MACHINE_TYPE`: GPU node machine type (`a4x-highgpu-4g` / `g4-standard-96`).
- `NUM_HERMES_AGENTS`: Total Hermes agents (default `1000`).

---

## KV Cache Tiering on the GPU Pool (`ENABLE_KV_TIERING=1`)

Opt-in third tier for the GPU pool: KV blocks evicted from HBM go to a host-DRAM pool and are
restored on the agent's next turn instead of being re-prefilled. It uses vLLM's native
`OffloadingConnector`, so it is a model-server flag change only; the llm-d EPPs, the gateway, and the
CRDs are untouched. Validated with `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` on `vllm/vllm-openai`
0.31.0 (B200, TP=2). Measured numbers, the test scripts, and the problems hit along the way are in
the PR that added this section.

### Enable

```bash
export ENABLE_KV_TIERING=1
export KV_TIER_CPU_BYTES=68719476736    # 64 GiB host pool per replica (default)
export KV_TIER_HBM_CAP_BYTES=8589934592 # optional, demo only: cap HBM KV at 8 GiB per GPU so the host tier is exercised
./scripts/60_deploy_llmd_heterogeneous.sh
```

This renders [`gpu-pool-kvtier.yaml.tmpl`](manifests/modelserver/gpu-vllm/gpu-pool-kvtier.yaml.tmpl)
instead of `gpu-pool.yaml.tmpl`. The differences are marked `kv-tiering` in the template:
`--no-disable-hybrid-kv-cache-manager`, the `--kv-transfer-config`, an optional
`--kv-cache-memory-bytes`, `PYTHONHASHSEED`, a pod memory request, and a `/dev/shm` size limit.

Pin the image for reproducible deployments: `GPU_SERVING_IMAGE=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`
is the digest that was tested (`:latest` on 2026-10-09).

### Live toggle for the dashboard

The tier can be switched off **per request**, so one deployment serves both legs of a comparison:

```json
{"model": "...", "messages": [...], "stream": true,
 "kv_transfer_params": {"max_load_tokens": 0, "max_offload_tokens": 0}}
```

Requests with that field recompute every turn and offload nothing; requests without it get the tier.
Gauges, from the pod's `/metrics` (same scrape as the existing KV hit gauge):

| What | Metric |
|---|---|
| HBM prefix hit % | `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` (already scraped) |
| Host-tier hit % | `vllm:external_prefix_cache_hits_total / vllm:external_prefix_cache_queries_total` |
| Bytes restored / offloaded | `vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` and `{transfer_type="GPU_to_CPU"}` |

### Measured (B200 pair, TP=2, HBM capped at 8 GiB/GPU, 64 GiB host pool)

[`scripts/88_kv_tiering_bench.py`](scripts/88_kv_tiering_bench.py): 128 agent sessions, each with its
own ~8.1k-token prefix, 5 turns, 32 concurrent, one turn per session per round so every prefix leaves
HBM between its turns. Legs `notier` / `tier` / `notier` on the same pods; the two `notier` legs agree
within 1% ([`results/kv-tiering-gemma4-b200-s128-c32-20261009.json`](results/kv-tiering-gemma4-b200-s128-c32-20261009.json)).

| | tier off | tier on |
| :--- | :---: | :---: |
| returning-turn TTFT p50 / p95 | 1,059 / 1,804 ms | **303 / 540 ms** |
| throughput | 16.5 req/s · 345 tok/s | **44.6 req/s · 931 tok/s** |
| 640 requests, wall | 38.9 s | **14.3 s** |
| cold-turn TTFT p50 (control) | 1,041 ms | 485 ms (less queueing) |
| prefix tokens restored from host | 0 | 4.67M of 5.25M |

Single-request restores land at HBM-hit speed (1.76k tokens: 136 ms host hit vs 165 ms recompute vs
129 ms HBM hit; 14k tokens: 173 vs 248 vs 161 ms), so the gain is in prefill work avoided under
concurrency, not in one request's latency.

```bash
kubectl -n llm-d-gpu port-forward svc/llmd-gpu-service 18003:8000 &
python3 scripts/88_kv_tiering_bench.py --url http://127.0.0.1:18003 \
  --model "${SERVED_MODEL_NAME}" --sessions 128 --rounds 5 --concurrency 32 --paras 140 \
  --legs notier,tier,notier --out results/kv-tiering-$(date +%Y%m%d-%H%M%S).json
```

### Gemma 4 specifics (so nobody re-discovers them)

- **Keep the hybrid KV cache manager on.** Gemma 4's full-attention layers (head_dim 512) and
  sliding-window layers (head_dim 256) cannot share one KV spec; with a connector configured vLLM
  auto-disables the hybrid manager unless told otherwise, and the server fails to build the cache.
  `--no-disable-hybrid-kv-cache-manager` is required (the plain pool uses the opposite flag).
- **Do not set `block_size` in `kv_connector_extra_config`.** Under the hybrid manager Gemma 4 has six
  KV groups with two GPU block sizes (64 tokens for sliding-window groups, 128 for full attention),
  and the connector asserts on that knob. The llm-d tiered-prefix-cache recipe sets `block_size: 256`
  because it was written for Qwen3-32B. Use `blocks_per_chunk` if larger offload chunks are wanted.
- **Size the host pool at 3x or more of the replica's total HBM KV budget.** In HBM the sliding-window
  layers keep only the 1024-token window, so a ~2k-token agent prefix is compact (~90 MB per GPU);
  its host copy is ~240 MB (all sliding-window tokens are stored for prompts near the window; ~46
  KB/token at 8k). A host pool the size of HBM evicts prefixes before HBM does and never hits.
- **The host pool lives in `/dev/shm`.** `GPU_DSHM_SIZE` must exceed `KV_TIER_CPU_BYTES`, and
  `GPU_POD_MEMORY` must cover pool + weights + runtime.
- **A filesystem tier is not worth it on this hardware.** Measured with vLLM's `TieringOffloadingSpec`
  (`secondary_tiers: [{type: fs}]`): the fs tier writes one file and does one `exists()` per 64-token
  chunk per KV group per rank (six groups for Gemma 4). On a remote Managed Lustre instance that cost
  +1.6 s on every miss and 7.4 s on a hit against a 0.17 s recompute (`blocks_per_chunk: 4` brought it
  to +0.24 s / 2.1 s, still a loss). On node-local SSD the miss penalty disappears but a hit is still
  slightly slower than recomputing (195 vs 167 ms at 1.76k; 274 vs 193 ms at 8k) because a B200 pair
  prefills this 4B-active MoE at ~55k tok/s. Keep it to HBM + DRAM for reproducible deployments.
- **SGLang HiCache was validated too** (v0.5.22, L2 host pool + `file` backend): works on Gemma 4
  (`UnifiedRadixCache hybrid_swa=True hicache_attached=True`), but it has no text-only load for Gemma 4
  on that release (`--language-model-only` is gated to another architecture, `--language-only` is
  encoder disaggregation), it stores every sliding-window token on host (102 KB/token/rank, ~3x worse
  host capacity than vLLM for the same bytes), and it has no per-request switch, so a with/without
  comparison needs two deployments. vLLM native is the lighter fit for this harness.

---

## Shared-storage KV tier (`ENABLE_KV_STORAGE_TIER=1`)

Third tier below host DRAM: evicted KV blocks are written through to a shared filesystem
(Google Cloud Managed Lustre) and **any replica** can restore them. The host tier survives an
eviction; the storage tier survives the replica. A new pod on scale-out, or a restarted pod,
serves returning agents from storage instead of re-prefilling all of them at once.
Requires `ENABLE_KV_TIERING=1` and an RWX PVC (see
[`manifests/storage/README.md`](manifests/storage/README.md)).

### Enable

```bash
export ENABLE_KV_TIERING=1 ENABLE_KV_STORAGE_TIER=1
export KV_TIER_HBM_CAP_BYTES=8589934592          # 8 GiB HBM cap per GPU so 24 sessions evict to DRAM/Lustre
export KV_STORAGE_PVC=llm-d-kv-cache-storage      # RWX PVC in ${GPU_NAMESPACE} (default)
export KV_STORAGE_BLOCKS_PER_CHUNK=16              # GPU blocks per file (default); do not go smaller on a network fs
./scripts/60_deploy_llmd_heterogeneous.sh
```

The rendered Deployment differs from the host-DRAM-only one in the `--kv-transfer-config` JSON
(`TieringOffloadingSpec` with an `fs` secondary tier) and a volume plus mount at `KV_STORAGE_MOUNT_PATH`.
With the switch off it renders exactly as before. The per-request toggle and the metrics are the same as
for the host tier; the storage share of restores is in `vllm:kv_offload_tiering_chunk_hits_total{tier="1:fs"}`
and `vllm:kv_offload_tiering_read_bytes_total{tier="1:fs"}`.

### Measured (GB200 TP=1, same-region Lustre)

[`scripts/89_kv_storage_crossreplica.py`](scripts/89_kv_storage_crossreplica.py): replica A serves 24
agent sessions with ~9.5k-token prefixes; replica B is a fresh pod; the same 24 sessions then go to B,
one at a time and all at once, with restore allowed and with `max_load_tokens: 0` (recompute). Lustre
and GPUs in different zones of one region.
[`results/kv-storage-crossreplica-gb200-lustre-same-region-20261013.json`](results/kv-storage-crossreplica-gb200-lustre-same-region-20261013.json).

| Cold replica B | recompute | restore from storage |
| :--- | :---: | :---: |
| one request at a time, TTFT p50 / p95 | 219 / 266 ms | 380 / 430 ms |
| 24 at once (scale-out burst), TTFT p50 / p95, wall | 1,177 / 1,978 ms, 2.2 s | **1,214 / 1,794 ms, 1.9 s** |
| prefix tokens from storage | 0 | 196,608 of 222,378 (9.06 GB) |
| warm replica A while the tier is on | 594 ms p50 cold prefill | 578–599 ms (unchanged) |

What to claim: under a scale-out burst a cold replica serves returning sessions **at the same TTFT with
zero recompute** (24 × 9.5k tokens of prefill avoided here), and the storage tier costs the warm replicas
nothing. What not to claim: a latency win over recompute; one request at a time, restore is 1.7x slower.
On Gemma 4 the KV bytes per token are large relative to the prefill FLOPs of a 4B-active MoE, so restore
and recompute land in the same place on a GB200.

On `B200 TP=2`, the same test against a **cross-region** Lustre is 7x slower than recompute with 1k-token chunks
([`...cross-region-20261012.json`](results/kv-storage-crossreplica-b200-lustre-cross-region-20261012.json):
1.5 s / 6.9 s seq / burst p50) and 80–100x slower with the 64-token default
([`...cross-region-default-chunk-20261012.json`](results/kv-storage-crossreplica-b200-lustre-cross-region-default-chunk-20261012.json):
16.5 s / 121 s), and it slows the warm replica's own prefills to 2–49 s. Keep the instance in the region
and the chunks large.

### Scale-out and restart runbook

Prerequisite: the pool is deployed with both switches on and a free GPU exists.

```bash
export GPU_NAMESPACE=llm-d-gpu
# 1. Agents have been running for a while, so the pool has written KV through to storage.
kubectl -n ${GPU_NAMESPACE} exec deploy/llmd-gpu-decode -- \
  sh -c 'curl -s localhost:8000/metrics | grep -E "kv_offload_tiering_(write_bytes|chunk_hits)_total"'

# 2. Scale-out: a fifth replica comes up cold.
kubectl -n ${GPU_NAMESPACE} scale deploy/llmd-gpu-decode --replicas=5
kubectl -n ${GPU_NAMESPACE} get pods -l app=llmd-gpu-decode -w

# 3. Watch the new pod: returning agents routed to it hit storage, not recompute.
NEW=$(kubectl -n ${GPU_NAMESPACE} get pods -l app=llmd-gpu-decode --sort-by=.metadata.creationTimestamp -o name | tail -1)
watch -n2 "kubectl -n ${GPU_NAMESPACE} exec ${NEW#pod/} -- sh -c 'curl -s localhost:8000/metrics' \
  | grep -E 'external_prefix_cache_(hits|queries)_total|tiering_read_bytes_total|tiering_chunk_hits_total'"

# 4. Restart variant: delete one warm pod; its replacement serves the same sessions from storage.
kubectl -n ${GPU_NAMESPACE} delete pod $(kubectl -n ${GPU_NAMESPACE} get pods -l app=llmd-gpu-decode -o name | head -1)

# 5. Control, same pods: a dashboard leg (or curl) with
#    "kv_transfer_params": {"max_load_tokens": 0, "max_offload_tokens": 0}  -> the new pod recomputes.
```

### Scale-out replicas for the dashboard's Tier 3 panel (`ENABLE_KV_STORAGE_REPLICAS=1`)

The live GPU pool keeps the host-DRAM tier only. With the `fs` tier on the pool itself, vLLM 0.31 writes every
chunk that reaches the host pool through to storage and looks storage up on every cold turn; under the agents'
~80 req/s of new prefixes that was 0.8 GB/s of writes with no capacity bound, and TTFT went from 102 to 210 ms
(measured Oct 14 on the GB200 pool, rolled back after 11 minutes). The storage tier pays off for a bounded set
of returning sessions, so the dashboard shows it on two separate replicas instead:

- [`scripts/62_deploy_kv_storage_replicas.sh`](scripts/62_deploy_kv_storage_replicas.sh) deploys
  `llmd-gpu-storage` ([`gpu-storage-replicas.yaml.tmpl`](manifests/modelserver/gpu-vllm/gpu-storage-replicas.yaml.tmpl)):
  `KV_STORAGE_REPLICAS=2` copies of the GPU model server (TP=1, one GPU each), same image and model, with a 4 GiB
  HBM cap and a 4 GiB host pool (`KV_STORAGE_REPLICA_*`) and the `fs` tier under `KV_STORAGE_REPLICA_ROOT_DIR`.
  They carry a different `app` label, so the InferencePool, the router and the live A/B never see them.
- The dashboard (`86_deploy_dashboard_image.sh` with `ENABLE_KV_STORAGE_REPLICAS=1`) drives a fixed ring of
  `KV_STORAGE_WORKERS x KV_STORAGE_RING` sessions across them. Each turn of a session goes to the other
  replica, so a returning turn lands on a replica that never computed it; with the small HBM cap and host pool
  the session is also gone from the replica that did, and the prefix comes back from Lustre. Every fourth turn
  opts out (`kv_transfer_params: {max_load_tokens: 0, max_offload_tokens: 0}`) and recomputes, as the control.
- The panel ("Tier 3 · Lustre" in the Memory view) shows the prefix reuse split (HBM / host / Lustre), the Lustre
  read and write rates, the prefill tokens avoided, per-replica chips, and restore-from-Lustre against recompute
  TTFT. The same numbers are in `/api/state` under `llmd.compare.kv_tier.storage`.
- Tier 3 is opt-in on the page. Until a presenter clicks the Tier 3 button (or opens the page with `?tier3=1`),
  the page is the two-tier page: two pool boxes, the Tier 3 button dashed, ↑/↓ stepping between Tier 1 and 2.
  Selecting Tier 3 adds the "Scale-out replicas" box; Node pool B, the router and the *With llm-d* numbers
  do not change, because the replicas sit outside the InferencePool and the ring never goes through the gateway.
  Reloading the page returns to the two-tier flow.

```bash
export ENABLE_KV_TIERING=1 ENABLE_KV_STORAGE_REPLICAS=1    # the PVC from manifests/storage/README.md must be Bound
./scripts/62_deploy_kv_storage_replicas.sh                  # needs two free GPUs on the GPU node pool
./scripts/86_deploy_dashboard_image.sh                      # AB_KV_STORAGE_HOSTS is set from the headless Service
./scripts/62_deploy_kv_storage_replicas.sh --down           # scales the replicas to 0; the dashboard disables Tier 3
```

Measured with the defaults (GB200 TP=1, same-region Lustre, 30 sessions with ~9.5k-token prefixes): at steady
state all returning turns are restored from Lustre at 1.0–1.4 GB/s of reads, restore TTFT p50 / p95 is about
415 / 530 ms against 245 / 300 ms for recompute (see the table above for why), and writes stop once the session
set is on storage. Each dashboard process writes its own session set once, about 13 GB; delete
`${KV_STORAGE_MOUNT_PATH}/kv-cache/vllm-scaleout` from any pod with the mount if it grows.

---

## KV Cache Tiering on the TPU v7x Pool (`ENABLE_KV_OFFLOAD=1` + `ENABLE_KV_STORAGE_TIER=1`)

Both L2 (Host DRAM) and L3 (Shared Storage / Managed Lustre) KV cache offloading also run on the
TPU v7x Ironwood pool (`vllm-torchtpu`) with `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`.
When `ENABLE_KV_OFFLOAD=1` is set, [`scripts/60_deploy_llmd_heterogeneous.sh`](scripts/60_deploy_llmd_heterogeneous.sh)
mounts [`manifests/modelserver/tpu-torchtpu/quant_patch/cpu_tpu.py`](manifests/modelserver/tpu-torchtpu/quant_patch/cpu_tpu.py)
into the `torchtpu-quant-patch` ConfigMap and configures `TPUCPUOffloadingSpec` (`TieringOffloadingSpec`
subclass supporting both L2 host DRAM and L3 `secondary_tiers: [{"type": "fs", ...}]`).

### Enable on TPU v7x

```bash
export ENABLE_KV_OFFLOAD=1                        # L2 Host DRAM tier on TPU v7x
export ENABLE_KV_STORAGE_TIER=1                   # optional: add L3 Shared Lustre tier below L2 DRAM
export KV_TIER_CPU_BYTES=68719476736              # 64 GiB host DRAM pool per engine
export KV_TIER_HBM_CAP_BYTES=838860800            # optional demo cap to force L1 HBM -> L2 DRAM -> L3 Lustre eviction
export KV_STORAGE_PVC=llm-d-kv-cache-storage      # RWX PVC in ${NAMESPACE}
./scripts/60_deploy_llmd_heterogeneous.sh
```

### Measured on TPU v7x (`tpu7x-standard-4t`, TP=2, FP8 KV, 256-token blocks, Managed Lustre)

Validated with a 6,588-token (25 full blocks, 750 MiB KV) prompt across L1 TPU HBM, L2 Host DRAM,
L3 Managed Lustre (both in-process eviction and cold-instance restart hydration), and per-request
recompute (`kv_transfer_params: {"max_load_tokens": 0, "max_offload_tokens": 0}`):

| Step | Tier Hit | TTFT | `gpu_prefix_hits` | `ext_prefix_hits` | `l2_chunk_hits` (`0:primary`) | `l3_chunk_hits` (`1:fs`) | Transfer Bytes | Greedy Output |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Cold prefill (`P1`, 6,588 tok) | Miss (offload L1 → L2 → L3) | 2,630.1 ms | +0 | +0 | +0 | +0 | 750.0 MiB D2H + 750.0 MiB L3 write | `DOC_ALPHA_01` |
| 2. Repeat `P1` in HBM | **L1 TPU HBM** | 520.4 ms | +6,400 | +0 | +0 | +0 | 0 B | `DOC_ALPHA_01` |
| 3. Evict from L1 (`P2..P3`), repeat `P1` | **L2 Host DRAM** | **357.4 ms** | +0 | +6,400 | **+25** | +0 | 750.0 MiB H2D (`0` L3 read) | `DOC_ALPHA_01` |
| 4. Evict from L1+L2 (`P2..P5`), repeat `P1` | **L3 Shared Lustre** | **344.8 ms** | +0 | +6,400 | +0 | **+25** | 750.0 MiB L3 read + 750.0 MiB H2D | `DOC_ALPHA_01` |
| 5. Cold-instance restart (`/dev/shm` wiped), repeat `P1` | **L3 Cold Hydrate** | **505.7 ms** | +0 | +6,400 | +0 | **+25** | 750.0 MiB L3 read + 750.0 MiB H2D | `DOC_ALPHA_01` |
| 6. Per-request toggle (`max_load_tokens: 0`) | **Recompute** | 148.2 ms | +0 | +0 | +0 | +0 | 0 B | `DOC_ALPHA_01` |
