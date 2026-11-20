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

"""TPU <-> CPU and Multi-Tier (L2 Host DRAM + L3 Shared Storage) KV cache offloading handler.

Supports both:
- L2-only CPU offloading (`TPUCPUOffloadingSpec` with no `secondary_tiers`)
- L2 + L3 tiered offloading (`TPUCPUOffloadingSpec` or `TPUTieringOffloadingSpec`
  with `secondary_tiers: [{"type": "fs", "root_dir": "/mnt/lustre/..."}]`)
- Hybrid models with `UniformTypeKVCacheSpecs` (such as Gemma 4 26B-A4B where
  sliding-window and full-attention layers have different head counts/dims
  that pad to the same Pallas page_size_bytes).
"""
from __future__ import annotations

import os
import queue as _queue
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch_tpu._internal.sync import synchronize as _tpu_sync
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _get_mmap_instance_id(vllm_config: VllmConfig) -> str:
    """Return a DP-rank-scoped instance_id for SharedOffloadRegion."""
    world_size = int(getattr(vllm_config.parallel_config, "world_size", 1) or 1)
    dp_rank = int(
        getattr(vllm_config.parallel_config, "data_parallel_rank", 0) or 0
    )
    if dp_rank == 0 and "RANK" in os.environ and world_size > 0:
        try:
            dp_rank = int(os.environ["RANK"]) // world_size
        except ValueError:
            dp_rank = 0
    return f"{vllm_config.instance_id}_dp{dp_rank}"


def _install_fs_io_direct_fix() -> None:
    """Disable O_DIRECT on pure-Python memoryview slices and use PID/TID-unique tmp files."""
    try:
        from vllm.v1.kv_offload.tiering.fs import io as fs_io
    except ImportError:
        return
    if getattr(fs_io, "_tpu_fs_io_patched", False):
        return

    fs_io.O_DIRECT = 0

    def store_block_safe(
        dest_path: str,
        buffer: memoryview,
        offset: int,
        block_size: int,
    ) -> bool:
        if os.path.exists(dest_path):
            return True
        tmp_path = f"{dest_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        fd = -1
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            fd = os.open(
                tmp_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC,
                0o644,
            )
            slice_view = buffer.cast("B")[offset : offset + block_size]
            written = 0
            while written < block_size:
                n = os.write(fd, slice_view[written:])
                if n <= 0:
                    raise OSError(
                        f"os.write returned {n} ({written}/{block_size} bytes)"
                    )
                written += n
            os.close(fd)
            fd = -1
            os.replace(tmp_path, dest_path)
            return True
        except Exception as e:
            if os.path.exists(dest_path):
                return True
            logger.error("Failed to store block to %s: %s", dest_path, e)
            return False
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    fs_io.store_block = store_block_safe
    try:
        from vllm.v1.kv_offload.tiering.fs import manager as fs_mgr

        fs_mgr.store_block = store_block_safe
    except ImportError:
        pass
    try:
        from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

        if "fs" not in SecondaryTierFactory._registry:
            SecondaryTierFactory.register_tier(
                "fs",
                "vllm.v1.kv_offload.tiering.fs.manager",
                "FileSystemTierManager",
            )
    except Exception:
        pass
    fs_io._tpu_fs_io_patched = True  # type: ignore[attr-defined]


def _install_tiering_metrics_hooks() -> None:
    """Expose vllm:kv_offload_tiering_{read_bytes,write_bytes,chunk_hits}_total and TPU worker transfer stats."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
            metrics as offload_metrics,
        )
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager
    except ImportError:
        return

    if not getattr(offload_metrics.OffloadPromMetrics, "_tpu_tiering_patched", False):
        orig_prom_init = offload_metrics.OffloadPromMetrics.__init__
        orig_prom_observe = offload_metrics.OffloadPromMetrics.observe

        def prom_init_with_tiering(
            self,
            vllm_config: VllmConfig,
            metric_types: dict,
            labelnames: list[str],
            per_engine_labelvalues: dict[int, list[object]],
        ) -> None:
            orig_prom_init(
                self, vllm_config, metric_types, labelnames, per_engine_labelvalues
            )
            self._counter_tiering_read_bytes = self._counter_cls(
                name="vllm:kv_offload_tiering_read_bytes",
                documentation="Bytes read from secondary tiers into CPU primary tier",
                labelnames=labelnames + ["tier"],
            )
            self._counter_tiering_write_bytes = self._counter_cls(
                name="vllm:kv_offload_tiering_write_bytes",
                documentation="Bytes written from CPU primary tier to secondary tiers",
                labelnames=labelnames + ["tier"],
            )
            self._counter_tiering_chunk_hits = self._counter_cls(
                name="vllm:kv_offload_tiering_chunk_hits",
                documentation="Chunk lookup hits on secondary tiers",
                labelnames=labelnames + ["tier"],
            )
            self._tiering_metric_cache: dict[tuple[int, str], tuple[Any, Any, Any]] = {}

        def prom_observe_with_tiering(
            self, transfer_stats_data: dict[str, Any], engine_idx: int = 0
        ) -> None:
            if not transfer_stats_data:
                return
            tiering_ops = transfer_stats_data.get("__tiering__")
            filtered_data = (
                {k: v for k, v in transfer_stats_data.items() if k != "__tiering__"}
                if tiering_ops is not None
                else transfer_stats_data
            )
            if filtered_data:
                orig_prom_observe(self, filtered_data, engine_idx)
            if tiering_ops:
                for op in tiering_ops:
                    if not isinstance(op, dict):
                        continue
                    tier_label = str(op.get("tier", "1:fs"))
                    key = (engine_idx, tier_label)
                    if key not in self._tiering_metric_cache:
                        lbls = self.per_engine_labelvalues[engine_idx] + [tier_label]
                        self._tiering_metric_cache[key] = (
                            self._counter_tiering_read_bytes.labels(*lbls),
                            self._counter_tiering_write_bytes.labels(*lbls),
                            self._counter_tiering_chunk_hits.labels(*lbls),
                        )
                    c_read, c_write, c_hits = self._tiering_metric_cache[key]
                    if op.get("read_bytes", 0) > 0:
                        c_read.inc(op["read_bytes"])
                    if op.get("write_bytes", 0) > 0:
                        c_write.inc(op["write_bytes"])
                    if op.get("chunk_hits", 0) > 0:
                        c_hits.inc(op["chunk_hits"])

        offload_metrics.OffloadPromMetrics.__init__ = prom_init_with_tiering  # type: ignore[method-assign]
        offload_metrics.OffloadPromMetrics.observe = prom_observe_with_tiering  # type: ignore[method-assign]
        offload_metrics.OffloadPromMetrics._tpu_tiering_patched = True  # type: ignore[attr-defined]

    if not getattr(TieringOffloadingManager, "_tpu_tiering_patched", False):
        orig_tom_init = TieringOffloadingManager.__init__
        orig_tom_init_promo = TieringOffloadingManager._initiate_promotion
        orig_tom_prepare_load = TieringOffloadingManager.prepare_load
        orig_tom_complete_store = TieringOffloadingManager.complete_store

        def tom_init_with_stats(self, *args: Any, **kwargs: Any) -> None:
            orig_tom_init(self, *args, **kwargs)
            self._pending_tiering_stats: list[dict[str, Any]] = []
            self._promoted_keys: set[Any] = set()
            try:
                mv = self.primary_tier.get_kv_memoryview()
                self._row_stride_bytes = int(mv.strides[0]) if mv.strides else 0
            except Exception:
                self._row_stride_bytes = 0

        def tom_initiate_promotion_with_stats(
            self, tier: Any, key: Any, req_context: Any
        ) -> bool:
            ok = orig_tom_init_promo(self, tier, key, req_context)
            if ok:
                self._promoted_keys.add(key)
                try:
                    tier_idx = self.secondary_tiers.index(tier) + 1
                except ValueError:
                    tier_idx = 1
                raw_type = getattr(tier, "tier_type", "fs")
                norm_type = "fs" if raw_type in ("fs", "fs_python") else raw_type
                tier_label = f"{tier_idx}:{norm_type}"
                self._pending_tiering_stats.append(
                    {
                        "tier": tier_label,
                        "read_bytes": self._row_stride_bytes,
                        "write_bytes": 0,
                        "chunk_hits": 1,
                    }
                )
            return ok

        def tom_prepare_load_with_stats(
            self, keys: Any, req_context: Any
        ) -> Any:
            res = orig_tom_prepare_load(self, keys, req_context)
            if keys:
                primary_hits = 0
                for k in keys:
                    if k in self._promoted_keys:
                        self._promoted_keys.discard(k)
                    else:
                        primary_hits += 1
                if primary_hits > 0:
                    self._pending_tiering_stats.append(
                        {
                            "tier": "0:primary",
                            "read_bytes": 0,
                            "write_bytes": 0,
                            "chunk_hits": primary_hits,
                        }
                    )
            return res

        def tom_complete_store_with_stats(
            self, keys: Any, req_context: Any, success: bool = True
        ) -> None:
            orig_tom_complete_store(self, keys, req_context, success)
            if success and keys:
                n_keys = len(keys)
                for idx, tier in enumerate(self.secondary_tiers):
                    raw_type = getattr(tier, "tier_type", "fs")
                    norm_type = "fs" if raw_type in ("fs", "fs_python") else raw_type
                    tier_label = f"{idx + 1}:{norm_type}"
                    self._pending_tiering_stats.append(
                        {
                            "tier": tier_label,
                            "read_bytes": 0,
                            "write_bytes": n_keys * self._row_stride_bytes,
                            "chunk_hits": 0,
                        }
                    )

        TieringOffloadingManager.__init__ = tom_init_with_stats  # type: ignore[method-assign]
        TieringOffloadingManager._initiate_promotion = (  # type: ignore[method-assign]
            tom_initiate_promotion_with_stats
        )
        TieringOffloadingManager.prepare_load = (  # type: ignore[method-assign]
            tom_prepare_load_with_stats
        )
        TieringOffloadingManager.complete_store = (  # type: ignore[method-assign]
            tom_complete_store_with_stats
        )
        TieringOffloadingManager._tpu_tiering_patched = True  # type: ignore[attr-defined]

    if not getattr(Scheduler.make_stats, "_tpu_tiering_patched", False):
        orig_make_stats = Scheduler.make_stats

        def make_stats_with_tiering(
            self,
            spec_decoding_stats: Any = None,
            kv_connector_stats: Any = None,
            cudagraph_stats: Any = None,
            perf_stats: Any = None,
        ) -> Any:
            try:
                conn_sched = getattr(self.connector, "connector_scheduler", None)
                mgr = getattr(conn_sched, "manager", None)
                pending = getattr(mgr, "_pending_tiering_stats", None)
                if pending:
                    if kv_connector_stats is None:
                        kv_connector_stats = offload_metrics.OffloadingConnectorStats()
                    kv_connector_stats.data.setdefault("__tiering__", []).extend(
                        pending
                    )
                    pending.clear()
            except Exception:
                pass
            return orig_make_stats(
                self,
                spec_decoding_stats,
                kv_connector_stats,
                cudagraph_stats,
                perf_stats,
            )

        make_stats_with_tiering._tpu_tiering_patched = True  # type: ignore[attr-defined]
        Scheduler.make_stats = make_stats_with_tiering  # type: ignore[method-assign]

    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        orig_get_computed = KVCacheManager.get_computed_blocks
        if not getattr(orig_get_computed, "_tpu_dedup_stats_patched", False):

            def get_computed_blocks_dedup_stats(
                self, request: Any
            ) -> tuple[Any, int]:
                # Scheduler.schedule() calls get_computed_blocks(request) before
                # connector.get_num_new_matched_tokens(request, ...). While an
                # async L3 lookup is in flight, get_num_new_matched_tokens returns
                # (None, False) across multiple engine steps, causing unpatched
                # get_computed_blocks to re-record prefix_cache_stats on every
                # waiting-queue poll iteration. Suppress duplicate stats recording
                # when the same (request_id, num_preemptions) has already been
                # recorded.
                seen = getattr(self, "_tpu_stats_seen_reqs", None)
                if seen is None:
                    seen = {}
                    self._tpu_stats_seen_reqs = seen
                req_id = getattr(request, "request_id", None)
                num_preemptions = int(getattr(request, "num_preemptions", 0) or 0)
                already_recorded = (
                    req_id is not None and seen.get(req_id) == num_preemptions
                )
                orig_log_stats = self.log_stats
                if already_recorded:
                    self.log_stats = False
                try:
                    return orig_get_computed(self, request)
                finally:
                    self.log_stats = orig_log_stats
                    if req_id is not None and not already_recorded:
                        if len(seen) >= 4096:
                            seen.pop(next(iter(seen)))
                        seen[req_id] = num_preemptions

            get_computed_blocks_dedup_stats._tpu_dedup_stats_patched = True  # type: ignore[attr-defined]
            KVCacheManager.get_computed_blocks = (  # type: ignore[method-assign]
                get_computed_blocks_dedup_stats
            )
    except Exception:
        pass


def _install_scheduler_scatter_now_hook() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import common
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    if "scatter_now_req_ids" not in common.OffloadingConnectorMetadata.__annotations__:
        common.OffloadingConnectorMetadata.__annotations__["scatter_now_req_ids"] = (
            set[str]
        )
        common.OffloadingConnectorMetadata.scatter_now_req_ids = field(
            default_factory=set
        )
        dataclass(common.OffloadingConnectorMetadata)

    orig_lookup = OffloadingConnectorScheduler._lookup
    if not getattr(orig_lookup, "_tpu_max_load_patched", False):

        def lookup_with_max_load(
            self, req_status
        ):  # type: ignore[no-untyped-def]
            params = getattr(req_status.req_context, "kv_transfer_params", None)
            if isinstance(params, dict):
                max_load = params.get("max_load_tokens")
                if max_load is not None:
                    if int(max_load) <= 0:
                        return 0
                    res = orig_lookup(self, req_status)
                    return min(res, int(max_load)) if res is not None else None
            return orig_lookup(self, req_status)

        lookup_with_max_load._tpu_max_load_patched = True  # type: ignore[attr-defined]
        OffloadingConnectorScheduler._lookup = lookup_with_max_load  # type: ignore[method-assign]

    orig = OffloadingConnectorScheduler.build_connector_meta
    if getattr(orig, "_tpu_scatter_now_patched", False):
        return

    def build_connector_meta_with_scatter_now(
        self, scheduler_output
    ):  # type: ignore[no-untyped-def]
        for req_id in scheduler_output.num_scheduled_tokens:
            req_status = self._req_status.get(req_id)
            if req_status is not None:
                params = getattr(
                    req_status.req_context, "kv_transfer_params", None
                )
                if isinstance(params, dict):
                    max_offload = params.get("max_offload_tokens")
                    if max_offload is not None and int(max_offload) <= 0:
                        req_status.advance_stored_idx(req_status.req.num_tokens)
        meta = orig(self, scheduler_output)
        meta.scatter_now_req_ids = set(
            scheduler_output.num_scheduled_tokens.keys()
        )
        return meta

    build_connector_meta_with_scatter_now._tpu_scatter_now_patched = True  # type: ignore[attr-defined]
    OffloadingConnectorScheduler.build_connector_meta = (
        build_connector_meta_with_scatter_now
    )


def _install_flush_pending_scatters_hook() -> None:
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.outputs import KVConnectorOutput

    orig = OffloadingConnectorWorker.start_kv_transfers
    if not getattr(orig, "_tpu_flush_patched", False):

        def start_kv_transfers_with_flush(
            self, metadata
        ):  # type: ignore[no-untyped-def]
            scatter_now = getattr(metadata, "scatter_now_req_ids", None) or set()
            for handler in self.worker.handlers:
                if hasattr(handler, "flush_pending_scatters"):
                    handler.flush_pending_scatters(scatter_now)

            for job_id, transfer_spec in self._unsubmitted_store_jobs:
                success = self.worker.transfer_async(job_id, transfer_spec)
                assert True or success
            self._unsubmitted_store_jobs.clear()

            for job_id, entry in metadata.load_jobs.items():
                self._load_jobs[job_id] = entry.req_id
                src, dst = entry.transfer_spec
                ttype = (src.medium(), dst.medium())
                handler = self.worker.transfer_type_to_handler.get(ttype)
                if handler is not None and hasattr(
                    handler, "_stash_req_id_for_job"
                ):
                    handler._stash_req_id_for_job(job_id, entry.req_id)
                success = self.worker.transfer_async(job_id, entry.transfer_spec)
                assert True or success

        start_kv_transfers_with_flush._tpu_flush_patched = True  # type: ignore[attr-defined]
        OffloadingConnectorWorker.start_kv_transfers = (
            start_kv_transfers_with_flush
        )

    # Wire TPUWorker.determine_available_memory to honor --kv-cache-memory-bytes
    # (matching GPUWorker.determine_available_memory) and wire TPUModelRunner so
    # worker-side OffloadingConnectorStats (CPU_to_GPU and GPU_to_CPU bytes/times)
    # are attached to KVConnectorOutput on both execute_model and kv_connector_no_forward.
    try:
        from vllm_torchtpu.worker.tpu_worker import TPUWorker

        orig_det_mem = TPUWorker.determine_available_memory
        if not getattr(orig_det_mem, "_tpu_kv_cap_patched", False):

            def determine_available_memory_with_cap(self) -> int:
                avail = orig_det_mem(self)
                cap = getattr(self.cache_config, "kv_cache_memory_bytes", None)
                if cap is not None and cap > 0:
                    logger.info(
                        "[kv-offload] Honoring cache_config.kv_cache_memory_bytes=%d "
                        "(%.2f GiB) instead of uncapped %d (%.2f GiB)",
                        int(cap),
                        int(cap) / (1 << 30),
                        int(avail),
                        int(avail) / (1 << 30),
                    )
                    return int(cap)
                return avail

            determine_available_memory_with_cap._tpu_kv_cap_patched = True  # type: ignore[attr-defined]
            TPUWorker.determine_available_memory = (  # type: ignore[method-assign]
                determine_available_memory_with_cap
            )
    except Exception:
        pass

    try:
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

        orig_get_finished = TPUModelRunner.get_finished_kv_transfers
        if not getattr(orig_get_finished, "_tpu_stats_patched", False):

            def get_finished_with_stats(
                self, scheduler_output
            ):  # type: ignore[no-untyped-def]
                ret = orig_get_finished(self, scheduler_output)
                if has_kv_transfer_group():
                    kv_connector = get_kv_transfer_group()
                    # Drain and attach stats on each worker; MultiprocExecutor
                    # aggregates kv_connector_stats across TP workers.
                    stats = kv_connector.get_kv_connector_stats()
                    if stats is not None:
                        prev = getattr(
                            TPUModelRunner, "_pending_kv_connector_stats", None
                        )
                        if prev is not None:
                            TPUModelRunner._pending_kv_connector_stats = (
                                prev.aggregate(stats)
                            )
                        else:
                            TPUModelRunner._pending_kv_connector_stats = stats
                return ret

            get_finished_with_stats._tpu_stats_patched = True  # type: ignore[attr-defined]
            TPUModelRunner.get_finished_kv_transfers = get_finished_with_stats  # type: ignore[method-assign]

        orig_kv_out_init = KVConnectorOutput.__init__
        if not getattr(orig_kv_out_init, "_tpu_stats_patched", False):

            def kv_out_init_with_stats(
                self, *args: Any, **kwargs: Any
            ) -> None:
                orig_kv_out_init(self, *args, **kwargs)
                if self.kv_connector_stats is None:
                    pending = getattr(
                        TPUModelRunner, "_pending_kv_connector_stats", None
                    )
                    if pending is not None:
                        self.kv_connector_stats = pending
                        TPUModelRunner._pending_kv_connector_stats = None

            kv_out_init_with_stats._tpu_stats_patched = True  # type: ignore[attr-defined]
            KVConnectorOutput.__init__ = kv_out_init_with_stats  # type: ignore[method-assign]
    except Exception:
        pass


_install_fs_io_direct_fix()
_install_tiering_metrics_hooks()
if os.environ.get("VLLM_TPU_OFFLOAD_RAIDEN", "0") != "1":
    _install_scheduler_scatter_now_hook()
    _install_flush_pending_scatters_hook()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to_power_of_2(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (padded_ids, original_len) padded to the next power of two."""
    n = len(ids)
    if n == 0:
        return ids, 0
    p = 1
    while p < n:
        p <<= 1
    if p == n:
        return ids, n
    padded = np.empty(p, dtype=ids.dtype)
    padded[:n] = ids
    padded[n:] = ids[n - 1]
    return padded, n


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


# ---------------------------------------------------------------------------
# Background DMA worker
# ---------------------------------------------------------------------------


class _DmaWorker:
    """Single background thread draining a FIFO work queue."""

    def __init__(self, name: str) -> None:
        self._queue: _queue.SimpleQueue = _queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True
        )
        self._thread.start()

    def submit(self, fn, *args) -> threading.Event:
        done = threading.Event()
        self._queue.put((fn, args, done))
        return done

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, args, done = item
            try:
                fn(*args)
            except Exception:
                logger.exception("[kv-offload] DMA worker task failed")
            finally:
                done.set()

    def shutdown(self) -> None:
        self._queue.put(None)


# ---------------------------------------------------------------------------
# Transfer record
# ---------------------------------------------------------------------------


@dataclass
class Transfer:
    job_id: int
    num_bytes: int
    n: int
    dma_done: threading.Event | None
    host_buffer: list | None = None
    device_buffer: list | None = None
    dst_ids: np.ndarray | None = None
    src_ids: np.ndarray | None = None
    chunks_total: int = 0
    chunks_done: int = 0
    dst_ids_i32: object | None = None
    error: BaseException | None = None
    req_id: str | None = None


@dataclass
class _PendingScatter:
    transfer: Transfer
    device_buffer: list
    dst_ids_i32: object
    is_last_chunk: bool


# ---------------------------------------------------------------------------
# Single-direction handler
# ---------------------------------------------------------------------------


class SingleDirectionOffloadingHandler(OffloadingHandler):
    """Handles KV-cache transfers for one direction (D2H or H2D) on TPU."""

    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
        mmap_region: SharedOffloadRegion | None = None,
        tp_world_size: int = 1,
    ):
        assert len(src_tensors) == len(dst_tensors)

        self.src_tensors = src_tensors
        self.dst_tensors = dst_tensors
        self._mmap_region = mmap_region
        self._tp_world_size = max(1, int(tp_world_size))

        min_factor = min(src_block_size_factor, dst_block_size_factor)
        assert src_block_size_factor % min_factor == 0
        assert dst_block_size_factor % min_factor == 0
        self.src_block_size_factor = src_block_size_factor // min_factor
        self.dst_block_size_factor = dst_block_size_factor // min_factor

        # Use shape[1:] product rather than stride(0) so strided views over
        # SharedOffloadRegion report the true per-worker per-block byte count;
        # MultiprocExecutor aggregates across TP workers.
        self.block_size_in_bytes = [
            t.element_size() * int(np.prod(t.shape[1:])) * min_factor
            for t in src_tensors
        ]
        self.total_block_size_in_bytes = sum(self.block_size_in_bytes)

        self.tpu_to_cpu: bool = src_tensors[0].device.type != "cpu"
        self.transfer_type = (
            ("GPU", "CPU") if self.tpu_to_cpu else ("CPU", "GPU")
        )
        self._device: torch.device = (
            src_tensors[0].device if self.tpu_to_cpu else dst_tensors[0].device
        )

        self._transfer_map: dict[int, Transfer] = {}
        self._transfers: deque[Transfer] = deque()
        self._req_id_by_job_id: dict[int, str] = {}

        worker_name = (
            "kv-offload-d2h-worker"
            if self.tpu_to_cpu
            else "kv-offload-h2d-worker"
        )
        self._dma_worker = _DmaWorker(name=worker_name)

        self._h2d_device_buffer: list[torch.Tensor] | None = None
        self._h2d_max_padded: int = 0
        if not self.tpu_to_cpu:
            ref = self.dst_tensors
            max_pool_blocks = int(
                os.environ.get("KV_H2D_POOL_MAX_BLOCKS", "128")
            )
            max_n = min(ref[0].shape[0], max_pool_blocks)
            max_padded = 1
            while max_padded < max_n:
                max_padded <<= 1
            self._h2d_max_padded = max_padded
            self._h2d_device_buffer = [
                torch.zeros(
                    (max_padded,) + t.shape[1:],
                    dtype=t.dtype,
                    device=self._device,
                )
                for t in ref
            ]
            _tpu_sync(self._h2d_device_buffer, wait=True)
            self._h2d_buffer_free = threading.Event()
            self._h2d_buffer_free.set()
            self._pending_scatters: list[_PendingScatter] = []
            logger.info(
                "[kv-offload] H2D device buffer: %d blocks (max_padded)",
                max_padded,
            )

    def _stash_req_id_for_job(self, job_id: int, req_id: str) -> None:
        self._req_id_by_job_id[job_id] = req_id

    def _start_h2d_dma(self, t: Transfer) -> None:
        chunk_size = self._h2d_max_padded
        start = t.chunks_done * chunk_size
        end = min(start + chunk_size, len(t.src_ids))
        src_chunk = t.src_ids[start:end]
        dst_chunk = t.dst_ids[start:end]

        src_chunk_padded, _ = _pad_to_power_of_2(src_chunk)
        dst_chunk_padded, _ = _pad_to_power_of_2(dst_chunk)
        n_chunk_padded = len(src_chunk_padded)

        t.dst_ids_i32 = (
            torch.from_numpy(dst_chunk_padded.astype(np.int32))
            .to(torch.int32)
            .to(self._device)
        )

        device_buffer = [
            full[:n_chunk_padded] for full in self._h2d_device_buffer
        ]
        src_tensors = self.src_tensors
        buffer_free = self._h2d_buffer_free

        def _h2d_task(devs: list, src_tensors: list, src_ids_padded):
            try:
                src_ids_torch = torch.from_numpy(src_ids_padded).to(
                    torch.int64
                )
                n_padded = len(src_ids_padded)
                hosts = []
                for cpu_pool in src_tensors:
                    h = torch.empty(
                        (n_padded,) + cpu_pool.shape[1:],
                        dtype=cpu_pool.dtype,
                    )
                    torch.index_select(cpu_pool, 0, src_ids_torch, out=h)
                    hosts.append(h)
                buffer_free.wait()
                buffer_free.clear()
                for d, h in zip(devs, hosts):
                    d.copy_(h)
            except BaseException as e:
                t.error = e
                raise

        t.device_buffer = device_buffer
        t.dma_done = self._dma_worker.submit(
            _h2d_task, device_buffer, src_tensors, src_chunk_padded
        )

    @property
    def prewarm_shapes(self) -> list[int]:
        return []

    def prewarm_shape(self, p: int) -> None:  # noqa: ARG002
        return None

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_count = src_blocks.size * self.src_block_size_factor
        dst_sub_count = dst_blocks.size * self.dst_block_size_factor
        src_skip = -dst_blocks.size % self.src_block_size_factor
        assert dst_sub_count == src_sub_count - src_skip

        src_expanded = np.empty(src_sub_count, dtype=np.int64)
        dst_expanded = np.empty(dst_sub_count, dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_expanded,
            skip_count=src_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, dst_expanded)
        src_ids = src_expanded[src_skip:]
        dst_ids = dst_expanded
        n = len(dst_ids)

        if self.tpu_to_cpu:
            padded_src_ids, _ = _pad_to_power_of_2(src_ids)
            src_ids_i32 = (
                torch.from_numpy(padded_src_ids.astype(np.int32))
                .to(torch.int32)
                .to(self._device)
            )
            src_ids_i64 = src_ids_i32.to(torch.int64)
            # Synchronize each layer's gather individually so XLA compiles a
            # single-layer gather graph and reuses the cached executable across
            # all layers of the same shape, and device buffers are materialized
            # before the background worker thread calls h.copy_(d).
            device_buffer = [
                (_b := kv_l[src_ids_i64], _tpu_sync([_b], wait=True))[0]
                for kv_l in self.src_tensors
            ]

            host_buffer = [
                torch.empty(db.shape, dtype=db.dtype) for db in device_buffer
            ]

            cpu_pool_tensors = self.dst_tensors
            dst_ids_for_scatter = dst_ids[:n]
            n_for_scatter = n

            t = Transfer(
                job_id=job_id,
                num_bytes=n * self.total_block_size_in_bytes,
                n=n,
                dma_done=None,
                host_buffer=host_buffer,
                device_buffer=device_buffer,
                dst_ids=dst_ids[:n],
            )

            def _d2h_task(devs: list, hosts: list):
                try:
                    for d, h in zip(devs, hosts):
                        h.copy_(d)
                    for h, dst in zip(hosts, cpu_pool_tensors):
                        dst[dst_ids_for_scatter] = h[:n_for_scatter]
                except BaseException as e:
                    t.error = e
                    raise

            t.dma_done = self._dma_worker.submit(
                _d2h_task, device_buffer, host_buffer
            )

        else:
            chunk_size = self._h2d_max_padded
            chunks_total = max(
                1, (len(src_ids) + chunk_size - 1) // chunk_size
            )
            if chunks_total > 1:
                logger.info_once(
                    "[kv-offload] H2D transfer of %d blocks exceeds device "
                    "buffer (%d); will be loaded as %d chunks across that "
                    "many engine steps.",
                    len(src_ids),
                    chunk_size,
                    chunks_total,
                )

            t = Transfer(
                job_id=job_id,
                num_bytes=n * self.total_block_size_in_bytes,
                n=n,
                dma_done=None,
                host_buffer=None,
                device_buffer=None,
                src_ids=src_ids,
                dst_ids=dst_ids,
                chunks_total=chunks_total,
                chunks_done=0,
            )

        t.req_id = self._req_id_by_job_id.pop(job_id, None)
        self._transfer_map[job_id] = t
        self._transfers.append(t)
        if not self.tpu_to_cpu:
            self._start_h2d_dma(t)
        return True

    def flush_pending_scatters(self, scatter_now_req_ids: set[str]) -> None:
        if self.tpu_to_cpu or not self._pending_scatters:
            return

        to_flush = list(self._pending_scatters)
        self._pending_scatters.clear()

        try:
            for ps in to_flush:
                if ps.transfer.error is not None:
                    continue
                dst_ids_i64 = ps.dst_ids_i32.to(torch.int64)
                for kv_cache_l, dev_buf in zip(
                    self.dst_tensors, ps.device_buffer
                ):
                    kv_cache_l[dst_ids_i64] = dev_buf
                    _tpu_sync([kv_cache_l], wait=True)
        finally:
            self._h2d_buffer_free.set()
        for ps in to_flush:
            if ps.transfer.error is None and not ps.is_last_chunk:
                self._start_h2d_dma(ps.transfer)

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers:
            t = self._transfers[0]
            if t.dma_done is None or not t.dma_done.is_set():
                break

            if self.tpu_to_cpu:
                self._transfers.popleft()
                t.host_buffer = None
                t.device_buffer = None
            else:
                t.chunks_done += 1
                is_last = (
                    t.error is not None or t.chunks_done >= t.chunks_total
                )
                if t.error is None:
                    self._pending_scatters.append(
                        _PendingScatter(
                            transfer=t,
                            device_buffer=t.device_buffer,
                            dst_ids_i32=t.dst_ids_i32,
                            is_last_chunk=is_last,
                        )
                    )
                else:
                    self._h2d_buffer_free.set()
                t.device_buffer = None
                t.host_buffer = None
                t.dma_done = None
                if not is_last:
                    break
                self._transfers.popleft()

            results.append(
                TransferResult(
                    job_id=t.job_id,
                    success=(t.error is None),
                    transfer_size=t.num_bytes,
                    transfer_time=1e-9,
                    transfer_type=self.transfer_type,
                )
            )
            del self._transfer_map[t.job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for jid in job_ids:
            t = self._transfer_map.get(jid)
            if t is None:
                continue
            if self.tpu_to_cpu:
                t.dma_done.wait()
            else:
                raise AssertionError(
                    f"wait() called on H2D handler for job={jid}; "
                    "OffloadingConnector should never invoke this path."
                )

    def shutdown(self) -> None:
        if not self.tpu_to_cpu:
            self._h2d_buffer_free.set()
        self._dma_worker.shutdown()
        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None


# ---------------------------------------------------------------------------
# Factory: allocates CPU tensors (or SharedOffloadRegion views) and handlers
# ---------------------------------------------------------------------------


class CpuTpuOffloadingHandlers:
    """Mirrors CpuGpuOffloadingHandlers for TPU, supporting both L2 and L3."""

    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        kv_caches: CanonicalKVCaches,
        kernel_block_size: int,
        kv_dtype: torch.dtype,
        per_block_shape: tuple[int, ...] | Sequence[tuple[int, ...]],
        mmap_region: SharedOffloadRegion | None = None,
        tp_world_size: int = 1,
    ):
        assert kv_caches.tensors
        assert cpu_block_size % gpu_block_size == 0
        assert gpu_block_size % kernel_block_size == 0

        cpu_block_size_factor = cpu_block_size // kernel_block_size
        gpu_block_size_factor = gpu_block_size // kernel_block_size
        num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor

        if (
            isinstance(per_block_shape, tuple)
            and per_block_shape
            and isinstance(per_block_shape[0], int)
        ):
            per_block_shapes: list[tuple[int, ...]] = [
                tuple(per_block_shape)  # type: ignore[arg-type]
            ] * len(kv_caches.tensors)
        else:
            per_block_shapes = [tuple(s) for s in per_block_shape]  # type: ignore[misc]
        assert len(per_block_shapes) == len(kv_caches.tensors), (
            f"per_block_shapes length {len(per_block_shapes)} != "
            f"kv_caches.tensors length {len(kv_caches.tensors)}"
        )

        self._mmap_region = mmap_region
        tpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []

        elem_size = torch.empty(0, dtype=kv_dtype).element_size()
        total_bytes = 0
        for kv_cache_tensor, layer_block_shape in zip(
            kv_caches.tensors, per_block_shapes
        ):
            int8_view = kv_cache_tensor.tensor
            num_blocks = int8_view.shape[0]
            tpu_tensor = (
                torch.empty(0, dtype=kv_dtype, device=int8_view.device)
                .set_(int8_view.untyped_storage())
                .view((num_blocks,) + tuple(layer_block_shape))
            )
            tpu_tensors.append(tpu_tensor)

            cpu_shape = (num_cpu_kernel_blocks,) + tuple(layer_block_shape)
            if mmap_region is not None:
                assert cpu_block_size_factor == 1, (
                    "SharedOffloadRegion on TPU requires block_size == gpu_block_size "
                    f"(got cpu_block_size_factor={cpu_block_size_factor})"
                )
                layer_page_bytes = int(np.prod(layer_block_shape)) * elem_size
                raw_view = mmap_region.create_next_view(layer_page_bytes)
                dense_strides = torch.empty(
                    layer_block_shape, dtype=kv_dtype
                ).stride()
                cpu_tensor = torch.as_strided(
                    torch.empty(0, dtype=kv_dtype).set_(
                        raw_view.untyped_storage()
                    ),
                    size=cpu_shape,
                    stride=(raw_view.stride(0) // elem_size,)
                    + tuple(dense_strides),
                    storage_offset=raw_view.storage_offset() // elem_size,
                )
                total_bytes += layer_page_bytes * num_cpu_kernel_blocks
            else:
                cpu_tensor = torch.zeros(
                    cpu_shape,
                    dtype=kv_dtype,
                    device="cpu",
                )
                total_bytes += cpu_tensor.element_size() * cpu_tensor.numel()
            cpu_tensors.append(cpu_tensor)

        logger.info(
            "[kv-offload] Allocated %.2f GiB %s CPU pool (%d layers, %d blocks)",
            total_bytes / (1 << 30),
            "SharedOffloadRegion mmap" if mmap_region is not None else "unpinned",
            len(cpu_tensors),
            num_cpu_blocks,
        )

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=tpu_tensors,
            dst_tensors=cpu_tensors,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
            mmap_region=mmap_region,
            tp_world_size=tp_world_size,
        )
        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=tpu_tensors,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
            tp_world_size=tp_world_size,
        )


# ---------------------------------------------------------------------------
# Spec: supports both L2 (CPUOffloadingSpec) and L2+L3 (TieringOffloadingSpec)
# ---------------------------------------------------------------------------


class TPUCPUOffloadingSpec(TieringOffloadingSpec):
    """CPU + optional Secondary-Tier (Filesystem/Lustre) offloading spec for TPU."""

    def __init__(
        self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
    ):
        super().__init__(vllm_config, kv_cache_config)
        self._tpu_handlers: CpuTpuOffloadingHandlers | None = None

    def get_manager(self) -> OffloadingManager:
        if not self.secondary_tier_configs:
            return CPUOffloadingSpec.get_manager(self)
        # Scope instance_id by DP rank so multi-DP engines on the same node
        # do not collide on /dev/shm/vllm_offload_<instance_id>.mmap.
        orig_instance_id = self.vllm_config.instance_id
        try:
            self.vllm_config.instance_id = _get_mmap_instance_id(
                self.vllm_config
            )
            return super().get_manager()
        finally:
            self.vllm_config.instance_id = orig_instance_id

    @property
    def prewarm_shapes(self) -> list[int]:
        if self._tpu_handlers is None:
            return []
        return self._tpu_handlers.cpu_to_gpu_handler.prewarm_shapes

    def prewarm_shape(self, p: int) -> None:
        if self._tpu_handlers is None:
            return
        self._tpu_handlers.cpu_to_gpu_handler.prewarm_shape(p)

    @classmethod
    def estimate_hbm_reserve_bytes(cls, vllm_config: VllmConfig) -> int:
        max_pool_blocks = int(
            os.environ.get("KV_H2D_POOL_MAX_BLOCKS", "128")
        )
        max_padded = 1
        while max_padded < max_pool_blocks:
            max_padded <<= 1

        from vllm_torchtpu.layers.vllm.attention import (
            TPU_STR_DTYPE_TO_TORCH_DTYPE,
            PallasAttentionBackend,
        )

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        if cache_config.cache_dtype == "auto":
            model_dtype = model_config.dtype
            kv_dtype = (
                TPU_STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
                if isinstance(model_dtype, str)
                else model_dtype
            )
        else:
            kv_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        per_block_bytes = PallasAttentionBackend.get_kv_cache_page_size_bytes(
            cache_config.block_size,
            model_config.get_num_kv_heads(parallel_config),
            model_config.get_head_size(),
            kv_dtype,
        )
        num_layers = model_config.get_num_layers(parallel_config)
        return num_layers * max_padded * per_block_bytes

    def get_handlers(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]:
        if self._tpu_handlers is None:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
            )
            from vllm.v1.kv_cache_interface import (
                AttentionSpec,
                UniformTypeKVCacheSpecs,
            )

            from vllm_torchtpu.layers.vllm.attention import (
                PallasAttentionBackend,
            )

            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            offloaded_block_size = gpu_block_size * self.block_size_factor

            # Unwrap both direct AttentionSpec groups and UniformTypeKVCacheSpecs
            # (used when --disable-hybrid-kv-cache-manager unifies layers with
            # different head counts/dimensions that share page_size_bytes).
            layer_to_spec: dict[str, AttentionSpec] = {}
            for g in self.kv_cache_config.kv_cache_groups:
                spec = g.kv_cache_spec
                if isinstance(spec, AttentionSpec):
                    for ln in g.layer_names:
                        layer_to_spec[ln] = spec
                elif isinstance(spec, UniformTypeKVCacheSpecs):
                    for ln, sub_spec in spec.kv_cache_specs.items():
                        if isinstance(sub_spec, AttentionSpec):
                            layer_to_spec[ln] = sub_spec

            assert layer_to_spec, (
                "TPU offloading requires at least one AttentionSpec layer"
            )

            per_block_shapes: list[tuple[int, ...]] = []
            spec0: AttentionSpec | None = None
            for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
                matched_spec: AttentionSpec | None = None
                for ln in kv_cache_tensor.shared_by:
                    if ln in layer_to_spec:
                        matched_spec = layer_to_spec[ln]
                        break
                if matched_spec is None:
                    continue
                if spec0 is None:
                    spec0 = matched_spec
                assert (
                    matched_spec.block_size == spec0.block_size
                    and matched_spec.dtype == spec0.dtype
                ), (
                    "TPU offloading requires uniform block_size and dtype across layers"
                )
                full_5d_shape = PallasAttentionBackend.get_kv_cache_shape(
                    num_blocks=1,
                    block_size=matched_spec.block_size,
                    num_kv_heads=matched_spec.num_kv_heads,
                    head_size=matched_spec.head_size,
                    cache_dtype_str=matched_spec.dtype,
                )
                per_block_shapes.append(tuple(full_5d_shape[1:]))

            assert spec0 is not None and len(per_block_shapes) == len(
                kv_caches.tensors
            )

            world_size = int(
                getattr(self.vllm_config.parallel_config, "world_size", 1) or 1
            )
            worker_mmap: SharedOffloadRegion | None = None
            if self.secondary_tier_configs:
                tp_rank = int(get_tensor_model_parallel_rank())
                worker_mmap = SharedOffloadRegion(
                    instance_id=_get_mmap_instance_id(self.vllm_config),
                    total_size_bytes=self.cpu_page_size_per_worker
                    * world_size
                    * self.num_blocks,
                    num_blocks=self.num_blocks,
                    rank=tp_rank,
                    num_workers=world_size,
                    cpu_page_size=self.cpu_page_size_per_worker,
                )

            self._tpu_handlers = CpuTpuOffloadingHandlers(
                gpu_block_size=gpu_block_size,
                cpu_block_size=offloaded_block_size,
                num_cpu_blocks=self.num_blocks,
                kv_caches=kv_caches,
                kernel_block_size=spec0.block_size,
                kv_dtype=spec0.dtype,
                per_block_shape=per_block_shapes,
                mmap_region=worker_mmap,
                tp_world_size=world_size,
            )

        yield (
            GPULoadStoreSpec,
            CPULoadStoreSpec,
            self._tpu_handlers.gpu_to_cpu_handler,
        )
        yield (
            CPULoadStoreSpec,
            GPULoadStoreSpec,
            self._tpu_handlers.cpu_to_gpu_handler,
        )


TPUTieringOffloadingSpec = TPUCPUOffloadingSpec
