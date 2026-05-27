"""Latency and throughput tracking for pruning experiments.

Per-sample metrics:
  - total wall-clock time for one bot.inference() call
  - GPU peak memory during that call
  - n_visual_pre / n_visual_post (sanity check that pruning fired)
  - n_generated tokens

PHASE BREAKDOWN (added 2026-05-27):
  - prune_time_s: time inside the pruner's select_indices call
  - prefill_time_s: time in the LLM body's first forward (input processing)
  - decode_time_s: time in subsequent LLM body forwards (autoregressive gen)

The phase accumulators are reset at the start of each sample by the
SampleTimer. The patcher writes into them during pruning and LM forward.
"""
import time
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional
import json
import os


@dataclass
class SampleLatency:
    sample_id: str
    total_time_s: float
    gpu_peak_mb: float
    n_visual_pre: int
    n_visual_post: int
    n_generated: int
    prune_time_s: float = 0.0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0


class LatencyTracker:
    """Per-sample latency capture. Use as a context manager around inference."""

    def __init__(self):
        self.records: List[SampleLatency] = []
        self._current_visual_pre = 576
        self._current_visual_post = 576
        # Per-sample phase accumulators; reset by _SampleTimer.__enter__.
        self._prune_time_s = 0.0
        self._prefill_time_s = 0.0
        self._decode_time_s = 0.0

    def set_visual_counts(self, pre: int, post: int):
        self._current_visual_pre = pre
        self._current_visual_post = post

    def add_prune_time(self, dt: float):
        self._prune_time_s += dt

    def add_prefill_time(self, dt: float):
        self._prefill_time_s += dt

    def add_decode_time(self, dt: float):
        self._decode_time_s += dt

    def _reset_phase_accumulators(self):
        self._prune_time_s = 0.0
        self._prefill_time_s = 0.0
        self._decode_time_s = 0.0

    def time_sample(self, sample_id: str, n_generated: Optional[int] = None):
        return _SampleTimer(self, sample_id, n_generated)

    def summary(self):
        if not self.records:
            return {}
        times = [r.total_time_s for r in self.records]
        mems = [r.gpu_peak_mb for r in self.records]
        n_post = [r.n_visual_post for r in self.records]
        prune_t = [r.prune_time_s for r in self.records]
        prefill_t = [r.prefill_time_s for r in self.records]
        decode_t = [r.decode_time_s for r in self.records]
        n = len(self.records)

        def pct(lst, p):
            return sorted(lst)[min(int(p * n), n - 1)]

        return {
            "n_samples": n,
            "total_time_s": sum(times),
            "mean_time_s": sum(times) / n,
            "p50_time_s": pct(times, 0.5),
            "p95_time_s": pct(times, 0.95),
            "mean_gpu_peak_mb": sum(mems) / n,
            "mean_visual_post_prune": sum(n_post) / n,
            "mean_prune_time_s": sum(prune_t) / n,
            "mean_prefill_time_s": sum(prefill_t) / n,
            "p50_prefill_time_s": pct(prefill_t, 0.5),
            "p95_prefill_time_s": pct(prefill_t, 0.95),
            "mean_decode_time_s": sum(decode_t) / n,
            "p50_decode_time_s": pct(decode_t, 0.5),
            "p95_decode_time_s": pct(decode_t, 0.95),
        }

    def save_jsonl(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r)) + "\n")


class _SampleTimer:
    def __init__(self, tracker: LatencyTracker, sample_id: str, n_generated: Optional[int]):
        self.tracker = tracker
        self.sample_id = sample_id
        self.n_generated = n_generated
        self.t0 = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.tracker._reset_phase_accumulators()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.t0
        peak_mb = 0.0
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        self.tracker.records.append(SampleLatency(
            sample_id=self.sample_id,
            total_time_s=elapsed,
            gpu_peak_mb=peak_mb,
            n_visual_pre=self.tracker._current_visual_pre,
            n_visual_post=self.tracker._current_visual_post,
            n_generated=self.n_generated if self.n_generated is not None else -1,
            prune_time_s=self.tracker._prune_time_s,
            prefill_time_s=self.tracker._prefill_time_s,
            decode_time_s=self.tracker._decode_time_s,
        ))
        return False

    def set_n_generated(self, n: int):
        self.n_generated = n