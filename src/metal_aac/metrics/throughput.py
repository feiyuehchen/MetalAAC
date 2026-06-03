"""Throughput and performance metrics.

Defined in BENCHMARK.md Part A.
"""

from __future__ import annotations

import resource
import time
from typing import Any, Callable


def compute_rtf(processing_time_sec: float, audio_duration_sec: float) -> float:
    """Real-Time Factor. < 1.0 means faster than real-time."""
    if audio_duration_sec <= 0:
        return float("inf")
    return processing_time_sec / audio_duration_sec


def compute_frame_throughput(num_frames: int, processing_time_sec: float) -> float:
    """Frames processed per second."""
    if processing_time_sec <= 0:
        return float("inf")
    return num_frames / processing_time_sec


def compute_gpu_speedup(cpu_time_sec: float, gpu_time_sec: float) -> float:
    """GPU speedup factor. > 1.0 means GPU is faster."""
    if gpu_time_sec <= 0:
        return float("inf")
    return cpu_time_sec / gpu_time_sec


def measure_peak_memory(func: Callable, *args: Any) -> tuple[float, Any]:
    """Measure peak memory usage of a function call.

    Returns (peak_mb, result).
    Uses maxrss which includes all process memory.
    """
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = func(*args)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports maxrss in bytes
    peak_mb = (after - before) / (1024 * 1024)
    if peak_mb < 0:
        peak_mb = after / (1024 * 1024)
    return peak_mb, result


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
