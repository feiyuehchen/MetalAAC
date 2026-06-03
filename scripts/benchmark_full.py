#!/usr/bin/env python3
"""Full benchmark: 50 runs × 4 durations, CPU vs GPU vs Apple AAC.

Reports mean ± std for each stage and end-to-end throughput.
Compares against Apple's afconvert AAC encoder.

Usage:
    python scripts/benchmark_full.py
    python scripts/benchmark_full.py --n-runs 10   # fewer runs for quick test
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlx.core as mx

from metal_aac.core.mdct import (
    MDCTBasis,
    MDCTBasisGPU,
    frame_signal,
    imdct_cpu,
    imdct_gpu,
    mdct_cpu,
    mdct_gpu,
)
from metal_aac.core.psychoacoustic import (
    PsychoacousticTables,
    PsychoacousticTablesGPU,
    psychoacoustic_cpu,
    psychoacoustic_gpu,
)
from metal_aac.metrics.throughput import Timer
from metal_aac.tables.windows import get_window


SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_SIZE = 1024
DURATIONS = [10, 60, 300, 3600]


def generate_random_signal(duration: float, seed: int) -> np.ndarray:
    """Generate a random harmonic signal (deterministic per seed)."""
    rng = np.random.RandomState(seed)
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE

    n_partials = rng.randint(3, 8)
    freqs = rng.uniform(100, 4000, n_partials)
    amps = rng.uniform(0.05, 0.3, n_partials)
    phases = rng.uniform(0, 2 * np.pi, n_partials)

    signal = np.zeros(n, dtype=np.float64)
    for f, a, p in zip(freqs, amps, phases):
        signal += a * np.sin(2 * np.pi * f * t + p)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7

    return signal.astype(np.float32)


def write_wav(path: str, pcm: np.ndarray, sr: int = 44100):
    pcm16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
    data_size = len(pcm16) * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm16.tobytes())


def benchmark_metal_aac(pcm: np.ndarray, bitrate: int, tmpdir: str) -> float:
    """Encode with afconvert, return encode wall-clock time."""
    wav_path = os.path.join(tmpdir, "in.wav")
    m4a_path = os.path.join(tmpdir, "out.m4a")
    write_wav(wav_path, pcm)
    start = time.perf_counter()
    subprocess.run(
        ["afconvert", "-f", "m4af", "-d", "aac", "-b", str(bitrate),
         "-s", "3", wav_path, m4a_path],
        check=True, capture_output=True,
    )
    elapsed = time.perf_counter() - start
    os.unlink(m4a_path)
    return elapsed


def run_full_benchmark(n_runs: int = 50, durations: list[int] | None = None):
    if durations is None:
        durations = DURATIONS

    window = get_window("kbd", FRAME_SIZE)
    basis_cpu = MDCTBasis.create(FRAME_SIZE)
    basis_gpu = MDCTBasisGPU(FRAME_SIZE)
    psy_tables = PsychoacousticTables.create(SAMPLE_RATE, FRAME_SIZE)
    psy_gpu = PsychoacousticTablesGPU(psy_tables)

    all_results = {}

    for dur in durations:
        print(f"\n{'='*70}")
        print(f"Duration: {dur}s  |  {n_runs} runs  |  generating signal...")
        print(f"{'='*70}")

        # Generate ONE signal for this duration (stages don't depend on content)
        pcm = generate_random_signal(dur, seed=42)
        n_frames_total = (len(pcm) + HOP_SIZE - 1) // HOP_SIZE
        print(f"  Samples: {len(pcm):,}  Frames: ~{n_frames_total:,}")

        # Frame once
        frames_np = frame_signal(pcm, FRAME_SIZE, HOP_SIZE, window)
        n_frames = len(frames_np)
        print(f"  Actual frames: {n_frames:,}")

        frames_mx = mx.array(frames_np)
        mx.eval(frames_mx)

        # ---- Stage benchmarks: 50 runs each ----

        # MDCT CPU
        mdct_cpu(frames_np, basis_cpu)  # warmup
        cpu_mdct_times = []
        for _ in range(n_runs):
            with Timer() as t:
                mdct_cpu(frames_np, basis_cpu)
            cpu_mdct_times.append(t.elapsed)

        # MDCT GPU
        r = mdct_gpu(frames_mx, basis_gpu); mx.eval(r)  # warmup
        gpu_mdct_times = []
        for _ in range(n_runs):
            with Timer() as t:
                r = mdct_gpu(frames_mx, basis_gpu)
                mx.eval(r)
            gpu_mdct_times.append(t.elapsed)

        # Psychoacoustic CPU
        psychoacoustic_cpu(frames_np, psy_tables)  # warmup
        cpu_psy_times = []
        for _ in range(n_runs):
            with Timer() as t:
                psychoacoustic_cpu(frames_np, psy_tables)
            cpu_psy_times.append(t.elapsed)

        # Psychoacoustic GPU
        r = psychoacoustic_gpu(frames_mx, psy_gpu); mx.eval(r)  # warmup
        gpu_psy_times = []
        for _ in range(n_runs):
            with Timer() as t:
                r = psychoacoustic_gpu(frames_mx, psy_gpu)
                mx.eval(r)
            gpu_psy_times.append(t.elapsed)

        # IMDCT CPU
        spectra_np = mdct_cpu(frames_np, basis_cpu)
        imdct_cpu(spectra_np, basis_cpu)  # warmup
        cpu_imdct_times = []
        for _ in range(n_runs):
            with Timer() as t:
                imdct_cpu(spectra_np, basis_cpu)
            cpu_imdct_times.append(t.elapsed)

        # IMDCT GPU
        spectra_mx = mx.array(spectra_np); mx.eval(spectra_mx)
        r = imdct_gpu(spectra_mx, basis_gpu); mx.eval(r)  # warmup
        gpu_imdct_times = []
        for _ in range(n_runs):
            with Timer() as t:
                r = imdct_gpu(spectra_mx, basis_gpu)
                mx.eval(r)
            gpu_imdct_times.append(t.elapsed)

        # ---- Apple AAC: 50 runs (afconvert is fast) ----
        apple_times = []
        with tempfile.TemporaryDirectory() as tmpdir:
            # warmup
            benchmark_metal_aac(pcm, 128000, tmpdir)
            for i in range(n_runs):
                elapsed = benchmark_metal_aac(pcm, 128000, tmpdir)
                apple_times.append(elapsed)

        # Free GPU memory
        del frames_mx, spectra_mx
        mx.metal.clear_cache()

        # ---- Compute stats ----
        def stats(times):
            a = np.array(times)
            return float(np.mean(a)), float(np.std(a))

        mdct_cpu_m, mdct_cpu_s = stats(cpu_mdct_times)
        mdct_gpu_m, mdct_gpu_s = stats(gpu_mdct_times)
        psy_cpu_m, psy_cpu_s = stats(cpu_psy_times)
        psy_gpu_m, psy_gpu_s = stats(gpu_psy_times)
        imdct_cpu_m, imdct_cpu_s = stats(cpu_imdct_times)
        imdct_gpu_m, imdct_gpu_s = stats(gpu_imdct_times)
        apple_m, apple_s = stats(apple_times)

        mdct_speedup = mdct_cpu_m / mdct_gpu_m if mdct_gpu_m > 0 else 0
        psy_speedup = psy_cpu_m / psy_gpu_m if psy_gpu_m > 0 else 0
        imdct_speedup = imdct_cpu_m / imdct_gpu_m if imdct_gpu_m > 0 else 0

        apple_rtf = apple_m / dur

        result = {
            "duration_sec": dur,
            "n_frames": n_frames,
            "n_runs": n_runs,
            "mdct_cpu_ms": (mdct_cpu_m * 1000, mdct_cpu_s * 1000),
            "mdct_gpu_ms": (mdct_gpu_m * 1000, mdct_gpu_s * 1000),
            "mdct_speedup": mdct_speedup,
            "psy_cpu_ms": (psy_cpu_m * 1000, psy_cpu_s * 1000),
            "psy_gpu_ms": (psy_gpu_m * 1000, psy_gpu_s * 1000),
            "psy_speedup": psy_speedup,
            "imdct_cpu_ms": (imdct_cpu_m * 1000, imdct_cpu_s * 1000),
            "imdct_gpu_ms": (imdct_gpu_m * 1000, imdct_gpu_s * 1000),
            "imdct_speedup": imdct_speedup,
            "apple_sec": (apple_m, apple_s),
            "apple_rtf": apple_rtf,
        }
        all_results[dur] = result

        # Print per-duration summary
        def fmt(mean_ms, std_ms):
            return f"{mean_ms:9.2f} ± {std_ms:.2f}"

        print(f"\n  {'Stage':<16s} {'CPU (ms)':>18s} {'GPU (ms)':>18s} {'Speedup':>8s}")
        print(f"  {'-'*62}")
        print(f"  {'MDCT':<16s} {fmt(*result['mdct_cpu_ms']):>18s} {fmt(*result['mdct_gpu_ms']):>18s} {mdct_speedup:>7.2f}x")
        print(f"  {'Psychoacoustic':<16s} {fmt(*result['psy_cpu_ms']):>18s} {fmt(*result['psy_gpu_ms']):>18s} {psy_speedup:>7.2f}x")
        print(f"  {'IMDCT':<16s} {fmt(*result['imdct_cpu_ms']):>18s} {fmt(*result['imdct_gpu_ms']):>18s} {imdct_speedup:>7.2f}x")
        print(f"\n  Apple AAC (afconvert): {apple_m*1000:.1f} ± {apple_s*1000:.1f} ms  (RTF: {apple_rtf:.6f})")

    # ---- Final summary table ----
    print(f"\n\n{'='*90}")
    print("SUMMARY: GPU Speedup Scaling (mean ± std, {n_runs} runs)")
    print(f"{'='*90}")
    print(f"{'Duration':>10s} {'Frames':>8s} | {'MDCT':>12s} {'Psycho':>12s} {'IMDCT':>12s} | {'Apple RTF':>12s}")
    print(f"{'-'*10} {'-'*8}-+-{'-'*12}-{'-'*12}-{'-'*12}-+-{'-'*12}")

    for dur in durations:
        r = all_results[dur]
        print(
            f"{dur:>8d}s {r['n_frames']:>8d} | "
            f"{r['mdct_speedup']:>10.2f}x "
            f"{r['psy_speedup']:>10.2f}x "
            f"{r['imdct_speedup']:>10.2f}x | "
            f"{r['apple_rtf']:>10.6f}"
        )

    print(f"\nDetailed timing (ms, mean ± std):")
    print(f"{'Duration':>10s} | {'MDCT CPU':>16s} {'MDCT GPU':>16s} | {'Psycho CPU':>16s} {'Psycho GPU':>16s} | {'Apple':>16s}")
    print("-" * 105)
    for dur in durations:
        r = all_results[dur]
        def f(t): return f"{t[0]:.2f}±{t[1]:.2f}"
        print(
            f"{dur:>8d}s | "
            f"{f(r['mdct_cpu_ms']):>16s} {f(r['mdct_gpu_ms']):>16s} | "
            f"{f(r['psy_cpu_ms']):>16s} {f(r['psy_gpu_ms']):>16s} | "
            f"{f(r['apple_sec']):>16s}s"
        )

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--durations", type=int, nargs="*", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_full_benchmark(args.n_runs, args.durations)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for k, v in results.items():
            serializable[k] = {
                kk: list(vv) if isinstance(vv, tuple) else vv
                for kk, vv in v.items()
            }
        with open(out, "w") as fp:
            json.dump(serializable, fp, indent=2)
        print(f"\nSaved to {out}")
