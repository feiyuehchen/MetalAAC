"""Benchmark framework for comparing CPU vs GPU performance per stage.

Measures each pipeline stage independently and end-to-end, following
the evaluation protocol defined in BENCHMARK.md Part A.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from metal_aac.core.mdct import (
    MDCTBasis,
    MDCTBasisGPU,
    frame_signal,
    mdct_cpu,
    mdct_gpu,
    imdct_cpu,
    imdct_gpu,
)
from metal_aac.core.psychoacoustic import (
    PsychoacousticTables,
    PsychoacousticTablesGPU,
    psychoacoustic_cpu,
    psychoacoustic_gpu,
)
from metal_aac.core.quantization import (
    quantize_cpu,
    precompute_sf_gains,
    quantize_frame_gpu,
)
from metal_aac.decoder import DecoderConfig, decode
from metal_aac.encoder import EncoderConfig, encode
from metal_aac.metrics.quality import compute_snr, compute_spectral_convergence
from metal_aac.metrics.throughput import (
    Timer,
    compute_frame_throughput,
    compute_gpu_speedup,
    compute_rtf,
)
from metal_aac.tables.scalefactor_bands import get_sfb_offsets
from metal_aac.tables.windows import get_window


@dataclass
class StageResult:
    stage: str
    cpu_time: float
    gpu_time: float
    speedup: float
    cpu_throughput: float
    gpu_throughput: float
    num_frames: int


@dataclass
class QualityResult:
    signal_name: str
    snr_db: float
    spectral_convergence: float
    actual_bitrate_kbps: float


@dataclass
class BenchmarkResult:
    stages: list[StageResult] = field(default_factory=list)
    quality: list[QualityResult] = field(default_factory=list)
    encoder_cpu_rtf: float = 0.0
    encoder_gpu_rtf: float = 0.0
    decoder_cpu_rtf: float = 0.0
    decoder_gpu_rtf: float = 0.0
    hardware: str = ""
    mlx_version: str = ""

    def to_dict(self) -> dict:
        return {
            "stages": [asdict(s) for s in self.stages],
            "quality": [asdict(q) for q in self.quality],
            "encoder_cpu_rtf": self.encoder_cpu_rtf,
            "encoder_gpu_rtf": self.encoder_gpu_rtf,
            "decoder_cpu_rtf": self.decoder_cpu_rtf,
            "decoder_gpu_rtf": self.decoder_gpu_rtf,
            "hardware": self.hardware,
            "mlx_version": self.mlx_version,
        }


def _detect_hardware() -> str:
    chip = "unknown"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        chip = result.stdout.strip()
    except Exception:
        chip = platform.processor()
    return f"{chip} ({platform.machine()})"


def benchmark_stage_mdct(
    frames_np: np.ndarray,
    n_warmup: int = 1,
    n_runs: int = 10,
) -> StageResult:
    """Benchmark MDCT stage: CPU vs GPU."""
    num_frames = len(frames_np)
    frame_size = frames_np.shape[1]
    basis_cpu = MDCTBasis.create(frame_size)

    # CPU
    for _ in range(n_warmup):
        mdct_cpu(frames_np, basis_cpu)
    cpu_times = []
    for _ in range(n_runs):
        with Timer() as t:
            mdct_cpu(frames_np, basis_cpu)
        cpu_times.append(t.elapsed)
    cpu_time = np.mean(cpu_times)

    # GPU
    gpu_time = cpu_time  # fallback
    if HAS_MLX:
        basis_gpu = MDCTBasisGPU(frame_size)
        frames_mx = mx.array(frames_np)
        mx.eval(frames_mx)

        for _ in range(n_warmup):
            r = mdct_gpu(frames_mx, basis_gpu)
            mx.eval(r)
        gpu_times = []
        for _ in range(n_runs):
            with Timer() as t:
                r = mdct_gpu(frames_mx, basis_gpu)
                mx.eval(r)
            gpu_times.append(t.elapsed)
        gpu_time = np.mean(gpu_times)

    return StageResult(
        stage="mdct",
        cpu_time=cpu_time,
        gpu_time=gpu_time,
        speedup=compute_gpu_speedup(cpu_time, gpu_time),
        cpu_throughput=compute_frame_throughput(num_frames, cpu_time),
        gpu_throughput=compute_frame_throughput(num_frames, gpu_time),
        num_frames=num_frames,
    )


def benchmark_stage_psychoacoustic(
    frames_np: np.ndarray,
    sample_rate: int = 44100,
    n_warmup: int = 1,
    n_runs: int = 10,
) -> StageResult:
    """Benchmark psychoacoustic model: CPU vs GPU."""
    num_frames = len(frames_np)
    frame_size = frames_np.shape[1]
    tables = PsychoacousticTables.create(sample_rate, frame_size)

    # CPU
    for _ in range(n_warmup):
        psychoacoustic_cpu(frames_np, tables)
    cpu_times = []
    for _ in range(n_runs):
        with Timer() as t:
            psychoacoustic_cpu(frames_np, tables)
        cpu_times.append(t.elapsed)
    cpu_time = np.mean(cpu_times)

    # GPU
    gpu_time = cpu_time
    if HAS_MLX:
        tables_gpu = PsychoacousticTablesGPU(tables)
        frames_mx = mx.array(frames_np)
        mx.eval(frames_mx)

        for _ in range(n_warmup):
            r = psychoacoustic_gpu(frames_mx, tables_gpu)
            mx.eval(r)
        gpu_times = []
        for _ in range(n_runs):
            with Timer() as t:
                r = psychoacoustic_gpu(frames_mx, tables_gpu)
                mx.eval(r)
            gpu_times.append(t.elapsed)
        gpu_time = np.mean(gpu_times)

    return StageResult(
        stage="psychoacoustic",
        cpu_time=cpu_time,
        gpu_time=gpu_time,
        speedup=compute_gpu_speedup(cpu_time, gpu_time),
        cpu_throughput=compute_frame_throughput(num_frames, cpu_time),
        gpu_throughput=compute_frame_throughput(num_frames, gpu_time),
        num_frames=num_frames,
    )


def benchmark_stage_imdct(
    frames_np: np.ndarray,
    n_warmup: int = 1,
    n_runs: int = 10,
) -> StageResult:
    """Benchmark IMDCT: CPU vs GPU."""
    frame_size = frames_np.shape[1]
    basis_cpu = MDCTBasis.create(frame_size)
    spectra = mdct_cpu(frames_np, basis_cpu)
    num_frames = len(spectra)

    # CPU
    for _ in range(n_warmup):
        imdct_cpu(spectra, basis_cpu)
    cpu_times = []
    for _ in range(n_runs):
        with Timer() as t:
            imdct_cpu(spectra, basis_cpu)
        cpu_times.append(t.elapsed)
    cpu_time = np.mean(cpu_times)

    # GPU
    gpu_time = cpu_time
    if HAS_MLX:
        basis_gpu = MDCTBasisGPU(frame_size)
        spectra_mx = mx.array(spectra)
        mx.eval(spectra_mx)

        for _ in range(n_warmup):
            r = imdct_gpu(spectra_mx, basis_gpu)
            mx.eval(r)
        gpu_times = []
        for _ in range(n_runs):
            with Timer() as t:
                r = imdct_gpu(spectra_mx, basis_gpu)
                mx.eval(r)
            gpu_times.append(t.elapsed)
        gpu_time = np.mean(gpu_times)

    return StageResult(
        stage="imdct",
        cpu_time=cpu_time,
        gpu_time=gpu_time,
        speedup=compute_gpu_speedup(cpu_time, gpu_time),
        cpu_throughput=compute_frame_throughput(num_frames, cpu_time),
        gpu_throughput=compute_frame_throughput(num_frames, gpu_time),
        num_frames=num_frames,
    )


def benchmark_end_to_end(
    pcm: np.ndarray,
    signal_name: str,
    sample_rate: int = 44100,
    target_bitrate: float = 128.0,
) -> tuple[QualityResult, dict]:
    """Benchmark end-to-end encode -> decode with quality measurement."""
    audio_duration = len(pcm) / sample_rate
    timings_all = {}

    for backend, use_gpu in [("cpu", False), ("gpu", True)]:
        if use_gpu and not HAS_MLX:
            continue

        enc_cfg = EncoderConfig(
            sample_rate=sample_rate,
            use_gpu=use_gpu,
            target_bitrate_kbps=target_bitrate,
        )
        dec_cfg = DecoderConfig(
            use_gpu=use_gpu,
            output_length=len(pcm),
        )

        enc_result = encode(pcm, enc_cfg)
        dec_result = decode(enc_result.bitstream, dec_cfg)

        rtf_enc = compute_rtf(
            sum(enc_result.timings.values()), audio_duration
        )
        rtf_dec = compute_rtf(
            sum(dec_result.timings.values()), audio_duration
        )

        timings_all[backend] = {
            "encode_rtf": rtf_enc,
            "decode_rtf": rtf_dec,
            "encode_timings": enc_result.timings,
            "decode_timings": dec_result.timings,
        }

    # Quality from CPU path (deterministic)
    enc_cfg = EncoderConfig(
        sample_rate=sample_rate,
        use_gpu=False,
        target_bitrate_kbps=target_bitrate,
    )
    dec_cfg = DecoderConfig(use_gpu=False, output_length=len(pcm))
    enc_result = encode(pcm, enc_cfg)
    dec_result = decode(enc_result.bitstream, dec_cfg)

    snr = compute_snr(pcm, dec_result.pcm)
    sc = compute_spectral_convergence(pcm, dec_result.pcm)

    quality = QualityResult(
        signal_name=signal_name,
        snr_db=snr,
        spectral_convergence=sc,
        actual_bitrate_kbps=enc_result.actual_bitrate_kbps,
    )

    return quality, timings_all


def run_benchmark(
    signals: dict[str, dict],
    target_bitrate: float = 128.0,
    n_warmup: int = 1,
    n_runs: int = 10,
) -> BenchmarkResult:
    """Run the full benchmark suite across all test signals."""
    result = BenchmarkResult()
    result.hardware = _detect_hardware()
    if HAS_MLX:
        result.mlx_version = mx.__version__

    # Use the longest signal for stage benchmarks
    longest_name = max(signals, key=lambda k: len(signals[k]["pcm"]))
    longest = signals[longest_name]
    pcm = longest["pcm"]
    sr = longest["sample_rate"]

    window = get_window("kbd", 2048)
    frames = frame_signal(pcm, 2048, 1024, window)

    print(f"Stage benchmarks: {len(frames)} frames from '{longest_name}'")
    print(f"  {n_warmup} warmup + {n_runs} timed runs per backend\n")

    # Per-stage benchmarks
    for bench_fn in [
        benchmark_stage_mdct,
        benchmark_stage_psychoacoustic,
        benchmark_stage_imdct,
    ]:
        if bench_fn == benchmark_stage_psychoacoustic:
            stage_result = bench_fn(frames, sr, n_warmup, n_runs)
        else:
            stage_result = bench_fn(frames, n_warmup, n_runs)
        result.stages.append(stage_result)
        print(
            f"  {stage_result.stage:20s}  "
            f"CPU: {stage_result.cpu_time*1000:8.2f} ms  "
            f"GPU: {stage_result.gpu_time*1000:8.2f} ms  "
            f"Speedup: {stage_result.speedup:6.2f}x"
        )

    print("\nEnd-to-end benchmarks:")

    # Quality + e2e timing per signal
    all_enc_cpu_rtf = []
    all_enc_gpu_rtf = []
    all_dec_cpu_rtf = []
    all_dec_gpu_rtf = []

    for name, sig in signals.items():
        quality, timings = benchmark_end_to_end(
            sig["pcm"], name, sig["sample_rate"], target_bitrate
        )
        result.quality.append(quality)

        if "cpu" in timings:
            all_enc_cpu_rtf.append(timings["cpu"]["encode_rtf"])
            all_dec_cpu_rtf.append(timings["cpu"]["decode_rtf"])
        if "gpu" in timings:
            all_enc_gpu_rtf.append(timings["gpu"]["encode_rtf"])
            all_dec_gpu_rtf.append(timings["gpu"]["decode_rtf"])

        snr_str = f"{quality.snr_db:.1f}" if quality.snr_db != float("inf") else "inf"
        print(
            f"  {name:20s}  "
            f"SNR: {snr_str:>8s} dB  "
            f"SC: {quality.spectral_convergence:.4f}  "
            f"Bitrate: {quality.actual_bitrate_kbps:.1f} kbps"
        )

    if all_enc_cpu_rtf:
        result.encoder_cpu_rtf = float(np.mean(all_enc_cpu_rtf))
    if all_enc_gpu_rtf:
        result.encoder_gpu_rtf = float(np.mean(all_enc_gpu_rtf))
    if all_dec_cpu_rtf:
        result.decoder_cpu_rtf = float(np.mean(all_dec_cpu_rtf))
    if all_dec_gpu_rtf:
        result.decoder_gpu_rtf = float(np.mean(all_dec_gpu_rtf))

    print(f"\nEncoder RTF — CPU: {result.encoder_cpu_rtf:.4f}  GPU: {result.encoder_gpu_rtf:.4f}")
    print(f"Decoder RTF — CPU: {result.decoder_cpu_rtf:.4f}  GPU: {result.decoder_gpu_rtf:.4f}")

    return result
