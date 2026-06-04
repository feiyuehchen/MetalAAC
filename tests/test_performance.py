"""Performance regression tests.

Each test asserts a maximum wall-clock time for a given stage at a
fixed input size. Thresholds are set at 3x the current best measurement
on M3 Pro to allow headroom for CI variance while catching major regressions.

These tests also serve as documentation: the assert value IS the
performance target for that stage.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.encoder import EncoderConfig, encode
from metal_aac.decoder import DecoderConfig, decode
from metal_aac.core.mdct import (
    MDCTBasis, MDCTBasisGPU, frame_signal, frame_signal_mlx,
    mdct_cpu, mdct_gpu, imdct_cpu, imdct_gpu, mdct_short_gpu,
)
from metal_aac.core.psychoacoustic import (
    PsychoacousticTables, PsychoacousticTablesGPU,
    psychoacoustic_cpu, psychoacoustic_gpu,
)
from metal_aac.core.window_switching import detect_transients_cpu
from metal_aac.tables.windows import get_window
from metal_aac.metrics.throughput import Timer

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

SR = 44100
FRAME_SIZE = 2048
N_FRAMES_60S = 2584  # 60s of audio


def _make_pcm(duration: float = 60.0) -> np.ndarray:
    n = int(SR * duration)
    t = np.arange(n) / SR
    return (0.7 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _make_frames(duration: float = 60.0) -> np.ndarray:
    pcm = _make_pcm(duration)
    window = get_window("kbd", FRAME_SIZE)
    return frame_signal(pcm, FRAME_SIZE, 1024, window)


def _warmup_and_time(func, n_runs=5):
    """Run func once for warmup, then n_runs times, return best time."""
    func()
    times = []
    for _ in range(n_runs):
        with Timer() as t:
            func()
        times.append(t.elapsed)
    return min(times)


# ---- Per-stage timing tests (60s audio, ~2584 frames) ----


class TestFramingPerformance:
    def test_numpy_framing_60s(self):
        pcm = _make_pcm(60.0)
        window = get_window("kbd", FRAME_SIZE)
        elapsed = _warmup_and_time(
            lambda: frame_signal(pcm, FRAME_SIZE, 1024, window))
        assert elapsed < 0.1, f"numpy framing took {elapsed*1000:.1f}ms (limit 100ms)"

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_mlx_framing_60s(self):
        pcm_mx = mx.array(_make_pcm(60.0))
        window_mx = mx.array(get_window("kbd", FRAME_SIZE))
        mx.eval(pcm_mx, window_mx)

        def run():
            f = frame_signal_mlx(pcm_mx, FRAME_SIZE, 1024, window_mx)
            mx.eval(f)
        elapsed = _warmup_and_time(run)
        assert elapsed < 0.03, f"MLX framing took {elapsed*1000:.1f}ms (limit 30ms)"


class TestMDCTPerformance:
    def test_mdct_cpu_60s(self):
        frames = _make_frames(60.0)
        basis = MDCTBasis.create(FRAME_SIZE)
        elapsed = _warmup_and_time(lambda: mdct_cpu(frames, basis))
        assert elapsed < 0.05, f"CPU MDCT took {elapsed*1000:.1f}ms (limit 50ms)"

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_mdct_gpu_60s(self):
        frames_mx = mx.array(_make_frames(60.0))
        basis = MDCTBasisGPU(FRAME_SIZE)
        mx.eval(frames_mx)

        def run():
            r = mdct_gpu(frames_mx, basis)
            mx.eval(r)
        elapsed = _warmup_and_time(run)
        assert elapsed < 0.03, f"GPU MDCT took {elapsed*1000:.1f}ms (limit 30ms)"

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_mdct_short_gpu_60s(self):
        frames_mx = mx.array(_make_frames(60.0))
        basis = MDCTBasisGPU(256)
        mx.eval(frames_mx)

        def run():
            r = mdct_short_gpu(frames_mx, basis)
            mx.eval(r)
        elapsed = _warmup_and_time(run)
        assert elapsed < 0.05, f"Short MDCT GPU took {elapsed*1000:.1f}ms (limit 50ms)"


class TestPsychoacousticPerformance:
    def test_psychoacoustic_cpu_60s(self):
        frames = _make_frames(60.0)
        tables = PsychoacousticTables.create(SR, FRAME_SIZE)
        elapsed = _warmup_and_time(lambda: psychoacoustic_cpu(frames, tables))
        assert elapsed < 0.1, f"CPU psychoacoustic took {elapsed*1000:.1f}ms (limit 100ms)"

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_psychoacoustic_gpu_60s(self):
        frames_mx = mx.array(_make_frames(60.0))
        tables = PsychoacousticTablesGPU(PsychoacousticTables.create(SR, FRAME_SIZE))
        mx.eval(frames_mx)

        def run():
            r = psychoacoustic_gpu(frames_mx, tables)
            mx.eval(r)
        elapsed = _warmup_and_time(run)
        assert elapsed < 0.015, f"GPU psychoacoustic took {elapsed*1000:.1f}ms (limit 15ms)"


class TestTransientDetectPerformance:
    def test_transient_detect_cpu_60s(self):
        pcm = _make_pcm(60.0)
        raw_frames = frame_signal(pcm, FRAME_SIZE, 1024)
        elapsed = _warmup_and_time(lambda: detect_transients_cpu(raw_frames))
        assert elapsed < 0.01, f"CPU transient detect took {elapsed*1000:.1f}ms (limit 10ms)"


# ---- End-to-end timing tests ----


class TestEncoderPerformance:
    def test_cpu_encoder_10s(self):
        pcm = _make_pcm(10.0)
        encode(pcm[:SR], EncoderConfig(use_gpu=False))  # warmup
        elapsed = _warmup_and_time(
            lambda: encode(pcm, EncoderConfig(use_gpu=False)), n_runs=3)
        assert elapsed < 5.0, f"CPU encoder 10s took {elapsed:.2f}s (limit 5s)"

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_gpu_encoder_10s(self):
        pcm = _make_pcm(10.0)
        encode(pcm[:SR], EncoderConfig(use_gpu=True))  # warmup
        elapsed = _warmup_and_time(
            lambda: encode(pcm, EncoderConfig(use_gpu=True)), n_runs=3)
        assert elapsed < 0.15, f"GPU encoder 10s took {elapsed*1000:.1f}ms (limit 150ms)"

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_gpu_encoder_60s(self):
        pcm = _make_pcm(60.0)
        encode(pcm[:SR], EncoderConfig(use_gpu=True))
        elapsed = _warmup_and_time(
            lambda: encode(pcm, EncoderConfig(use_gpu=True)), n_runs=3)
        assert elapsed < 0.3, f"GPU encoder 60s took {elapsed*1000:.1f}ms (limit 300ms)"


class TestDecoderPerformance:
    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_gpu_roundtrip_10s(self):
        pcm = _make_pcm(10.0)
        enc = encode(pcm, EncoderConfig(use_gpu=True))
        decode(enc.bitstream, DecoderConfig(use_gpu=True, output_length=len(pcm)))

        def run():
            decode(enc.bitstream, DecoderConfig(use_gpu=True, output_length=len(pcm)))
        elapsed = _warmup_and_time(run, n_runs=3)
        assert elapsed < 0.5, f"GPU decode 10s took {elapsed*1000:.1f}ms (limit 500ms)"
