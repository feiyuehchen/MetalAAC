"""Contract tests: verify metric implementations match BENCHMARK.md Part A."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.metrics.quality import compute_snr, compute_spectral_convergence
from metal_aac.metrics.throughput import (
    compute_frame_throughput,
    compute_gpu_speedup,
    compute_rtf,
    measure_peak_memory,
)


class TestQualityMetricSignatures:
    def test_snr_signature(self):
        sig = inspect.signature(compute_snr)
        params = list(sig.parameters)
        assert params == ["original", "reconstructed"]

    def test_spectral_convergence_signature(self):
        sig = inspect.signature(compute_spectral_convergence)
        params = list(sig.parameters)
        assert params == ["original", "reconstructed", "n_fft", "hop_length"]

    def test_spectral_convergence_defaults(self):
        sig = inspect.signature(compute_spectral_convergence)
        assert sig.parameters["n_fft"].default == 2048
        assert sig.parameters["hop_length"].default == 512


class TestThroughputMetricSignatures:
    def test_rtf_signature(self):
        sig = inspect.signature(compute_rtf)
        params = list(sig.parameters)
        assert params == ["processing_time_sec", "audio_duration_sec"]

    def test_frame_throughput_signature(self):
        sig = inspect.signature(compute_frame_throughput)
        params = list(sig.parameters)
        assert params == ["num_frames", "processing_time_sec"]

    def test_gpu_speedup_signature(self):
        sig = inspect.signature(compute_gpu_speedup)
        params = list(sig.parameters)
        assert params == ["cpu_time_sec", "gpu_time_sec"]


class TestSNRBehavior:
    def test_identical_signals(self):
        x = np.random.randn(1000).astype(np.float32)
        assert compute_snr(x, x) == float("inf")

    def test_zero_signal(self):
        x = np.zeros(1000, dtype=np.float32)
        x_hat = np.ones(1000, dtype=np.float32)
        assert compute_snr(x, x_hat) == float("-inf")

    def test_known_snr(self):
        x = np.ones(1000, dtype=np.float32)
        noise = np.ones(1000, dtype=np.float32) * 0.01
        x_hat = x + noise
        snr = compute_snr(x, x_hat)
        # SNR = 10*log10(1 / 0.01^2) = 10*log10(10000) = 40 dB
        assert abs(snr - 40.0) < 0.5

    def test_higher_noise_lower_snr(self):
        x = np.ones(1000, dtype=np.float32)
        snr_low_noise = compute_snr(x, x + 0.01)
        snr_high_noise = compute_snr(x, x + 0.1)
        assert snr_low_noise > snr_high_noise


class TestSpectralConvergence:
    def test_identical_signals(self):
        x = np.random.randn(4096).astype(np.float32)
        sc = compute_spectral_convergence(x, x)
        assert sc < 1e-5

    def test_different_signals(self):
        x = np.sin(2 * np.pi * 440 * np.arange(4096) / 44100).astype(np.float32)
        x_hat = np.sin(2 * np.pi * 880 * np.arange(4096) / 44100).astype(np.float32)
        sc = compute_spectral_convergence(x, x_hat)
        assert sc > 0.1


class TestRTF:
    def test_realtime(self):
        assert compute_rtf(0.5, 1.0) == 0.5

    def test_slower_than_realtime(self):
        assert compute_rtf(2.0, 1.0) == 2.0

    def test_zero_duration(self):
        assert compute_rtf(1.0, 0.0) == float("inf")


class TestGPUSpeedup:
    def test_faster_gpu(self):
        assert compute_gpu_speedup(2.0, 1.0) == 2.0

    def test_slower_gpu(self):
        assert compute_gpu_speedup(1.0, 2.0) == 0.5
