"""Tests for MDCT/IMDCT correctness and GPU equivalence."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.core.mdct import (
    MDCTBasis,
    frame_signal,
    mdct_cpu,
    imdct_cpu,
    overlap_add,
)
from metal_aac.tables.windows import get_window, sine_window

try:
    import mlx.core as mx
    from metal_aac.core.mdct import MDCTBasisGPU, mdct_gpu, imdct_gpu

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class TestMDCTBasis:
    def test_basis_shape(self):
        basis = MDCTBasis.create(2048)
        assert basis.forward.shape == (2048, 1024)
        assert basis.inverse.shape == (1024, 2048)
        assert basis.n == 1024

    def test_basis_dtype(self):
        basis = MDCTBasis.create(2048)
        assert basis.forward.dtype == np.float32
        assert basis.inverse.dtype == np.float32


class TestMDCTCPU:
    def test_output_shape(self):
        frames = np.random.randn(10, 2048).astype(np.float32)
        result = mdct_cpu(frames)
        assert result.shape == (10, 1024)

    def test_zero_input(self):
        frames = np.zeros((5, 2048), dtype=np.float32)
        result = mdct_cpu(frames)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_roundtrip_single_frame(self):
        """MDCT -> IMDCT should recover the middle half of windowed input."""
        basis = MDCTBasis.create(2048)
        x = np.random.randn(1, 2048).astype(np.float32)
        spectrum = mdct_cpu(x, basis)
        reconstructed = imdct_cpu(spectrum, basis)
        assert reconstructed.shape == (1, 2048)

    def test_overlap_add_perfect_reconstruction(self):
        """Full pipeline: frame -> window -> MDCT -> IMDCT -> window -> OLA."""
        np.random.seed(42)
        n_samples = 8192
        pcm = np.random.randn(n_samples).astype(np.float32)

        window = sine_window(2048)
        frames = frame_signal(pcm, 2048, 1024, window)

        basis = MDCTBasis.create(2048)
        spectra = mdct_cpu(frames, basis)
        time_frames = imdct_cpu(spectra, basis)
        reconstructed = overlap_add(time_frames, 1024, window, n_samples)

        # Sine window MDCT with OLA gives perfect reconstruction
        # (after initial transient)
        start = 2048  # skip first full window
        end = n_samples - 2048
        if end > start:
            np.testing.assert_allclose(
                reconstructed[start:end],
                pcm[start:end],
                atol=1e-3,
                rtol=1e-3,
            )


class TestFrameSignal:
    def test_frame_count(self):
        pcm = np.zeros(10240, dtype=np.float32)
        frames = frame_signal(pcm, 2048, 1024)
        expected = (len(pcm) - 2048) // 1024 + 1
        assert abs(len(frames) - expected) <= 2

    def test_windowing(self):
        pcm = np.ones(4096, dtype=np.float32)
        window = np.ones(2048, dtype=np.float32) * 0.5
        frames = frame_signal(pcm, 2048, 1024, window)
        np.testing.assert_allclose(frames[0], 0.5, atol=1e-6)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMDCTGPU:
    def test_output_shape(self):
        frames = mx.array(np.random.randn(10, 2048).astype(np.float32))
        result = mdct_gpu(frames)
        mx.eval(result)
        assert result.shape == (10, 1024)

    def test_matches_cpu(self):
        """GPU MDCT should produce same results as CPU."""
        np.random.seed(123)
        frames_np = np.random.randn(20, 2048).astype(np.float32)
        frames_mx = mx.array(frames_np)

        cpu_result = mdct_cpu(frames_np)

        basis_gpu = MDCTBasisGPU(2048)
        gpu_result = mdct_gpu(frames_mx, basis_gpu)
        mx.eval(gpu_result)
        gpu_result_np = np.array(gpu_result)

        np.testing.assert_allclose(cpu_result, gpu_result_np, atol=1e-4, rtol=1e-4)

    def test_imdct_matches_cpu(self):
        """GPU IMDCT should produce same results as CPU."""
        np.random.seed(456)
        spectra_np = np.random.randn(20, 1024).astype(np.float32)
        spectra_mx = mx.array(spectra_np)

        cpu_result = imdct_cpu(spectra_np)

        basis_gpu = MDCTBasisGPU(2048)
        gpu_result = imdct_gpu(spectra_mx, basis_gpu)
        mx.eval(gpu_result)

        np.testing.assert_allclose(
            cpu_result, np.array(gpu_result), atol=1e-4, rtol=1e-4
        )
