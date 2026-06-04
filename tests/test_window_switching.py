"""Tests for window switching and transient detection."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.core.window_switching import (
    WindowSequence,
    compute_window_sequences,
    detect_transients_cpu,
)
from metal_aac.core.mdct import MDCTBasisGPU, mdct_short_gpu
from metal_aac.tables.windows import (
    get_window_for_sequence,
    long_start_window,
    long_stop_window,
)

try:
    import mlx.core as mx
    from metal_aac.core.window_switching import detect_transients_gpu

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class TestTransientDetection:
    def test_steady_signal_no_transient(self):
        frames = np.sin(2 * np.pi * 440 * np.arange(10 * 2048).reshape(10, 2048) / 44100).astype(np.float32)
        transients = detect_transients_cpu(frames)
        assert not np.any(transients)

    def test_impulse_detected(self):
        frames = np.zeros((5, 2048), dtype=np.float32)
        frames[2, 1500] = 1.0  # impulse in second half of frame 2
        transients = detect_transients_cpu(frames, threshold=2.0)
        assert transients[2]

    def test_silence_no_transient(self):
        frames = np.zeros((3, 2048), dtype=np.float32)
        transients = detect_transients_cpu(frames)
        assert not np.any(transients)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_gpu_matches_cpu(self):
        np.random.seed(42)
        frames = np.random.randn(20, 2048).astype(np.float32)
        frames[5, 1024:] *= 10  # create a transient
        cpu_result = detect_transients_cpu(frames)
        gpu_result = np.array(detect_transients_gpu(mx.array(frames)))
        np.testing.assert_array_equal(cpu_result, gpu_result)


class TestWindowSequenceStateMachine:
    def test_all_steady(self):
        transients = np.array([False] * 10)
        seqs = compute_window_sequences(transients)
        assert np.all(seqs == WindowSequence.ONLY_LONG)

    def test_single_transient(self):
        transients = np.array([False, False, True, False, False, False])
        seqs = compute_window_sequences(transients)
        # Frame before transient should start transitioning (look-ahead)
        assert seqs[0] == WindowSequence.ONLY_LONG
        # Transient frame and surroundings
        assert WindowSequence.EIGHT_SHORT in seqs
        # Eventually returns to ONLY_LONG
        assert seqs[-1] == WindowSequence.ONLY_LONG

    def test_sequence_transitions(self):
        transients = np.array([False, True, False, False])
        seqs = compute_window_sequences(transients)
        # Must go through LONG_START before EIGHT_SHORT
        found_start = False
        found_short = False
        for s in seqs:
            if s == WindowSequence.LONG_START:
                found_start = True
            if s == WindowSequence.EIGHT_SHORT:
                found_short = True
                assert found_start, "EIGHT_SHORT without prior LONG_START"


class TestTransitionWindows:
    def test_long_start_shape(self):
        w = long_start_window(2048, 256)
        assert len(w) == 2048
        assert w[0] < 0.01  # rising edge starts near zero
        assert abs(w[512] - 1.0) < 0.01 or w[512] > 0.5  # rises

    def test_long_stop_shape(self):
        w = long_stop_window(2048, 256)
        assert len(w) == 2048
        assert abs(w[1536] - 1.0) < 0.01 or w[1536] > 0.5

    def test_get_window_for_all_sequences(self):
        for seq in range(4):
            w = get_window_for_sequence(seq)
            if seq == 2:  # short
                assert len(w) == 256
            else:
                assert len(w) == 2048


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestShortMDCT:
    def test_output_shape(self):
        frames = mx.array(np.random.randn(5, 2048).astype(np.float32))
        # Reshape to 8 short sub-frames per frame
        sub = mx.reshape(frames, (5, 8, 256))
        sub_flat = mx.reshape(sub, (40, 256))

        basis = MDCTBasisGPU(256)
        result = mdct_short_gpu(frames, basis)
        mx.eval(result)
        assert result.shape == (5, 8, 128)

    def test_total_coefficients(self):
        frames = mx.array(np.random.randn(3, 2048).astype(np.float32))
        result = mdct_short_gpu(frames)
        mx.eval(result)
        # 8 windows × 128 coefficients = 1024 total per frame (same as long)
        assert result.shape[1] * result.shape[2] == 1024
