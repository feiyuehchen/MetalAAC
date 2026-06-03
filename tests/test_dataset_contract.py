"""Contract tests: verify code matches DATASET.md v1 specification."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.data.synthetic import generate_test_signals


EXPECTED_SIGNALS = [
    "sine_440",
    "sine_1k",
    "sine_4k",
    "multitone",
    "chirp",
    "white_noise",
    "pink_noise",
    "silence",
    "impulse",
    "long_music",
]


class TestDatasetContract:
    @pytest.fixture(scope="class")
    def signals(self):
        return generate_test_signals()

    def test_all_signals_present(self, signals):
        for name in EXPECTED_SIGNALS:
            assert name in signals, f"Missing signal: {name}"

    def test_signal_sample_rate(self, signals):
        for name in EXPECTED_SIGNALS:
            assert signals[name]["sample_rate"] == 44100

    def test_signal_dtype(self, signals):
        for name in EXPECTED_SIGNALS:
            assert signals[name]["pcm"].dtype == np.float32

    def test_signal_mono(self, signals):
        for name in EXPECTED_SIGNALS:
            assert signals[name]["pcm"].ndim == 1

    def test_signal_durations(self, signals):
        expected_durations = {
            "sine_440": 1.0,
            "sine_1k": 1.0,
            "sine_4k": 1.0,
            "multitone": 1.0,
            "chirp": 2.0,
            "white_noise": 1.0,
            "pink_noise": 1.0,
            "silence": 0.5,
            "impulse": 1.0,
            "long_music": 10.0,
        }
        for name, dur in expected_durations.items():
            actual_dur = len(signals[name]["pcm"]) / signals[name]["sample_rate"]
            assert abs(actual_dur - dur) < 0.01, (
                f"{name}: expected {dur}s, got {actual_dur}s"
            )

    def test_normalized_range(self, signals):
        for name in EXPECTED_SIGNALS:
            pcm = signals[name]["pcm"]
            assert np.all(pcm >= -1.0) and np.all(pcm <= 1.0), (
                f"{name} outside [-1, 1]"
            )

    def test_deterministic(self, signals):
        signals2 = generate_test_signals()
        for name in EXPECTED_SIGNALS:
            np.testing.assert_array_equal(
                signals[name]["pcm"],
                signals2[name]["pcm"],
                err_msg=f"{name} not deterministic",
            )

    def test_silence_is_zero(self, signals):
        np.testing.assert_array_equal(
            signals["silence"]["pcm"],
            np.zeros_like(signals["silence"]["pcm"]),
        )

    def test_impulse_single_nonzero(self, signals):
        imp = signals["impulse"]["pcm"]
        nonzero = np.nonzero(imp)[0]
        assert len(nonzero) == 1
        assert imp[nonzero[0]] == 1.0
