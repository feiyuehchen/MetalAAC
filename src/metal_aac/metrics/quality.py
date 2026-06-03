"""Audio quality metrics for round-trip evaluation.

Defined in BENCHMARK.md Part A.
"""

from __future__ import annotations

import numpy as np


def compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB.

    SNR = 10 * log10(sum(x^2) / sum((x - x_hat)^2))
    Higher is better. Returns -inf for zero signal.
    """
    min_len = min(len(original), len(reconstructed))
    x = original[:min_len].astype(np.float64)
    x_hat = reconstructed[:min_len].astype(np.float64)

    signal_power = np.sum(x**2)
    noise_power = np.sum((x - x_hat) ** 2)

    if signal_power < 1e-20:
        return float("-inf")
    if noise_power < 1e-20:
        return float("inf")

    return 10.0 * np.log10(signal_power / noise_power)


def compute_spectral_convergence(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> float:
    """Spectral Convergence: ||STFT(x) - STFT(x_hat)||_F / ||STFT(x)||_F.

    Lower is better. Uses Hann window.
    """
    min_len = min(len(original), len(reconstructed))
    x = original[:min_len].astype(np.float64)
    x_hat = reconstructed[:min_len].astype(np.float64)

    def _stft_mag(signal: np.ndarray) -> np.ndarray:
        window = np.hanning(n_fft)
        n_frames = (len(signal) - n_fft) // hop_length + 1
        if n_frames <= 0:
            return np.abs(np.fft.rfft(signal * window[:len(signal)]))[None, :]
        frames = np.zeros((n_frames, n_fft))
        for i in range(n_frames):
            start = i * hop_length
            frames[i] = signal[start : start + n_fft] * window
        return np.abs(np.fft.rfft(frames, axis=-1))

    mag_x = _stft_mag(x)
    mag_x_hat = _stft_mag(x_hat)

    # Align shapes
    min_frames = min(mag_x.shape[0], mag_x_hat.shape[0])
    mag_x = mag_x[:min_frames]
    mag_x_hat = mag_x_hat[:min_frames]

    norm_x = np.linalg.norm(mag_x)
    if norm_x < 1e-20:
        return float("inf")

    return float(np.linalg.norm(mag_x - mag_x_hat) / norm_x)
