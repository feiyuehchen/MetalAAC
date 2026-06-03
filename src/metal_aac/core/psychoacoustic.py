"""Simplified psychoacoustic model with CPU/GPU backends.

Computes masking thresholds per scalefactor band using:
1. Power spectrum via FFT
2. Critical band (Bark scale) energy aggregation
3. Spreading function convolution on Bark scale
4. Masking threshold from spread energy + absolute threshold

GPU strategy: steps 1-4 are all expressible as batch operations --
FFT, matrix multiply (band grouping and spreading), and element-wise ops.
The entire psychoacoustic model runs as a single GPU graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from metal_aac.tables.scalefactor_bands import get_sfb_offsets


def _bark(f: float) -> float:
    """Convert frequency (Hz) to Bark scale."""
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


@dataclass
class PsychoacousticTables:
    """Precomputed tables for the psychoacoustic model."""

    sfb_offsets: list[int]
    num_sfb: int
    # (num_fft_bins, num_sfb): maps FFT bins to SFB energies
    bin_to_sfb: np.ndarray
    # (num_bark_bands, num_bark_bands): spreading function matrix
    spreading_matrix: np.ndarray
    # (num_sfb,): absolute threshold of hearing per band (dB SPL)
    absolute_threshold: np.ndarray
    sample_rate: int
    fft_size: int

    @staticmethod
    def create(
        sample_rate: int = 44100,
        fft_size: int = 2048,
    ) -> PsychoacousticTables:
        sfb_offsets = get_sfb_offsets(sample_rate)
        num_sfb = len(sfb_offsets) - 1
        num_bins = fft_size // 2 + 1

        # Bin-to-SFB grouping matrix: sum FFT bin powers within each SFB
        bin_to_sfb = np.zeros((num_bins, num_sfb), dtype=np.float32)
        for b in range(num_sfb):
            lo = sfb_offsets[b]
            hi = sfb_offsets[b + 1]
            # SFB indices map to MDCT lines; FFT bins have similar grouping
            fft_lo = min(lo, num_bins - 1)
            fft_hi = min(hi, num_bins)
            if fft_lo < fft_hi:
                bin_to_sfb[fft_lo:fft_hi, b] = 1.0

        # Spreading function matrix on the SFB scale
        sfb_center_freqs = np.array(
            [
                (sfb_offsets[b] + sfb_offsets[b + 1])
                / 2
                * sample_rate
                / fft_size
                for b in range(num_sfb)
            ]
        )
        sfb_barks = np.array([_bark(f) for f in sfb_center_freqs])

        spreading = np.zeros((num_sfb, num_sfb), dtype=np.float64)
        for i in range(num_sfb):
            for j in range(num_sfb):
                dz = sfb_barks[i] - sfb_barks[j]
                if dz >= 0:
                    spread_db = -27.0 * dz
                else:
                    spread_db = (-6.0 + abs(dz)) * abs(dz)
                spread_db = max(spread_db, -60.0)
                spreading[i, j] = 10.0 ** (spread_db / 10.0)

        row_sums = spreading.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-20)
        spreading = (spreading / row_sums).astype(np.float32)

        # Absolute threshold of hearing (simplified, per SFB center freq)
        ath = np.zeros(num_sfb, dtype=np.float32)
        for b in range(num_sfb):
            f_khz = sfb_center_freqs[b] / 1000.0
            f_khz = max(f_khz, 0.02)
            # ISO 226 approximation
            ath_db = (
                3.64 * f_khz**-0.8
                - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
                + 1e-3 * f_khz**4
            )
            ath_db = min(ath_db, 96.0)
            ath[b] = 10.0 ** ((ath_db - 96.0) / 10.0)

        return PsychoacousticTables(
            sfb_offsets=sfb_offsets,
            num_sfb=num_sfb,
            bin_to_sfb=bin_to_sfb,
            spreading_matrix=spreading,
            absolute_threshold=ath,
            sample_rate=sample_rate,
            fft_size=fft_size,
        )


# ---- CPU implementation ----


def psychoacoustic_cpu(
    frames: np.ndarray,
    tables: PsychoacousticTables,
) -> np.ndarray:
    """Compute masking thresholds per SFB for a batch of frames.

    frames: (B, frame_size) windowed time-domain frames
    returns: (B, num_sfb) masking thresholds (linear scale)
    """
    # Power spectrum
    spectra = np.fft.rfft(frames, n=tables.fft_size, axis=-1)
    power = np.abs(spectra) ** 2 + 1e-20  # (B, num_bins)

    # Aggregate power into SFBs
    sfb_power = power @ tables.bin_to_sfb  # (B, num_sfb)

    # Apply spreading function
    spread_power = sfb_power @ tables.spreading_matrix.T  # (B, num_sfb)

    # Masking threshold = max(spread_power * tonality_offset, absolute_threshold)
    tonality_offset = 0.1  # simplified: 10 dB below signal for tonal masking
    mask_threshold = np.maximum(
        spread_power * tonality_offset,
        tables.absolute_threshold[None, :],
    )

    return mask_threshold.astype(np.float32)


# ---- GPU implementation ----


class PsychoacousticTablesGPU:
    """Psychoacoustic tables on GPU."""

    def __init__(self, tables: PsychoacousticTables):
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        self.bin_to_sfb = mx.array(tables.bin_to_sfb)
        self.spreading_matrix_t = mx.array(tables.spreading_matrix.T)
        self.absolute_threshold = mx.array(tables.absolute_threshold)
        self.fft_size = tables.fft_size
        self.num_sfb = tables.num_sfb


def psychoacoustic_gpu(
    frames: mx.array,
    tables: PsychoacousticTablesGPU,
) -> mx.array:
    """Compute masking thresholds on GPU via MLX.

    frames: (B, frame_size) -> (B, num_sfb)
    """
    # Power spectrum via FFT
    spectra = mx.fft.rfft(frames, n=tables.fft_size, axis=-1)
    power = mx.abs(spectra) ** 2 + 1e-20

    # Band energy aggregation via matmul
    sfb_power = mx.matmul(power, tables.bin_to_sfb)

    # Spreading function via matmul
    spread_power = mx.matmul(sfb_power, tables.spreading_matrix_t)

    # Masking threshold
    tonality_offset = 0.1
    mask_threshold = mx.maximum(
        spread_power * tonality_offset,
        tables.absolute_threshold[None, :],
    )

    return mask_threshold
