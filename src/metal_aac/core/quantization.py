"""AAC quantization with CPU/GPU backends.

AAC quantization maps MDCT coefficients to integer indices using:
    x_q = sign(x) * nint(|x|^(3/4) * 2^((global_gain - sf_offset) / 4))

where global_gain controls overall bit budget and sf_offset is per-band.

GPU strategy: the quantization formula itself is element-wise and fully
parallelizable. The rate-distortion search loop (finding optimal global_gain
and scalefactors) is sequential but each iteration's quantization is a
batch GPU operation. This is a natural hybrid: CPU drives the search loop,
GPU evaluates the quantization.
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


@dataclass
class QuantizationResult:
    quantized: np.ndarray  # (B, N) int32
    scalefactors: np.ndarray  # (B, num_sfb) int32
    global_gain: np.ndarray  # (B,) int32
    total_bits: np.ndarray  # (B,) int32


def _estimate_bits_per_value(val: int) -> int:
    """Estimate bits for a single quantized value (signed exp-Golomb)."""
    if val > 0:
        code_num = 2 * val - 1
    elif val < 0:
        code_num = 2 * abs(val)
    else:
        code_num = 0
    v = code_num + 1
    m = v.bit_length() - 1
    return 2 * m + 1


# ---- CPU implementation ----


def quantize_cpu(
    mdct_coeffs: np.ndarray,
    masking_thresholds: np.ndarray,
    target_bits_per_frame: int,
    sample_rate: int = 44100,
    max_iterations: int = 20,
) -> QuantizationResult:
    """Quantize MDCT coefficients with rate control.

    mdct_coeffs: (B, N)
    masking_thresholds: (B, num_sfb)
    target_bits_per_frame: target bitcount per frame
    returns: QuantizationResult
    """
    batch, n_coeffs = mdct_coeffs.shape
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb = len(sfb_offsets) - 1

    all_quantized = np.zeros((batch, n_coeffs), dtype=np.int32)
    all_scalefactors = np.zeros((batch, num_sfb), dtype=np.int32)
    all_global_gain = np.zeros(batch, dtype=np.int32)
    all_bits = np.zeros(batch, dtype=np.int32)

    for b in range(batch):
        coeffs = mdct_coeffs[b]
        thresholds = masking_thresholds[b]

        # Scalefactors control per-band quantization step size.
        # Higher SF → lower gain_factor → coarser quantization.
        # Bands with high SMR (signal >> mask) need FINE quantization
        # (low SF). Bands with low SMR can be coarser (high SF).
        scalefactors = np.zeros(num_sfb, dtype=np.int32)
        for sb in range(num_sfb):
            lo, hi = sfb_offsets[sb], sfb_offsets[sb + 1]
            band_power = np.mean(coeffs[lo:hi] ** 2) + 1e-20
            mask_power = max(float(thresholds[sb]), 1e-20)
            if np.isnan(mask_power) or np.isinf(mask_power):
                mask_power = band_power
            smr_db = 10.0 * np.log10(band_power / mask_power)
            if np.isnan(smr_db):
                smr_db = 0.0
            # Invert: high SMR → low SF (fine); low SMR → high SF (coarse)
            scalefactors[sb] = int(np.clip(60 - smr_db * 0.25, 0, 60))

        # Binary search for global_gain to meet bit budget.
        # Higher gain → finer quantization → more bits.
        # Find the highest gain that fits within the bit budget.
        gain_lo, gain_hi = 0, 255
        best_gain = 0
        best_quantized = np.zeros(n_coeffs, dtype=np.int32)
        best_bits = 0

        for _ in range(max_iterations):
            gain = (gain_lo + gain_hi) // 2
            quantized = _apply_quantization(coeffs, gain, scalefactors, sfb_offsets)
            bits = sum(_estimate_bits_per_value(int(v)) for v in quantized)

            if bits <= target_bits_per_frame:
                best_gain = gain
                best_quantized = quantized
                best_bits = bits
                gain_lo = gain + 1
            else:
                gain_hi = gain - 1

            if gain_lo > gain_hi:
                break

        all_quantized[b] = best_quantized
        all_scalefactors[b] = scalefactors
        all_global_gain[b] = best_gain
        all_bits[b] = best_bits

    return QuantizationResult(
        quantized=all_quantized,
        scalefactors=all_scalefactors,
        global_gain=all_global_gain,
        total_bits=all_bits,
    )


def _apply_quantization(
    coeffs: np.ndarray,
    global_gain: int,
    scalefactors: np.ndarray,
    sfb_offsets: list[int],
) -> np.ndarray:
    """Apply AAC quantization formula to MDCT coefficients."""
    n = len(coeffs)
    quantized = np.zeros(n, dtype=np.int32)
    num_sfb = len(scalefactors)

    for sb in range(num_sfb):
        lo, hi = sfb_offsets[sb], sfb_offsets[sb + 1]
        sf_offset = scalefactors[sb]
        gain_factor = 2.0 ** ((global_gain - sf_offset * 4) / 16.0)

        band = coeffs[lo:hi]
        abs_band = np.abs(band)
        powered = np.power(abs_band + 1e-20, 0.75)
        q = np.sign(band) * np.floor(powered * gain_factor + 0.4054)
        quantized[lo:hi] = q.astype(np.int32)

    return quantized


def dequantize_cpu(
    quantized: np.ndarray,
    scalefactors: np.ndarray,
    global_gain: np.ndarray,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Inverse quantization: integer indices -> MDCT coefficients.

    quantized: (B, N) int32
    scalefactors: (B, num_sfb) int32
    global_gain: (B,) int32
    returns: (B, N) float32
    """
    batch, n_coeffs = quantized.shape
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb = len(sfb_offsets) - 1
    output = np.zeros((batch, n_coeffs), dtype=np.float32)

    for b in range(batch):
        for sb in range(num_sfb):
            lo, hi = sfb_offsets[sb], sfb_offsets[sb + 1]
            sf_offset = scalefactors[b, sb]
            gain_factor = 2.0 ** ((global_gain[b] - sf_offset * 4) / 16.0)

            q = quantized[b, lo:hi].astype(np.float32)
            signs = np.sign(q)
            abs_q = np.abs(q)
            # Inverse: x = sign * |q|^(4/3) / gain_factor^(4/3)
            reconstructed = signs * np.power(abs_q, 4.0 / 3.0) / (
                gain_factor ** (4.0 / 3.0) + 1e-20
            )
            output[b, lo:hi] = reconstructed

    return output


# ---- GPU implementation ----


def quantize_frame_gpu(
    coeffs: mx.array,
    global_gain: int,
    scalefactors: mx.array,
    sfb_starts: mx.array,
    sfb_ends: mx.array,
    sf_gains: mx.array,
) -> mx.array:
    """Apply quantization formula on GPU for a single gain value.

    This is the inner loop of rate control -- the formula application
    is fully parallelizable and benefits from GPU execution.

    coeffs: (B, N)
    sf_gains: (N,) precomputed per-coefficient gain factor
    returns: (B, N) quantized int values
    """
    abs_coeffs = mx.abs(coeffs)
    powered = mx.power(abs_coeffs + 1e-20, 0.75)
    q = mx.sign(coeffs) * mx.floor(powered * sf_gains[None, :] + 0.4054)
    return q.astype(mx.int32)


def dequantize_gpu(
    quantized: mx.array,
    sf_inv_gains: mx.array,
) -> mx.array:
    """Inverse quantization on GPU.

    quantized: (B, N) int32
    sf_inv_gains: (N,) precomputed inverse gain per coefficient
    returns: (B, N) float32
    """
    q = quantized.astype(mx.float32)
    signs = mx.sign(q)
    abs_q = mx.abs(q)
    reconstructed = signs * mx.power(abs_q, 4.0 / 3.0) * sf_inv_gains[None, :]
    return reconstructed


def precompute_sf_gains(
    global_gain: int,
    scalefactors: np.ndarray,
    sfb_offsets: list[int],
    n_coeffs: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-coefficient gain factors for GPU quantization.

    Returns (forward_gains, inverse_gains) each of shape (N,).
    """
    forward = np.ones(n_coeffs, dtype=np.float32)
    inverse = np.ones(n_coeffs, dtype=np.float32)
    num_sfb = len(scalefactors)

    for sb in range(num_sfb):
        lo, hi = sfb_offsets[sb], sfb_offsets[sb + 1]
        sf_offset = scalefactors[sb]
        gain = 2.0 ** ((global_gain - sf_offset * 4) / 16.0)
        forward[lo:hi] = gain
        inverse[lo:hi] = 1.0 / (gain ** (4.0 / 3.0) + 1e-20)

    return forward, inverse


# ---- Fully batched GPU quantization with vectorized rate control ----


def _build_sfb_map(sfb_offsets: list[int], n_coeffs: int) -> np.ndarray:
    """Map each coefficient index to its SFB index."""
    sfb_map = np.zeros(n_coeffs, dtype=np.int32)
    for sb in range(len(sfb_offsets) - 1):
        sfb_map[sfb_offsets[sb] : sfb_offsets[sb + 1]] = sb
    return sfb_map


def _gpu_estimate_bits(q: mx.array) -> mx.array:
    """Estimate exp-Golomb bit cost for all quantized values on GPU.

    q: (B, N) int32
    returns: (B,) total bits per frame
    """
    abs_q = mx.abs(q.astype(mx.float32))
    code_num = mx.where(
        q > 0,
        2 * abs_q - 1,
        mx.where(q < 0, 2 * abs_q, mx.zeros_like(abs_q)),
    )
    v = code_num + 1
    m = mx.floor(mx.log2(mx.maximum(v, 1.0) + 0.5))
    bit_lengths = 2 * m + 1
    return mx.sum(bit_lengths, axis=-1)


def quantize_batch_gpu(
    mdct_coeffs: mx.array,
    masking_thresholds: mx.array,
    target_bits_per_frame: int,
    sample_rate: int = 44100,
    max_iterations: int = 8,
) -> QuantizationResult:
    """Fully GPU-vectorized quantization with rate control.

    Instead of looping over each frame in Python, this function:
    1. Computes scalefactors for ALL frames simultaneously on GPU
    2. Runs the binary search with ALL frames evaluated in one GPU call per iteration
    3. Returns results transferred back to CPU

    mdct_coeffs: (B, N) on GPU
    masking_thresholds: (B, num_sfb) on GPU
    """
    batch, n_coeffs = mdct_coeffs.shape
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb = len(sfb_offsets) - 1
    sfb_map = _build_sfb_map(sfb_offsets, n_coeffs)
    sfb_map_mx = mx.array(sfb_map)

    # Step 1: compute band power and scalefactors for all frames at once
    band_powers = mx.zeros((batch, num_sfb))
    for sb in range(num_sfb):
        lo, hi = sfb_offsets[sb], sfb_offsets[sb + 1]
        band_powers = band_powers.at[:, sb].add(
            mx.mean(mdct_coeffs[:, lo:hi] ** 2, axis=-1) + 1e-20
        )

    masking_safe = mx.maximum(masking_thresholds, 1e-20)
    masking_safe = mx.where(
        mx.isnan(masking_safe), band_powers, masking_safe
    )
    smr_db = 10.0 * mx.log10(band_powers / masking_safe)
    smr_db = mx.where(mx.isnan(smr_db), mx.zeros_like(smr_db), smr_db)
    scalefactors = mx.clip(60 - smr_db * 0.25, 0, 60).astype(mx.int32)
    mx.eval(scalefactors)

    # Step 2: vectorized binary search across ALL frames
    per_coeff_sf = scalefactors[:, sfb_map_mx]  # (B, N)
    abs_coeffs = mx.abs(mdct_coeffs)
    powered = mx.power(abs_coeffs + 1e-20, 0.75)
    mx.eval(per_coeff_sf, powered)

    gain_lo = mx.zeros(batch, dtype=mx.int32)
    gain_hi = mx.full((batch,), 255, dtype=mx.int32)
    best_gains = mx.zeros(batch, dtype=mx.int32)
    best_bits = mx.zeros(batch, dtype=mx.float32)
    target = mx.array(target_bits_per_frame, dtype=mx.float32)

    for _ in range(max_iterations):
        gains = (gain_lo + gain_hi) // 2

        gain_factor = mx.power(
            2.0, (gains[:, None].astype(mx.float32) - per_coeff_sf * 4) / 16.0
        )
        q = mx.sign(mdct_coeffs) * mx.floor(powered * gain_factor + 0.4054)

        bits = _gpu_estimate_bits(q.astype(mx.int32))

        fits = bits <= target
        best_gains = mx.where(fits, gains, best_gains)
        best_bits = mx.where(fits, bits, best_bits)
        gain_lo = mx.where(fits, gains + 1, gain_lo)
        gain_hi = mx.where(~fits, gains - 1, gain_hi)
        mx.eval(gain_lo, gain_hi, best_gains, best_bits)

    # Step 3: final quantization with best gains
    final_gf = mx.power(
        2.0,
        (best_gains[:, None].astype(mx.float32) - per_coeff_sf * 4) / 16.0,
    )
    final_q = mx.sign(mdct_coeffs) * mx.floor(powered * final_gf + 0.4054)
    final_q = final_q.astype(mx.int32)
    mx.eval(final_q)

    return QuantizationResult(
        quantized=np.array(final_q),
        scalefactors=np.array(scalefactors),
        global_gain=np.array(best_gains),
        total_bits=np.array(best_bits, dtype=np.int32),
    )
