"""Simplified Huffman-style codebooks for AAC spectral data.

Full AAC defines 11 codebooks (ISO/IEC 14496-3 Table 4.A.2-4.A.12).
This implementation uses simplified exponential-Golomb coding with
section-based codebook selection, providing functional entropy coding
with measurable bitrate while keeping the implementation tractable.

The focus of this research is GPU optimization of the signal processing
stages (MDCT, psychoacoustic model, quantization), not entropy coding.
"""

from __future__ import annotations

import numpy as np


def _encode_signed_exp_golomb(value: int) -> tuple[int, int]:
    """Encode a signed integer using signed exp-Golomb (order 0).

    Mapping: 0→0, 1→1, -1→2, 2→3, -2→4, 3→5, -3→6, ...
    Then encode the mapped value with unsigned exp-Golomb.
    Returns (codeword_bits, num_bits).
    """
    if value > 0:
        code_num = 2 * value - 1
    elif value < 0:
        code_num = 2 * abs(value)
    else:
        code_num = 0

    # Unsigned exp-Golomb: write (code_num + 1) in binary,
    # preceded by floor(log2(code_num + 1)) zeros.
    v = code_num + 1
    m = v.bit_length() - 1  # floor(log2(v))
    total_bits = 2 * m + 1
    codeword = v  # v already starts with 1-bit, preceded by m zeros
    return codeword, total_bits


CODEBOOK_UNSIGNED_THRESHOLD = 8
CODEBOOK_ESCAPE_VALUE = 16


class ExpGolombCodebook:
    """Simplified codebook using exponential-Golomb coding.

    Groups quantized coefficients into pairs and encodes each value
    with signed exp-Golomb coding. Small values get short codes,
    large values get longer codes -- matching the statistical
    distribution of quantized MDCT coefficients.
    """

    @staticmethod
    def encode_pair(a: int, b: int) -> tuple[list[int], list[int]]:
        """Encode a pair of quantized values.

        Returns (codewords, bitlengths) as lists.
        """
        cw_a, bl_a = _encode_signed_exp_golomb(a)
        cw_b, bl_b = _encode_signed_exp_golomb(b)
        return [cw_a, cw_b], [bl_a, bl_b]

    @staticmethod
    def estimate_bits(values: np.ndarray) -> int:
        """Estimate total bits needed to encode an array of quantized values."""
        total = 0
        for v in values.flat:
            _, bits = _encode_signed_exp_golomb(int(v))
            total += bits
        return total

    @staticmethod
    def encode_section(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode a section of quantized MDCT coefficients.

        Returns (codewords, bitlengths) arrays.
        """
        n = len(values)
        if n % 2 != 0:
            values = np.append(values, 0)
            n += 1

        codewords = np.zeros(n, dtype=np.int64)
        bitlengths = np.zeros(n, dtype=np.int32)
        for i in range(n):
            cw, bl = _encode_signed_exp_golomb(int(values[i]))
            codewords[i] = cw
            bitlengths[i] = bl
        return codewords, bitlengths
