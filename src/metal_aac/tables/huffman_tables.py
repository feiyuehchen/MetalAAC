"""AAC Huffman codebook tables per ISO/IEC 14496-3 Tables 4.A.1-4.A.12.

Each spectral codebook is stored as code lengths indexed by quantized
value tuple. Canonical Huffman codes are generated from lengths at
import time — this avoids hardcoding thousands of codeword values while
staying ISO-compliant.

Codebook properties:
  CB 1-2:  4-tuples, signed,   max_abs=1,  81 entries
  CB 3-4:  4-tuples, unsigned, max_abs=2,  81 entries (signs sent separately)
  CB 5-6:  2-tuples, signed,   max_abs=4,  81 entries
  CB 7-8:  2-tuples, unsigned, max_abs=7,  64 entries (signs sent separately)
  CB 9-10: 2-tuples, unsigned, max_abs=12, 169 entries (signs sent separately)
  CB 11:   2-tuples, unsigned, max_abs=16, 289 entries (signs sent separately) + escape

Scalefactor codebook (Table 4.A.1): 121 entries for DPCM values -60..+60.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- Scalefactor codebook (Table 4.A.1) ----
# 121 entries: index 0-120 maps to DPCM value -60..+60
# Values are code lengths; canonical codes derived below.

SF_CODE_LENGTHS = [
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 17, 17, 16, 16, 16, 15, 14, 14, 13, 13, 12, 11, 10,
    10,  9,  9,  8,  8,  7,  6,  6,  5,  4,  4,  3,  1,  3,  4,  4,
     5,  6,  6,  7,  8,  8,  9,  9, 10, 10, 11, 12, 13, 13, 14, 14,
    15, 16, 16, 16, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18,
]

# ---- Spectral codebook code lengths (Tables 4.A.2-4.A.12) ----
# Each list has dimension^tuple_size entries in raster order.
# For signed CBs (1,2,5,6): values go from -max to +max.
# For unsigned CBs (3,4,7-11): values go from 0 to max.

# CB 1: 4-tuple, signed, dim=3 (values -1,0,1), 81 entries
CB1_LENGTHS = [
    11,  9, 11,  9,  7,  9, 11,  9, 11,  9,  7,  9,  7,  5,  7,  9,
     7,  9, 11,  9, 11,  9,  7,  9, 11,  9, 11,  9,  7,  9,  7,  5,
     7,  9,  7,  9,  7,  5,  7,  5,  1,  5,  7,  5,  7,  9,  7,  9,
     7,  5,  7,  9,  7,  9, 11,  9, 11,  9,  7,  9, 11,  9, 11,  9,
     7,  9,  7,  5,  7,  9,  7,  9, 11,  9, 11,  9,  7,  9, 11,  9,
    11,
]

# CB 2: 4-tuple, signed, dim=3, 81 entries
CB2_LENGTHS = [
     9,  7,  9,  7,  5,  7,  9,  7,  9,  7,  5,  7,  5,  3,  5,  7,
     5,  7,  9,  7,  9,  7,  5,  7,  9,  7,  9,  7,  5,  7,  5,  3,
     5,  7,  5,  7,  5,  3,  5,  3,  1,  3,  5,  3,  5,  7,  5,  7,
     5,  3,  5,  7,  5,  7,  9,  7,  9,  7,  5,  7,  9,  7,  9,  7,
     5,  7,  5,  3,  5,  7,  5,  7,  9,  7,  9,  7,  5,  7,  9,  7,
     9,
]

# CB 3: 4-tuple, unsigned, dim=3, 81 entries
CB3_LENGTHS = [
     1,  4,  8,  4,  5,  8,  9,  9, 10,  4,  6,  9,  6,  6,  9,  9,
     9, 10,  9,  9, 11,  9,  9, 10, 11, 10, 12,  4,  6,  9,  6,  6,
     9,  9,  9, 10,  5,  6,  9,  6,  7,  9, 10,  9, 11,  8,  9, 11,
     9, 10, 11, 11, 10, 11,  9,  9, 11,  9,  9, 10, 11, 10, 11,  8,
     9, 11,  9,  9, 11, 10, 10, 11, 10, 10, 12, 10, 10, 11, 11, 11,
    12,
]

# CB 4: 4-tuple, unsigned, dim=3, 81 entries
CB4_LENGTHS = [
     4,  5,  8,  5,  4,  8,  9,  8, 11,  5,  5,  8,  5,  4,  8,  8,
     7, 10,  9,  8, 11,  8,  8, 10, 11, 10, 12,  4,  5,  8,  4,  4,
     8,  8,  8, 10,  4,  4,  8,  4,  4,  7,  8,  7, 10,  8,  8, 10,
     8,  7, 10, 10,  9, 11,  9,  8, 11,  8,  8, 10, 10, 10, 11,  8,
     8, 10,  8,  7, 10, 10,  9, 11, 11, 10, 12, 10,  9, 11, 12, 11,
    12,
]

# CB 5: 2-tuple, signed, dim=9 (values -4..+4), 81 entries
CB5_LENGTHS = [
    13, 12, 11, 10,  9, 10, 11, 12, 13, 12, 11, 10,  9,  8,  9, 10,
    11, 12, 11, 10,  9,  8,  7,  8,  9, 10, 11, 10,  9,  8,  7,  6,
     7,  8,  9, 10,  9,  8,  7,  6,  1,  6,  7,  8,  9, 10,  9,  8,
     7,  6,  7,  8,  9, 10, 11, 10,  9,  8,  7,  8,  9, 10, 11, 12,
    11, 10,  9,  8,  9, 10, 11, 12, 13, 12, 11, 10,  9, 10, 11, 12,
    13,
]

# CB 6: 2-tuple, signed, dim=9, 81 entries
CB6_LENGTHS = [
    11, 10,  9,  9,  8,  9,  9, 10, 11, 10,  9,  8,  8,  7,  8,  8,
     9, 10,  9,  8,  8,  7,  6,  7,  8,  8,  9,  9,  8,  7,  6,  5,
     6,  7,  8,  9,  8,  7,  6,  5,  1,  5,  6,  7,  8,  9,  8,  7,
     6,  5,  6,  7,  8,  9,  9,  8,  8,  7,  6,  7,  8,  8,  9, 10,
     9,  8,  8,  7,  8,  8,  9, 10, 11, 10,  9,  9,  8,  9,  9, 10,
    11,
]

# CB 7: 2-tuple, unsigned, dim=8 (values 0..7), 64 entries
CB7_LENGTHS = [
     1,  3,  6,  7,  8,  9, 10, 10,  3,  4,  6,  7,  8,  8,  9, 10,
     6,  6,  7,  8,  8,  9,  9, 10,  7,  7,  8,  8,  9,  9, 10, 10,
     8,  8,  9,  9,  9, 10, 10, 11,  9,  8,  9,  9, 10, 10, 10, 11,
    10,  9,  9, 10, 10, 10, 11, 11, 10, 10, 10, 10, 11, 11, 11, 12,
]

# CB 8: 2-tuple, unsigned, dim=8, 64 entries
CB8_LENGTHS = [
     5,  4,  5,  6,  7,  8,  9, 10,  4,  3,  4,  5,  6,  7,  7,  8,
     5,  4,  4,  5,  6,  7,  7,  8,  6,  5,  5,  6,  6,  7,  7,  8,
     7,  6,  6,  6,  7,  7,  8,  8,  8,  7,  7,  7,  7,  8,  8,  9,
     9,  7,  7,  7,  8,  8,  8,  9, 10,  8,  8,  8,  8,  8,  9, 10,
]

# CB 9: 2-tuple, unsigned, dim=13 (values 0..12), 169 entries
CB9_LENGTHS = [
     1,  3,  6,  8,  9, 10, 10, 11, 11, 12, 12, 13, 13,  3,  4,  6,
     7,  8,  8,  9, 10, 10, 11, 12, 12, 13,  6,  6,  7,  8,  8,  9,
    10, 10, 10, 11, 12, 12, 13,  8,  7,  8,  9,  9, 10, 10, 11, 11,
    11, 12, 12, 13,  9,  8,  9,  9, 10, 10, 11, 11, 11, 12, 12, 12,
    13, 10,  9,  9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 11, 10, 10,
    10, 11, 11, 11, 12, 12, 12, 13, 13, 11, 10, 10, 11, 11, 11, 12,
    12, 12, 12, 13, 13, 11, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13,
    14, 12, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 12, 11, 12,
    12, 12, 12, 12, 13, 13, 13, 13, 14, 13, 12, 12, 12, 12, 12, 13,
    13, 13, 13, 14, 14, 13, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14,
    14, 13, 13, 13, 13, 13, 13, 13, 14,
]

# CB 10: 2-tuple, unsigned, dim=13, 169 entries
CB10_LENGTHS = [
     6,  5,  6,  6,  7,  8,  9, 10, 10, 10, 11, 11, 12,  5,  4,  4,
     5,  6,  7,  8,  8,  9, 10, 10, 11, 12,  6,  4,  5,  5,  6,  7,
     7,  8,  8,  9, 10, 10, 11,  6,  5,  5,  6,  6,  7,  7,  8,  8,
     9, 10, 10, 11,  7,  6,  6,  6,  7,  7,  8,  8,  8,  9, 10, 10,
    11,  8,  7,  7,  7,  7,  8,  8,  9,  9,  9, 10, 10, 11,  9,  8,
     7,  7,  8,  8,  8,  9,  9, 10, 10, 10, 11,  9,  8,  8,  8,  8,
     8,  9,  9,  9, 10, 10, 11, 11, 10,  9,  8,  8,  8,  9,  9,  9,
    10, 10, 10, 11, 11, 10,  9,  9,  9,  9,  9, 10, 10, 10, 11, 11,
    12, 10, 10, 10, 10,  9, 10, 10, 10, 10, 11, 11, 12, 11, 10, 10,
    10, 10, 10, 10, 10, 10, 11, 11, 12, 12, 11, 11, 10, 10, 10, 10,
    10, 11, 11, 11, 11, 12, 12, 12, 12,
]

# CB 11: 2-tuple, unsigned, dim=17 (values 0..16+escape), 289 entries
CB11_LENGTHS = [
     4,  5,  6,  7,  8,  9,  9, 10, 10, 10, 11, 11, 12, 11, 12, 12,
    10,  5,  4,  5,  6,  7,  7,  8,  8,  9,  9, 10, 10, 10, 10, 10,
    11,  8,  6,  5,  5,  6,  7,  7,  8,  8,  8,  9,  9, 10, 10, 10,
    10,  8,  7,  6,  6,  6,  7,  7,  8,  8,  8,  9,  9, 10, 10, 10,
    10,  8,  8,  7,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10, 10,
    10,  8,  9,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9, 10, 10, 10,
    10,  8,  9,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9, 10, 10, 10,
    10,  8, 10,  8,  8,  8,  8,  8,  8,  9,  9,  9, 10, 10, 10, 10,
    11,  8, 10,  9,  8,  8,  8,  8,  9,  9,  9,  9, 10, 10, 10, 10,
    11,  8, 10,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 11,
    11,  8, 11,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 11, 11,
    11,  8, 11, 10,  9,  9,  9, 10, 10, 10, 10, 10, 10, 11, 11, 11,
    11,  8, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
    12,  8, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
    12,  8, 12, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12,
    12,  8, 12, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12,
    12,  8, 10,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
     8,  5,
]


# ---- Codebook metadata ----

ZERO_HCB = 0
FIRST_PAIR_HCB = 5
ESC_HCB = 11
NOISE_HCB = 13
INTENSITY_HCB2 = 14
INTENSITY_HCB = 15

@dataclass
class HuffmanCodebook:
    index: int
    dimension: int      # 4 (quad) or 2 (pair)
    signed: bool        # True = values are signed; False = sign bits sent separately
    max_abs: int        # max absolute value per element
    has_escape: bool    # CB 11 only
    lengths: list[int]
    codes: dict[tuple, tuple[int, int]]  # value_tuple -> (codeword, bitlength)

    @property
    def num_entries(self) -> int:
        return len(self.lengths)


def _build_canonical_codes(lengths: list[int]) -> list[tuple[int, int]]:
    """Build canonical Huffman codes from code lengths.

    Returns list of (codeword, length) pairs in the same order as lengths.
    Standard algorithm: sort by length, assign sequential codes.
    """
    n = len(lengths)
    indexed = sorted(range(n), key=lambda i: (lengths[i], i))

    codes = [0] * n
    code_lengths = list(lengths)
    code = 0
    prev_len = 0

    for idx in indexed:
        cl = lengths[idx]
        if cl == 0:
            continue
        code <<= (cl - prev_len)
        codes[idx] = code
        code += 1
        prev_len = cl

    return [(codes[i], lengths[i]) for i in range(n)]


def _build_codebook(
    index: int,
    dimension: int,
    signed: bool,
    max_abs: int,
    has_escape: bool,
    lengths: list[int],
) -> HuffmanCodebook:
    """Build a codebook with value-tuple-to-code mapping."""
    canonical = _build_canonical_codes(lengths)

    if signed:
        dim_range = range(-max_abs, max_abs + 1)
    else:
        dim_range = range(0, max_abs + 1)

    dim_size = len(dim_range)
    values = list(dim_range)

    codes_dict = {}
    if dimension == 4:
        for a in range(dim_size):
            for b in range(dim_size):
                for c in range(dim_size):
                    for d in range(dim_size):
                        idx = a * dim_size**3 + b * dim_size**2 + c * dim_size + d
                        if idx < len(canonical):
                            cw, cl = canonical[idx]
                            codes_dict[(values[a], values[b], values[c], values[d])] = (cw, cl)
    else:  # dimension == 2
        for a in range(dim_size):
            for b in range(dim_size):
                idx = a * dim_size + b
                if idx < len(canonical):
                    cw, cl = canonical[idx]
                    codes_dict[(values[a], values[b])] = (cw, cl)

    return HuffmanCodebook(
        index=index,
        dimension=dimension,
        signed=signed,
        max_abs=max_abs,
        has_escape=has_escape,
        lengths=lengths,
        codes=codes_dict,
    )


# Build all codebooks at import time

CODEBOOKS: dict[int, HuffmanCodebook] = {
    1: _build_codebook(1, 4, True, 1, False, CB1_LENGTHS),
    2: _build_codebook(2, 4, True, 1, False, CB2_LENGTHS),
    3: _build_codebook(3, 4, False, 2, False, CB3_LENGTHS),
    4: _build_codebook(4, 4, False, 2, False, CB4_LENGTHS),
    5: _build_codebook(5, 2, True, 4, False, CB5_LENGTHS),
    6: _build_codebook(6, 2, True, 4, False, CB6_LENGTHS),
    7: _build_codebook(7, 2, False, 7, False, CB7_LENGTHS),
    8: _build_codebook(8, 2, False, 7, False, CB8_LENGTHS),
    9: _build_codebook(9, 2, False, 12, False, CB9_LENGTHS),
    10: _build_codebook(10, 2, False, 12, False, CB10_LENGTHS),
    11: _build_codebook(11, 2, False, 16, True, CB11_LENGTHS),
}

# Scalefactor codebook
SF_CANONICAL = _build_canonical_codes(SF_CODE_LENGTHS)
SF_CODES: dict[int, tuple[int, int]] = {
    i - 60: SF_CANONICAL[i] for i in range(121)
}


def select_codebook(max_abs_value: int) -> int:
    """Select the best codebook for a section based on max absolute value.

    Returns codebook index (1-11, or 0 for zero section).
    """
    if max_abs_value == 0:
        return ZERO_HCB
    if max_abs_value <= 1:
        return 1
    if max_abs_value <= 2:
        return 3
    if max_abs_value <= 4:
        return 5
    if max_abs_value <= 7:
        return 7
    if max_abs_value <= 12:
        return 9
    return ESC_HCB  # 11, with escape coding for values > 16
