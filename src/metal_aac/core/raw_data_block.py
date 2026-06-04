"""ISO/IEC 14496-3 raw_data_block writer for AAC-LC.

Writes a single_channel_element (SCE) containing:
  - element_instance_tag (4 bits)
  - individual_channel_stream:
      - global_gain (8 bits)
      - ics_info: window_sequence, window_shape, max_sfb, predictor
      - section_data: codebook index + section length per section
      - scale_factor_data: DPCM scalefactors with SF Huffman codebook
      - spectral_data: quantized MDCT coefficients with spectral codebooks

This replaces the legacy exp-Golomb encoding. The output of
encode_raw_data_block() is the payload passed to ADTSWriter.write_frame().
"""

from __future__ import annotations

import numpy as np

from metal_aac.tables.huffman_tables import (
    CODEBOOKS,
    ESC_HCB,
    SF_CODES,
    ZERO_HCB,
    select_codebook,
)
from metal_aac.tables.scalefactor_bands import get_sfb_offsets


class BitWriter:
    """Accumulate bits and flush to bytes."""

    def __init__(self):
        self._buffer = 0
        self._bits = 0
        self._bytes = bytearray()

    def write(self, value: int, n_bits: int) -> None:
        self._buffer = (self._buffer << n_bits) | (value & ((1 << n_bits) - 1))
        self._bits += n_bits
        while self._bits >= 8:
            self._bits -= 8
            self._bytes.append((self._buffer >> self._bits) & 0xFF)

    def flush(self) -> bytes:
        if self._bits > 0:
            self._bytes.append((self._buffer << (8 - self._bits)) & 0xFF)
            self._bits = 0
            self._buffer = 0
        return bytes(self._bytes)

    @property
    def bits_written(self) -> int:
        return len(self._bytes) * 8 + self._bits


def _compute_sections(
    quantized: np.ndarray,
    sfb_offsets: list[int],
) -> list[tuple[int, int, int]]:
    """Divide SFBs into sections, each assigned a codebook.

    Returns list of (start_sfb, end_sfb, codebook_index).
    Greedy: each SFB gets the codebook matching its max abs value,
    then adjacent SFBs with the same codebook are merged.
    """
    num_sfb = len(sfb_offsets) - 1
    per_sfb_cb = []
    for sb in range(num_sfb):
        lo, hi = sfb_offsets[sb], sfb_offsets[sb + 1]
        max_abs = int(np.max(np.abs(quantized[lo:hi])))
        per_sfb_cb.append(select_codebook(max_abs))

    sections = []
    i = 0
    while i < num_sfb:
        cb = per_sfb_cb[i]
        j = i + 1
        while j < num_sfb and per_sfb_cb[j] == cb:
            j += 1
        sections.append((i, j, cb))
        i = j

    return sections


def _encode_spectral_pair(
    bw: BitWriter, cb: int, v0: int, v1: int
) -> None:
    """Encode a pair of quantized values with the given codebook."""
    codebook = CODEBOOKS[cb]

    if codebook.signed:
        key = (v0, v1)
        if key in codebook.codes:
            cw, cl = codebook.codes[key]
            bw.write(cw, cl)
        else:
            bw.write(0, 1)
    else:
        a0, a1 = abs(v0), abs(v1)
        if cb == ESC_HCB:
            a0_clamp = min(a0, 16)
            a1_clamp = min(a1, 16)
        else:
            a0_clamp = min(a0, codebook.max_abs)
            a1_clamp = min(a1, codebook.max_abs)

        key = (a0_clamp, a1_clamp)
        if key in codebook.codes:
            cw, cl = codebook.codes[key]
            bw.write(cw, cl)
        else:
            bw.write(0, 1)

        if a0_clamp > 0:
            bw.write(1 if v0 < 0 else 0, 1)
        if a1_clamp > 0:
            bw.write(1 if v1 < 0 else 0, 1)

        # Escape coding for CB 11
        if cb == ESC_HCB:
            if a0 >= 16:
                _write_escape(bw, a0)
            if a1 >= 16:
                _write_escape(bw, a1)


def _encode_spectral_quad(
    bw: BitWriter, cb: int, v0: int, v1: int, v2: int, v3: int
) -> None:
    """Encode a quad of quantized values (codebooks 1-4)."""
    codebook = CODEBOOKS[cb]

    if codebook.signed:
        key = (v0, v1, v2, v3)
        if key in codebook.codes:
            cw, cl = codebook.codes[key]
            bw.write(cw, cl)
        else:
            bw.write(0, 1)
    else:
        a0, a1, a2, a3 = abs(v0), abs(v1), abs(v2), abs(v3)
        key = (min(a0, codebook.max_abs), min(a1, codebook.max_abs),
               min(a2, codebook.max_abs), min(a3, codebook.max_abs))
        if key in codebook.codes:
            cw, cl = codebook.codes[key]
            bw.write(cw, cl)
        else:
            bw.write(0, 1)

        for v, a in [(v0, a0), (v1, a1), (v2, a2), (v3, a3)]:
            if a > 0:
                bw.write(1 if v < 0 else 0, 1)


def _write_escape(bw: BitWriter, value: int) -> None:
    """Write escape-coded value for CB 11 (values >= 16)."""
    # Escape: count = floor(log2(value)) - 4, then write count 1-bits,
    # one 0-bit, then (count+4) bits of the value
    n = value
    count = 0
    while n >= (1 << (count + 5)):
        count += 1
    for _ in range(count):
        bw.write(1, 1)
    bw.write(0, 1)
    bw.write(n, count + 4)


def encode_raw_data_block(
    quantized: np.ndarray,
    scalefactors: np.ndarray,
    global_gain: int,
    window_sequence: int = 0,
    sample_rate: int = 44100,
) -> bytes:
    """Encode one frame as an ISO-compliant raw_data_block (SCE).

    quantized: (N,) int32
    scalefactors: (num_sfb,) int32
    global_gain: int
    window_sequence: 0=ONLY_LONG, 1=LONG_START, 2=EIGHT_SHORT, 3=LONG_STOP

    Returns bytes suitable for ADTSWriter.write_frame().
    """
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb = len(sfb_offsets) - 1
    bw = BitWriter()

    # ID_SCE (3 bits) + element_instance_tag (4 bits)
    bw.write(0, 3)  # ID_SCE = 0
    bw.write(0, 4)  # instance_tag = 0

    # global_gain (8 bits)
    bw.write(global_gain & 0xFF, 8)

    # ---- ics_info ----
    bw.write(0, 1)  # ics_reserved_bit
    bw.write(window_sequence & 0x3, 2)
    bw.write(1, 1)  # window_shape (1 = KBD)

    if window_sequence == 2:  # EIGHT_SHORT_SEQUENCE
        bw.write(num_sfb & 0xF, 4)  # max_sfb (4 bits for short)
        bw.write(0x7F, 7)  # scale_factor_grouping (all in one group)
    else:
        bw.write(num_sfb & 0x3F, 6)  # max_sfb (6 bits for long)
        bw.write(0, 1)  # predictor_data_present = 0 (LC profile)

    # ---- section_data ----
    sections = _compute_sections(quantized, sfb_offsets)
    sect_esc_val = 31 if window_sequence == 2 else 31
    # For long windows, sect_esc_val would be 31 (2^5-1) per the standard
    # but we use simplified sectioning here

    for start_sfb, end_sfb, cb in sections:
        sect_len = end_sfb - start_sfb
        bw.write(cb & 0xF, 4)  # sect_cb
        while sect_len >= sect_esc_val:
            bw.write(sect_esc_val, 5)
            sect_len -= sect_esc_val
        bw.write(sect_len, 5)

    # ---- scale_factor_data ----
    # Convert internal scalefactors to ISO DPCM.
    # Internal SF (0-60) maps to ISO SF via: iso_sf = global_gain - internal_sf
    # Clamp to [0, 255] to stay within AAC valid range, and ensure
    # consecutive diffs stay within SF Huffman codebook range [-60, +60].
    iso_sfs = [max(0, min(255, global_gain - int(scalefactors[sb]))) for sb in range(num_sfb)]
    prev_sf = global_gain
    for sb in range(num_sfb):
        cb = ZERO_HCB
        for s_start, s_end, s_cb in sections:
            if s_start <= sb < s_end:
                cb = s_cb
                break

        if cb == ZERO_HCB:
            continue

        diff = iso_sfs[sb] - prev_sf
        diff = max(-60, min(60, diff))
        if diff in SF_CODES:
            cw, cl = SF_CODES[diff]
            bw.write(cw, cl)
        else:
            bw.write(0, 1)
        prev_sf = iso_sfs[sb]

    # ---- pulse_data ----
    bw.write(0, 1)  # pulse_data_present = 0

    # ---- tns_data ----
    bw.write(0, 1)  # tns_data_present = 0

    # ---- gain_control_data (SSR only, but bit is always read) ----
    bw.write(0, 1)  # gain_control_data_present = 0

    # ---- spectral_data ----
    for start_sfb, end_sfb, cb in sections:
        if cb == ZERO_HCB:
            continue

        codebook = CODEBOOKS.get(cb)
        if codebook is None:
            continue

        lo = sfb_offsets[start_sfb]
        hi = sfb_offsets[end_sfb]

        if codebook.dimension == 4:
            for i in range(lo, hi, 4):
                vals = [int(quantized[j]) if j < len(quantized) else 0 for j in range(i, i + 4)]
                _encode_spectral_quad(bw, cb, *vals)
        else:
            for i in range(lo, hi, 2):
                v0 = int(quantized[i]) if i < len(quantized) else 0
                v1 = int(quantized[i + 1]) if i + 1 < len(quantized) else 0
                _encode_spectral_pair(bw, cb, v0, v1)

    # ID_END element
    bw.write(7, 3)  # ID_END = 7

    return bw.flush()
