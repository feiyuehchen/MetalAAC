"""ISO/IEC 14496-3 raw_data_block reader/writer for AAC-LC.

Encodes and decodes single_channel_element (SCE) containing:
  - element_instance_tag (4 bits)
  - individual_channel_stream:
      - global_gain (8 bits)
      - ics_info: window_sequence, window_shape, max_sfb, predictor
      - section_data: codebook index + section length per section
      - scale_factor_data: DPCM scalefactors with SF Huffman codebook
      - spectral_data: quantized MDCT coefficients with spectral codebooks
"""

from __future__ import annotations

import numpy as np

from metal_aac.tables.huffman_tables import (
    CODEBOOKS,
    ESC_HCB,
    SF_CODES,
    SF_CODE_LENGTHS,
    SF_CODE_VALUES,
    ZERO_HCB,
    select_codebook,
)
from metal_aac.tables.scalefactor_bands import get_sfb_offsets


class BitWriter:
    """Accumulate bits into a Python bigint, flush to bytes at the end.

    Using a single bigint avoids per-byte boundary checks in the hot loop.
    Python's arbitrary-precision int shift/OR is implemented in C and is
    significantly faster than per-write byte flushing.
    """

    __slots__ = ('_acc', '_nbits')

    def __init__(self):
        self._acc = 0
        self._nbits = 0

    def write(self, value: int, n_bits: int) -> None:
        self._acc = (self._acc << n_bits) | (value & ((1 << n_bits) - 1))
        self._nbits += n_bits

    def flush(self) -> bytes:
        n = self._nbits
        if n == 0:
            return b''
        pad = (8 - n % 8) % 8
        acc = self._acc << pad
        total_bytes = (n + pad) // 8
        return acc.to_bytes(total_bytes, 'big')

    @property
    def bits_written(self) -> int:
        return self._nbits


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
    """Write escape-coded value for CB 11 (values >= 16).

    ISO 14496-3 Section 4.6.3.3: write N ones + 0 + (N+4) bits of value,
    where N = floor(log2(value)) - 3, i.e. 2^(N+4) > value.
    """
    n = value
    count = 0
    while n >= (1 << (count + 5)):
        count += 1
    for _ in range(count):
        bw.write(1, 1)
    bw.write(0, 1)
    bw.write(n, count + 4)


def encode_raw_data_block_iso(
    quantized: np.ndarray,
    iso_scalefactors: np.ndarray,
    iso_global_gain: int,
    window_sequence: int = 0,
    sample_rate: int = 44100,
) -> bytes:
    """Encode one frame with ISO scalefactors directly (no formula conversion).

    iso_scalefactors: (num_sfb,) int32 — direct ISO SF values (100-255)
    iso_global_gain: int — written directly as global_gain in ADTS header
    """
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb = len(sfb_offsets) - 1
    quantized = np.clip(quantized, -255, 255)
    sections = _compute_sections(quantized, sfb_offsets)

    bw = BitWriter()
    bw.write(0, 3)
    bw.write(0, 4)
    bw.write(iso_global_gain & 0xFF, 8)

    bw.write(0, 1)
    bw.write(window_sequence & 0x3, 2)
    bw.write(1, 1)
    if window_sequence == 2:
        bw.write(num_sfb & 0xF, 4)
        bw.write(0x7F, 7)
    else:
        bw.write(num_sfb & 0x3F, 6)
        bw.write(0, 1)

    if window_sequence == 2:
        sect_esc_val, sect_bits = 7, 3
    else:
        sect_esc_val, sect_bits = 31, 5

    for start_sfb, end_sfb, cb in sections:
        sect_len = end_sfb - start_sfb
        bw.write(cb & 0xF, 4)
        while sect_len >= sect_esc_val:
            bw.write(sect_esc_val, sect_bits)
            sect_len -= sect_esc_val
        bw.write(sect_len, sect_bits)

    prev_sf = iso_global_gain
    for sb in range(num_sfb):
        cb = ZERO_HCB
        for s_start, s_end, s_cb in sections:
            if s_start <= sb < s_end:
                cb = s_cb
                break
        if cb == ZERO_HCB:
            continue
        diff = int(iso_scalefactors[sb]) - prev_sf
        diff = max(-60, min(60, diff))
        if diff in SF_CODES:
            cw, cl = SF_CODES[diff]
            bw.write(cw, cl)
        else:
            bw.write(0, 1)
        prev_sf += diff

    bw.write(0, 1)
    bw.write(0, 1)
    bw.write(0, 1)

    q_list = quantized.tolist() if hasattr(quantized, 'tolist') else list(quantized)
    n_coeffs = len(q_list)
    for start_sfb, end_sfb, cb in sections:
        if cb == ZERO_HCB:
            continue
        codebook = CODEBOOKS.get(cb)
        if codebook is None:
            continue
        lo = sfb_offsets[start_sfb]
        hi = min(sfb_offsets[end_sfb], n_coeffs)
        if codebook.dimension == 4:
            for i in range(lo, hi, 4):
                v0 = q_list[i] if i < n_coeffs else 0
                v1 = q_list[i+1] if i+1 < n_coeffs else 0
                v2 = q_list[i+2] if i+2 < n_coeffs else 0
                v3 = q_list[i+3] if i+3 < n_coeffs else 0
                _encode_spectral_quad(bw, cb, v0, v1, v2, v3)
        else:
            for i in range(lo, hi, 2):
                v0 = q_list[i] if i < n_coeffs else 0
                v1 = q_list[i+1] if i+1 < n_coeffs else 0
                _encode_spectral_pair(bw, cb, v0, v1)

    bw.write(7, 3)
    return bw.flush()


def _write_ics(
    bw: BitWriter,
    quantized: np.ndarray,
    iso_scalefactors: np.ndarray,
    iso_global_gain: int,
    sections: list,
    sfb_offsets: list[int],
    num_sfb: int,
    window_sequence: int = 0,
) -> None:
    """Write an individual_channel_stream (no ics_info — caller writes it)."""
    bw.write(iso_global_gain & 0xFF, 8)

    if window_sequence == 2:
        sect_esc_val, sect_bits = 7, 3
    else:
        sect_esc_val, sect_bits = 31, 5

    for start_sfb, end_sfb, cb in sections:
        sect_len = end_sfb - start_sfb
        bw.write(cb & 0xF, 4)
        while sect_len >= sect_esc_val:
            bw.write(sect_esc_val, sect_bits)
            sect_len -= sect_esc_val
        bw.write(sect_len, sect_bits)

    prev_sf = iso_global_gain
    for sb in range(num_sfb):
        cb = ZERO_HCB
        for s_start, s_end, s_cb in sections:
            if s_start <= sb < s_end:
                cb = s_cb
                break
        if cb == ZERO_HCB:
            continue
        diff = int(iso_scalefactors[sb]) - prev_sf
        diff = max(-60, min(60, diff))
        if diff in SF_CODES:
            cw, cl = SF_CODES[diff]
            bw.write(cw, cl)
        else:
            bw.write(0, 1)
        prev_sf += diff

    bw.write(0, 1)  # pulse
    bw.write(0, 1)  # tns
    bw.write(0, 1)  # gain control

    quantized = np.clip(quantized, -255, 255)
    q_list = quantized.tolist() if hasattr(quantized, 'tolist') else list(quantized)
    n_coeffs = len(q_list)
    for start_sfb, end_sfb, cb in sections:
        if cb == ZERO_HCB:
            continue
        codebook = CODEBOOKS.get(cb)
        if codebook is None:
            continue
        lo = sfb_offsets[start_sfb]
        hi = min(sfb_offsets[end_sfb], n_coeffs)
        if codebook.dimension == 4:
            for i in range(lo, hi, 4):
                v0 = q_list[i] if i < n_coeffs else 0
                v1 = q_list[i+1] if i+1 < n_coeffs else 0
                v2 = q_list[i+2] if i+2 < n_coeffs else 0
                v3 = q_list[i+3] if i+3 < n_coeffs else 0
                _encode_spectral_quad(bw, cb, v0, v1, v2, v3)
        else:
            for i in range(lo, hi, 2):
                v0 = q_list[i] if i < n_coeffs else 0
                v1 = q_list[i+1] if i+1 < n_coeffs else 0
                _encode_spectral_pair(bw, cb, v0, v1)


def encode_cpe_iso(
    q_l: np.ndarray, sf_l: np.ndarray, gg_l: int,
    q_r: np.ndarray, sf_r: np.ndarray, gg_r: int,
    window_sequence: int = 0,
    sample_rate: int = 44100,
    ms_used: np.ndarray | None = None,
) -> bytes:
    """Encode a stereo frame as a Channel Pair Element (CPE).

    ms_used: (num_sfb,) bool — which SFBs use M/S coding. None = no M/S.
    """
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb = len(sfb_offsets) - 1

    sections_l = _compute_sections(np.clip(q_l, -255, 255), sfb_offsets)
    sections_r = _compute_sections(np.clip(q_r, -255, 255), sfb_offsets)

    bw = BitWriter()
    bw.write(1, 3)   # ID_CPE
    bw.write(0, 4)   # element_instance_tag
    bw.write(1, 1)   # common_window

    bw.write(0, 1)   # reserved
    bw.write(window_sequence & 0x3, 2)
    bw.write(1, 1)   # KBD
    if window_sequence == 2:
        bw.write(num_sfb & 0xF, 4)
        bw.write(0x7F, 7)
    else:
        bw.write(num_sfb & 0x3F, 6)
        bw.write(0, 1)

    if ms_used is None or not np.any(ms_used):
        bw.write(0, 2)
    elif np.all(ms_used[:num_sfb]):
        bw.write(2, 2)
    else:
        bw.write(1, 2)
        for sb in range(num_sfb):
            bw.write(1 if ms_used[sb] else 0, 1)

    _write_ics(bw, q_l, sf_l, gg_l, sections_l, sfb_offsets, num_sfb, window_sequence)
    _write_ics(bw, q_r, sf_r, gg_r, sections_r, sfb_offsets, num_sfb, window_sequence)

    bw.write(7, 3)
    return bw.flush()


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

    quantized = np.clip(quantized, -255, 255)

    # Compute ISO scalefactors for the bitstream header.
    # ffmpeg decoder: scale = 2^((sf - 200) / 4)
    # Our quantizer: q = |x|^(3/4) * 2^((gg - sf_int*4)/16)
    # MDCT is now normalized (2/N), so iso_sf = 200 - (gg - sf_int*4) / 3
    sections = _compute_sections(quantized, sfb_offsets)
    iso_sfs = [max(0, min(255, int(round(200 - (global_gain - int(scalefactors[sb])*4) / 3.0)))) for sb in range(num_sfb)]
    non_zero_sfs = [iso_sfs[sb] for sb in range(num_sfb)
                    if any(s <= sb < e and cb != ZERO_HCB for s, e, cb in sections)]
    iso_global_gain = int(np.mean(non_zero_sfs)) if non_zero_sfs else 100

    bw = BitWriter()

    bw.write(0, 3)  # ID_SCE = 0
    bw.write(0, 4)  # instance_tag = 0
    bw.write(iso_global_gain & 0xFF, 8)  # global_gain in ISO convention

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
    # (sections already computed above for ISO SF derivation)
    # ISO: short windows use 3-bit section lengths (esc=7), long use 5-bit (esc=31)
    if window_sequence == 2:
        sect_esc_val = 7
        sect_bits = 3
    else:
        sect_esc_val = 31
        sect_bits = 5

    for start_sfb, end_sfb, cb in sections:
        sect_len = end_sfb - start_sfb
        bw.write(cb & 0xF, 4)  # sect_cb
        while sect_len >= sect_esc_val:
            bw.write(sect_esc_val, sect_bits)
            sect_len -= sect_esc_val
        bw.write(sect_len, sect_bits)

    # ---- scale_factor_data ----
    # Convert internal scalefactors to ISO DPCM.
    # The ISO decoder reconstructs: x_hat = |q|^(4/3) * 2^(0.25*(sf-100))
    # Our quantizer uses: q = |x|^(3/4) * 2^((gg - internal_sf*4)/16)
    # For round-trip correctness: iso_sf = 100 - (gg - internal_sf*4) / 3
    prev_sf = iso_global_gain
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
        prev_sf += diff  # track decoder's actual state, not intended value

    # ---- pulse_data ----
    bw.write(0, 1)  # pulse_data_present = 0

    # ---- tns_data ----
    bw.write(0, 1)  # tns_data_present = 0

    # ---- gain_control_data (SSR only, but bit is always read) ----
    bw.write(0, 1)  # gain_control_data_present = 0

    # ---- spectral_data ----
    # Pre-convert to Python ints once (avoids repeated numpy-to-Python conversion)
    q_list = quantized.tolist() if hasattr(quantized, 'tolist') else list(quantized)
    n_coeffs = len(q_list)

    for start_sfb, end_sfb, cb in sections:
        if cb == ZERO_HCB:
            continue

        codebook = CODEBOOKS.get(cb)
        if codebook is None:
            continue

        lo = sfb_offsets[start_sfb]
        hi = min(sfb_offsets[end_sfb], n_coeffs)

        if codebook.dimension == 4:
            for i in range(lo, hi, 4):
                v0 = q_list[i] if i < n_coeffs else 0
                v1 = q_list[i+1] if i+1 < n_coeffs else 0
                v2 = q_list[i+2] if i+2 < n_coeffs else 0
                v3 = q_list[i+3] if i+3 < n_coeffs else 0
                _encode_spectral_quad(bw, cb, v0, v1, v2, v3)
        else:
            for i in range(lo, hi, 2):
                v0 = q_list[i] if i < n_coeffs else 0
                v1 = q_list[i+1] if i+1 < n_coeffs else 0
                _encode_spectral_pair(bw, cb, v0, v1)

    # ID_END element
    bw.write(7, 3)  # ID_END = 7

    return bw.flush()


# ---- Decoder ----


class BitReader:
    __slots__ = ('_data', '_pos', '_nbits')

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0
        self._nbits = len(data) * 8

    def read(self, n: int) -> int:
        result = 0
        for _ in range(n):
            if self._pos >= self._nbits:
                return result
            byte_idx = self._pos >> 3
            bit_idx = 7 - (self._pos & 7)
            result = (result << 1) | ((self._data[byte_idx] >> bit_idx) & 1)
            self._pos += 1
        return result

    def read1(self) -> int:
        if self._pos >= self._nbits:
            return 0
        byte_idx = self._pos >> 3
        bit_idx = 7 - (self._pos & 7)
        self._pos += 1
        return (self._data[byte_idx] >> bit_idx) & 1

    @property
    def remaining(self) -> int:
        return self._nbits - self._pos


def _build_huff_tree(codes: list[int], lengths: list[int]) -> list:
    """Build a binary tree for Huffman decoding. [left, right, value]."""
    root = [None, None, None]
    for i, (code, length) in enumerate(zip(codes, lengths)):
        if length == 0:
            continue
        node = root
        for bit in range(length - 1, -1, -1):
            b = (code >> bit) & 1
            if node[b] is None:
                node[b] = [None, None, None]
            node = node[b]
        node[2] = i
    return root


def _huff_decode(br: BitReader, tree: list) -> int | None:
    node = tree
    while node[2] is None:
        if br.remaining <= 0:
            return None
        b = br.read1()
        node = node[b]
        if node is None:
            return None
    return node[2]


_DECODE_TREES: dict[int, list] = {}
_SF_DECODE_TREE: list | None = None


def _get_spectral_tree(cb_idx: int) -> list:
    if cb_idx not in _DECODE_TREES:
        from metal_aac.tables import huffman_tables as ht
        codes = getattr(ht, f'CB{cb_idx}_CODES')
        lengths = getattr(ht, f'CB{cb_idx}_LENGTHS')
        _DECODE_TREES[cb_idx] = _build_huff_tree(codes, lengths)
    return _DECODE_TREES[cb_idx]


def _get_sf_tree() -> list:
    global _SF_DECODE_TREE
    if _SF_DECODE_TREE is None:
        _SF_DECODE_TREE = _build_huff_tree(SF_CODE_VALUES, SF_CODE_LENGTHS)
    return _SF_DECODE_TREE


def _index_to_values(idx: int, dim: int, signed: bool, max_abs: int) -> tuple:
    dim_size = (2 * max_abs + 1) if signed else (max_abs + 1)
    offset = max_abs if signed else 0
    vals = []
    for _ in range(dim):
        vals.append(idx % dim_size - offset)
        idx //= dim_size
    vals.reverse()
    return tuple(vals)


def _read_escape(br: BitReader) -> int:
    count = 0
    while br.read1() == 1:
        count += 1
    return (1 << (count + 4)) | br.read(count + 4)


def _read_ics_body(
    br: BitReader, sfb_offsets: list[int], num_sfb: int,
    sect_esc_val: int, sect_bits: int, global_gain: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read section_data + scale_factor_data + spectral_data of an ICS."""
    num_sfb_max = len(sfb_offsets) - 1

    sections = []
    k = 0
    while k < num_sfb:
        cb = br.read(4)
        sect_len = 0
        while True:
            inc = br.read(sect_bits)
            sect_len += inc
            if inc < sect_esc_val:
                break
        end = min(k + sect_len, num_sfb)
        sections.append((k, end, cb))
        k = end

    scalefactors = np.zeros(num_sfb_max, dtype=np.int32)
    sf_tree = _get_sf_tree()
    prev_sf = global_gain
    for sb in range(num_sfb):
        cb = ZERO_HCB
        for s_start, s_end, s_cb in sections:
            if s_start <= sb < s_end:
                cb = s_cb
                break
        if cb == ZERO_HCB:
            scalefactors[sb] = prev_sf
            continue
        idx = _huff_decode(br, sf_tree)
        if idx is None:
            scalefactors[sb] = prev_sf
            continue
        diff = idx - 60
        prev_sf += diff
        scalefactors[sb] = prev_sf

    br.read(1); br.read(1); br.read(1)

    n_coeffs = sfb_offsets[-1] if sfb_offsets else 1024
    quantized = np.zeros(n_coeffs, dtype=np.int32)

    for start_sfb, end_sfb, cb in sections:
        if cb == ZERO_HCB or cb not in CODEBOOKS:
            continue
        codebook = CODEBOOKS[cb]
        tree = _get_spectral_tree(cb)
        lo = sfb_offsets[start_sfb]
        hi = min(sfb_offsets[end_sfb], n_coeffs)

        if codebook.dimension == 4:
            for i in range(lo, hi, 4):
                idx = _huff_decode(br, tree)
                if idx is None:
                    break
                vals = list(_index_to_values(idx, 4, codebook.signed, codebook.max_abs))
                if not codebook.signed:
                    for k in range(4):
                        if vals[k] > 0 and br.read1():
                            vals[k] = -vals[k]
                for k in range(4):
                    if i + k < n_coeffs:
                        quantized[i + k] = vals[k]
        else:
            for i in range(lo, hi, 2):
                idx = _huff_decode(br, tree)
                if idx is None:
                    break
                vals = list(_index_to_values(idx, 2, codebook.signed, codebook.max_abs))
                if not codebook.signed:
                    for k in range(2):
                        if vals[k] > 0 and br.read1():
                            vals[k] = -vals[k]
                    if cb == ESC_HCB:
                        for k in range(2):
                            if abs(vals[k]) >= 16:
                                sign = -1 if vals[k] < 0 else 1
                                vals[k] = sign * _read_escape(br)
                for k in range(2):
                    if i + k < n_coeffs:
                        quantized[i + k] = vals[k]

    return quantized, scalefactors


def _read_ics_info(br: BitReader, num_sfb_max: int) -> tuple[int, int, int]:
    """Read ics_info. Returns (num_sfb, sect_esc_val, sect_bits)."""
    _reserved = br.read(1)
    window_sequence = br.read(2)
    _shape = br.read(1)
    if window_sequence == 2:
        num_sfb = min(br.read(4), num_sfb_max)
        _grouping = br.read(7)
        return num_sfb, 7, 3
    else:
        num_sfb = min(br.read(6), num_sfb_max)
        _predictor = br.read(1)
        return num_sfb, 31, 5


def decode_raw_data_block_iso(
    data: bytes,
    sample_rate: int = 44100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Decode one ISO raw_data_block (SCE or CPE ch0) from bytes."""
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb_max = len(sfb_offsets) - 1
    br = BitReader(data)

    element_id = br.read(3)
    _tag = br.read(4)

    if element_id == 1:
        ch0, _ch1, _ms = decode_cpe_iso(data, sample_rate)
        return ch0

    # SCE: global_gain + ics_info + ICS body
    global_gain = br.read(8)
    num_sfb, sect_esc_val, sect_bits = _read_ics_info(br, num_sfb_max)
    quantized, scalefactors = _read_ics_body(
        br, sfb_offsets, num_sfb, sect_esc_val, sect_bits, global_gain
    )
    return quantized, scalefactors, global_gain


def decode_cpe_iso(
    data: bytes,
    sample_rate: int = 44100,
) -> tuple[tuple[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray, int], np.ndarray]:
    """Decode a Channel Pair Element.

    Returns ((q_l, sf_l, gg_l), (q_r, sf_r, gg_r), ms_used).
    ms_used: (num_sfb,) bool array.
    """
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb_max = len(sfb_offsets) - 1
    br = BitReader(data)

    _element_id = br.read(3)
    _tag = br.read(4)
    common_window = br.read(1)

    ms_used = np.zeros(num_sfb_max, dtype=bool)
    if common_window:
        num_sfb, sect_esc_val, sect_bits = _read_ics_info(br, num_sfb_max)
        ms_mask_present = br.read(2)
        if ms_mask_present == 1:
            for sb in range(num_sfb):
                ms_used[sb] = bool(br.read(1))
        elif ms_mask_present == 2:
            ms_used[:num_sfb] = True

    gg0 = br.read(8)
    if not common_window:
        num_sfb, sect_esc_val, sect_bits = _read_ics_info(br, num_sfb_max)
    q0, sf0 = _read_ics_body(br, sfb_offsets, num_sfb, sect_esc_val, sect_bits, gg0)
    gg1 = br.read(8)
    if not common_window:
        num_sfb, sect_esc_val, sect_bits = _read_ics_info(br, num_sfb_max)
    q1, sf1 = _read_ics_body(br, sfb_offsets, num_sfb, sect_esc_val, sect_bits, gg1)
    return (q0, sf0, gg0), (q1, sf1, gg1), ms_used
