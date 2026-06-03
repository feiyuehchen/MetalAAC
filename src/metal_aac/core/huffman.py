"""Entropy coding for quantized MDCT coefficients.

The coding pipeline is split into two parallelizable stages:

Stage 1 (GPU-parallelizable): compute exp-Golomb codeword + length for
  every coefficient across all frames simultaneously. This is pure
  element-wise arithmetic -- no data dependencies between coefficients.

Stage 2 (chunk-parallel CPU): bit packing. Given precomputed codewords
  and lengths, pack into bytes. Each frame is independent, so frames
  are distributed across CPU cores via ProcessPoolExecutor.

This decomposition follows the same principle as nvCOMP's chunk-based
Huffman and gpuhd's subsequence-parallel approach: separate the
"what to write" (parallel) from the "where to write" (prefix-sum +
pack).
"""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from metal_aac.tables.huffman_codebooks import ExpGolombCodebook
from metal_aac.tables.scalefactor_bands import get_sfb_offsets


def encode_spectral_data(
    quantized: np.ndarray,
    scalefactors: np.ndarray,
    global_gain: int,
    sample_rate: int = 44100,
) -> bytes:
    """Encode quantized MDCT coefficients for one frame.

    quantized: (N,) int32
    scalefactors: (num_sfb,) int32
    global_gain: int
    returns: packed bytes
    """
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb = len(sfb_offsets) - 1

    bits: list[tuple[int, int]] = []  # (value, num_bits) pairs

    # Header: global_gain (8 bits)
    bits.append((global_gain & 0xFF, 8))

    # Scalefactors: differential coding, 8 bits each
    prev_sf = 0
    for sb in range(num_sfb):
        diff = int(scalefactors[sb]) - prev_sf
        diff_clipped = max(-128, min(127, diff))
        bits.append((diff_clipped & 0xFF, 8))
        prev_sf = int(scalefactors[sb])

    # Spectral data: exp-Golomb coded
    for sb in range(num_sfb):
        lo, hi = sfb_offsets[sb], sfb_offsets[sb + 1]
        band_data = quantized[lo:hi]
        codewords, bitlengths = ExpGolombCodebook.encode_section(band_data)
        for cw, bl in zip(codewords, bitlengths):
            bits.append((int(cw), int(bl)))

    return _pack_bits(bits)


def decode_spectral_data(
    data: bytes,
    sample_rate: int = 44100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Decode spectral data for one frame.

    returns: (quantized, scalefactors, global_gain)
    """
    reader = _BitReader(data)
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb = len(sfb_offsets) - 1
    n_coeffs = sfb_offsets[-1]

    # Header
    global_gain = reader.read(8)

    # Scalefactors
    scalefactors = np.zeros(num_sfb, dtype=np.int32)
    prev_sf = 0
    for sb in range(num_sfb):
        diff = reader.read(8)
        if diff > 127:
            diff -= 256
        scalefactors[sb] = prev_sf + diff
        prev_sf = scalefactors[sb]

    # Spectral data
    quantized = np.zeros(n_coeffs, dtype=np.int32)
    for sb in range(num_sfb):
        lo, hi = sfb_offsets[sb], sfb_offsets[sb + 1]
        for i in range(lo, hi):
            quantized[i] = _decode_exp_golomb(reader)

    return quantized, scalefactors, global_gain


def _encode_chunk(args: tuple) -> list[bytes]:
    """Encode a chunk of frames (for multiprocessing)."""
    quantized_chunk, sf_chunk, gain_chunk, sr = args
    results = []
    for i in range(len(quantized_chunk)):
        results.append(encode_spectral_data(
            quantized_chunk[i], sf_chunk[i], int(gain_chunk[i]), sr))
    return results


def encode_frames_parallel(
    quantized: np.ndarray,
    scalefactors: np.ndarray,
    global_gains: np.ndarray,
    sample_rate: int = 44100,
    n_workers: int | None = None,
) -> list[bytes]:
    """Encode all frames in parallel using multiprocessing.

    Each frame's Huffman coding is independent, so we split the work
    across CPU cores. This is the "split into segments, write in parallel"
    approach.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    n_frames = len(quantized)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, n_frames)
    n_workers = max(1, n_workers)

    if n_frames < n_workers * 2:
        return [
            encode_spectral_data(quantized[i], scalefactors[i],
                                 int(global_gains[i]), sample_rate)
            for i in range(n_frames)
        ]

    chunk_size = max(1, n_frames // n_workers)
    chunks = []
    for i in range(0, n_frames, chunk_size):
        end = min(i + chunk_size, n_frames)
        chunks.append((
            quantized[i:end], scalefactors[i:end],
            global_gains[i:end], sample_rate,
        ))

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        chunk_results = list(pool.map(_encode_chunk, chunks))

    return [item for sublist in chunk_results for item in sublist]


# ---- Metal GPU encode/decode ----


def encode_frames_metal(
    quantized: np.ndarray,
    scalefactors: np.ndarray,
    global_gains: np.ndarray,
    sample_rate: int = 44100,
) -> list[bytes]:
    """Encode all frames using Metal GPU compute shaders."""
    from metal_aac.core.metal_bridge import MetalHuffman

    metal = MetalHuffman.shared()
    return metal.encode_frames(quantized, scalefactors, global_gains)


def decode_frames_metal(
    frame_payloads: list[bytes],
    N: int = 1024,
    num_sfb: int = 49,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode all frames using Metal GPU compute shaders.

    Returns: (quantized (B,N), scalefactors (B,num_sfb), global_gains (B,))
    """
    from metal_aac.core.metal_bridge import MetalHuffman

    metal = MetalHuffman.shared()
    return metal.decode_frames(frame_payloads, N, num_sfb)


# ---- GPU-precomputed codewords + vectorized NumPy packing ----
# Follows the nvCOMP / gpuhd design: separate "what to write" (GPU-parallel)
# from "where to write" (prefix-sum) from "do the write" (chunk-parallel CPU).


def gpu_precompute_codewords(quantized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute exp-Golomb codewords and lengths for all frames on GPU.

    quantized: (B, N) int32
    returns: (codewords (B, N) int64, lengths (B, N) int32)

    This is Stage 1 -- purely element-wise, no data dependencies,
    runs on all B*N values simultaneously on GPU.
    """
    if not HAS_MLX:
        return _cpu_compute_codewords(quantized)

    q = mx.array(quantized.astype(np.int32))
    abs_q = mx.abs(q).astype(mx.int32)

    # Signed exp-Golomb mapping: +v→2v-1, -v→2v, 0→0
    code_num = mx.where(
        q > 0,
        2 * abs_q - 1,
        mx.where(q < 0, 2 * abs_q, mx.zeros_like(abs_q)),
    )
    # Codeword = code_num + 1 (its binary representation IS the code)
    codewords = (code_num + 1).astype(mx.int32)
    # Length = 2 * floor(log2(codeword)) + 1
    cw_float = mx.maximum(codewords.astype(mx.float32), 1.0)
    m = mx.floor(mx.log2(cw_float + 0.5))
    lengths = (2 * m + 1).astype(mx.int32)
    mx.eval(codewords, lengths)

    return np.array(codewords, dtype=np.int64), np.array(lengths, dtype=np.int32)


def _cpu_compute_codewords(quantized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy fallback for codeword computation."""
    q = quantized.astype(np.int64)
    abs_q = np.abs(q)
    code_num = np.where(q > 0, 2 * abs_q - 1, np.where(q < 0, 2 * abs_q, 0))
    codewords = code_num + 1
    m = np.floor(np.log2(np.maximum(codewords.astype(np.float64), 1.0) + 0.5))
    lengths = (2 * m + 1).astype(np.int32)
    return codewords, lengths


def _fast_pack_frame(
    codewords: np.ndarray,
    lengths: np.ndarray,
    scalefactors: np.ndarray,
    global_gain: int,
    sample_rate: int,
) -> bytes:
    """Pack one frame using precomputed codewords/lengths.

    Much faster than encode_spectral_data because no per-value Python
    math -- just bit shifting and ORing with precomputed values.
    """
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb = len(sfb_offsets) - 1
    bits: list[tuple[int, int]] = []

    # Header
    bits.append((global_gain & 0xFF, 8))
    prev_sf = 0
    for sb in range(num_sfb):
        diff = int(scalefactors[sb]) - prev_sf
        bits.append((max(-128, min(127, diff)) & 0xFF, 8))
        prev_sf = int(scalefactors[sb])

    # Spectral: use precomputed codewords directly
    for i in range(len(codewords)):
        bits.append((int(codewords[i]), int(lengths[i])))

    return _pack_bits(bits)


def _fast_pack_chunk(args: tuple) -> list[bytes]:
    """Pack a chunk of frames using precomputed codewords."""
    cw_chunk, len_chunk, sf_chunk, gain_chunk, sr = args
    return [
        _fast_pack_frame(cw_chunk[i], len_chunk[i], sf_chunk[i],
                         int(gain_chunk[i]), sr)
        for i in range(len(cw_chunk))
    ]


def encode_frames_gpu_parallel(
    quantized: np.ndarray,
    scalefactors: np.ndarray,
    global_gains: np.ndarray,
    sample_rate: int = 44100,
    n_workers: int | None = None,
) -> list[bytes]:
    """Two-stage encoding: GPU codeword computation + vectorized packing.

    Stage 1 (GPU): compute all codewords and lengths in one batch.
    Stage 2 (CPU): pack bits. Uses direct loop (no multiprocessing)
    because IPC serialization of precomputed arrays costs more than
    the packing itself.
    """
    # Stage 1: GPU batch codeword computation
    codewords, lengths = gpu_precompute_codewords(quantized)

    # Stage 2: pack each frame
    n_frames = len(quantized)
    return [
        _fast_pack_frame(codewords[i], lengths[i], scalefactors[i],
                         int(global_gains[i]), sample_rate)
        for i in range(n_frames)
    ]


def estimate_frame_bits(
    quantized: np.ndarray,
    scalefactors: np.ndarray,
    sample_rate: int = 44100,
) -> int:
    """Estimate total bits for a frame without actually encoding."""
    sfb_offsets = get_sfb_offsets(sample_rate)
    num_sfb = len(sfb_offsets) - 1

    # Header + scalefactors
    header_bits = 8 + num_sfb * 8

    # Spectral data
    spectral_bits = ExpGolombCodebook.estimate_bits(quantized)

    return header_bits + spectral_bits


# ---- Bit packing utilities ----


def _pack_bits(bits: list[tuple[int, int]]) -> bytes:
    """Pack a list of (value, num_bits) pairs into bytes."""
    buffer = 0
    buf_bits = 0
    result = bytearray()

    for value, num_bits in bits:
        buffer = (buffer << num_bits) | (value & ((1 << num_bits) - 1))
        buf_bits += num_bits

        while buf_bits >= 8:
            buf_bits -= 8
            result.append((buffer >> buf_bits) & 0xFF)

    if buf_bits > 0:
        result.append((buffer << (8 - buf_bits)) & 0xFF)

    return bytes(result)


class _BitReader:
    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0

    def read(self, n: int) -> int:
        result = 0
        for _ in range(n):
            if self._byte_pos >= len(self._data):
                return result
            bit = (self._data[self._byte_pos] >> (7 - self._bit_pos)) & 1
            result = (result << 1) | bit
            self._bit_pos += 1
            if self._bit_pos >= 8:
                self._bit_pos = 0
                self._byte_pos += 1
        return result

    @property
    def bits_remaining(self) -> int:
        return (len(self._data) - self._byte_pos) * 8 - self._bit_pos


def _decode_exp_golomb(reader: _BitReader) -> int:
    """Decode a single signed exp-Golomb value from the bitstream."""
    if reader.bits_remaining <= 0:
        return 0

    # Count leading zeros
    m = 0
    while reader.bits_remaining > 0:
        bit = reader.read(1)
        if bit == 1:
            break
        m += 1

    # Read m-bit remainder
    remainder = reader.read(m) if m > 0 else 0
    code_num = (1 << m) + remainder - 1

    # Inverse signed mapping: 0→0, 1→+1, 2→-1, 3→+2, 4→-2, ...
    if code_num == 0:
        return 0
    elif code_num % 2 == 1:
        return (code_num + 1) // 2
    else:
        return -(code_num // 2)
