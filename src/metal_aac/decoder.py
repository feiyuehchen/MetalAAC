r"""AAC-LC decoder pipeline with CPU and GPU backends.

Pipeline:
    Bitstream -> Huffman decode -> Dequantize -> IMDCT -> Window -> Overlap-Add -> PCM
                 \____ CPU ____/   \________ GPU (MLX) ________/

The GPU path accelerates dequantization, IMDCT, and overlap-add.
Huffman decoding and bitstream parsing remain on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from metal_aac.core.adts import ADTSReader
from metal_aac.core.bitstream import INDEX_TO_SAMPLE_RATE, BitstreamReader
from metal_aac.core.huffman import decode_frames_metal, decode_spectral_data
from metal_aac.core.mdct import (
    MDCTBasis,
    MDCTBasisGPU,
    imdct_cpu,
    imdct_gpu,
    overlap_add,
)
from metal_aac.core.quantization import (
    dequantize_cpu,
    dequantize_gpu,
    precompute_sf_gains,
)
from metal_aac.metrics.throughput import Timer
from metal_aac.tables.scalefactor_bands import get_sfb_offsets
from metal_aac.tables.windows import get_window


@dataclass
class DecoderConfig:
    frame_size: int = 2048
    hop_size: int = 1024
    window_type: str = "kbd"
    use_gpu: bool = True
    output_length: int | None = None


@dataclass
class DecoderResult:
    pcm: np.ndarray
    sample_rate: int
    num_frames: int
    timings: dict[str, float] = field(default_factory=dict)


def decode(
    bitstream: bytes, config: DecoderConfig | None = None
) -> DecoderResult:
    """Decode AAC bitstream to PCM audio."""
    if config is None:
        config = DecoderConfig()

    if config.use_gpu and HAS_MLX:
        return _decode_gpu(bitstream, config)
    return _decode_cpu(bitstream, config)


def _is_adts(data: bytes) -> bool:
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF0) == 0xF0


def _parse_frames(bitstream: bytes) -> tuple[list[tuple], int, int]:
    """Parse bitstream (auto-detect ADTS vs legacy).

    Returns: (frame_list, sample_rate, num_sfb)
    Each frame is (payload_bytes,).
    """
    if _is_adts(bitstream):
        reader = ADTSReader(bitstream)
        raw_frames = reader.read_all_frames()
        if not raw_frames:
            return [], 44100, 49
        sr = raw_frames[0][0]["sample_rate"]
        from metal_aac.tables.scalefactor_bands import get_num_sfb
        num_sfb = get_num_sfb(sr)
        return [(payload,) for _, payload in raw_frames], sr, num_sfb
    else:
        reader = BitstreamReader(bitstream)
        raw_frames = reader.read_all_frames()
        if not raw_frames:
            return [], 44100, 49
        header = raw_frames[0][0]
        sr = INDEX_TO_SAMPLE_RATE.get(header.sample_rate_index, 44100)
        return [(payload,) for _, payload in raw_frames], sr, header.num_sfb


def _decode_cpu(bitstream: bytes, config: DecoderConfig) -> DecoderResult:
    timings = {}

    with Timer() as t:
        parsed_frames, sample_rate, num_sfb = _parse_frames(bitstream)

        if not parsed_frames:
            return DecoderResult(
                pcm=np.array([], dtype=np.float32),
                sample_rate=44100,
                num_frames=0,
                timings=timings,
            )

        decoded_frames = []
        for (payload,) in parsed_frames:
            quantized, scalefactors, global_gain = decode_spectral_data(
                payload, sample_rate
            )
            decoded_frames.append((quantized, scalefactors, global_gain))
    timings["huffman_bitstream"] = t.elapsed

    num_frames = len(decoded_frames)
    n_coeffs = config.frame_size // 2

    with Timer() as t:
        all_quantized = np.zeros((num_frames, n_coeffs), dtype=np.int32)
        all_sf = np.zeros((num_frames, num_sfb), dtype=np.int32)
        all_gain = np.zeros(num_frames, dtype=np.int32)

        for i, (q, sf, g) in enumerate(decoded_frames):
            all_quantized[i, : len(q)] = q[:n_coeffs]
            all_sf[i, : len(sf)] = sf[:num_sfb]
            all_gain[i] = g

        mdct_coeffs = dequantize_cpu(all_quantized, all_sf, all_gain, sample_rate)
    timings["dequantization"] = t.elapsed

    # IMDCT
    with Timer() as t:
        basis = MDCTBasis.create(config.frame_size)
        time_frames = imdct_cpu(mdct_coeffs, basis)
    timings["imdct"] = t.elapsed

    # Window + overlap-add
    with Timer() as t:
        window = get_window(config.window_type, config.frame_size)
        pcm = overlap_add(
            time_frames, config.hop_size, window, config.output_length
        )
    timings["overlap_add"] = t.elapsed

    return DecoderResult(
        pcm=pcm,
        sample_rate=sample_rate,
        num_frames=num_frames,
        timings=timings,
    )


def _decode_gpu(bitstream: bytes, config: DecoderConfig) -> DecoderResult:
    timings = {}

    with Timer() as t:
        parsed_frames, sample_rate, num_sfb = _parse_frames(bitstream)

        if not parsed_frames:
            return DecoderResult(
                pcm=np.array([], dtype=np.float32),
                sample_rate=44100,
                num_frames=0,
                timings=timings,
            )

        n_coeffs = config.frame_size // 2

        try:
            payloads = [p for (p,) in parsed_frames]
            all_quantized, all_sf, all_gain = decode_frames_metal(
                payloads, n_coeffs, num_sfb
            )
        except (OSError, RuntimeError, FileNotFoundError):
            decoded_frames = []
            for (payload,) in parsed_frames:
                quantized, scalefactors, global_gain = decode_spectral_data(
                    payload, sample_rate
                )
                decoded_frames.append((quantized, scalefactors, global_gain))

            all_quantized = np.zeros((len(decoded_frames), n_coeffs), dtype=np.int32)
            all_sf = np.zeros((len(decoded_frames), header.num_sfb), dtype=np.int32)
            all_gain = np.zeros(len(decoded_frames), dtype=np.int32)
            for i, (q, sf, g) in enumerate(decoded_frames):
                all_quantized[i, : len(q)] = q[:n_coeffs]
                all_sf[i, : len(sf)] = sf[:num_sfb]
                all_gain[i] = g
    timings["huffman_bitstream"] = t.elapsed

    num_frames = len(parsed_frames)
    n_coeffs = config.frame_size // 2
    sfb_offsets = get_sfb_offsets(sample_rate)

    # Dequantize on GPU
    with Timer() as t:
        # Precompute per-coefficient gain factors for each frame
        all_inv_gains = np.zeros((num_frames, n_coeffs), dtype=np.float32)
        for i in range(num_frames):
            _, inv_gains = precompute_sf_gains(
                int(all_gain[i]),
                all_sf[i],
                sfb_offsets,
                n_coeffs,
            )
            all_inv_gains[i] = inv_gains

        q_mx = mx.array(all_quantized)
        inv_gains_mx = mx.array(all_inv_gains)

        mdct_mx = dequantize_gpu(q_mx, inv_gains_mx[0])
        # Per-frame varying gains: process in batch with broadcast
        q_float = q_mx.astype(mx.float32)
        signs = mx.sign(q_float)
        abs_q = mx.abs(q_float)
        mdct_mx = signs * mx.power(abs_q, 4.0 / 3.0) * mx.array(all_inv_gains)
        mx.eval(mdct_mx)
    timings["dequantization"] = t.elapsed

    # IMDCT on GPU
    with Timer() as t:
        basis_gpu = MDCTBasisGPU(config.frame_size)
        time_frames_mx = imdct_gpu(mdct_mx, basis_gpu)
        mx.eval(time_frames_mx)
    timings["imdct"] = t.elapsed

    # Window + overlap-add (CPU, simple accumulation)
    with Timer() as t:
        time_frames = np.array(time_frames_mx)
        window = get_window(config.window_type, config.frame_size)
        pcm = overlap_add(
            time_frames, config.hop_size, window, config.output_length
        )
    timings["overlap_add"] = t.elapsed

    return DecoderResult(
        pcm=pcm,
        sample_rate=sample_rate,
        num_frames=num_frames,
        timings=timings,
    )
