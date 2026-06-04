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
from metal_aac.core.raw_data_block import decode_raw_data_block_iso, decode_cpe_iso
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
    dequantize_iso_cpu,
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


def _parse_frames(bitstream: bytes) -> tuple[list[tuple], int, int, bool, int]:
    """Parse bitstream (auto-detect ADTS vs legacy).

    Returns: (frame_list, sample_rate, num_sfb, is_adts, num_channels)
    """
    if _is_adts(bitstream):
        reader = ADTSReader(bitstream)
        raw_frames = reader.read_all_frames()
        if not raw_frames:
            return [], 44100, 49, True, 1
        sr = raw_frames[0][0]["sample_rate"]
        nch = raw_frames[0][0].get("channel_configuration", 1)
        from metal_aac.tables.scalefactor_bands import get_num_sfb
        num_sfb = get_num_sfb(sr)
        return [(payload,) for _, payload in raw_frames], sr, num_sfb, True, nch
    else:
        reader = BitstreamReader(bitstream)
        raw_frames = reader.read_all_frames()
        if not raw_frames:
            return [], 44100, 49, False, 1
        header = raw_frames[0][0]
        sr = INDEX_TO_SAMPLE_RATE.get(header.sample_rate_index, 44100)
        return [(payload,) for _, payload in raw_frames], sr, header.num_sfb, False, 1


def _decode_channel(
    decoded_frames: list, num_sfb: int, config: DecoderConfig,
    sample_rate: int, is_adts: bool, timings: dict | None = None,
) -> np.ndarray:
    """Dequantize + IMDCT + overlap-add for one channel."""
    n_coeffs = config.frame_size // 2
    num_frames = len(decoded_frames)
    all_q = np.zeros((num_frames, n_coeffs), dtype=np.int32)
    all_sf = np.zeros((num_frames, num_sfb), dtype=np.int32)
    all_gain = np.zeros(num_frames, dtype=np.int32)
    for i, (q, sf, g) in enumerate(decoded_frames):
        all_q[i, :len(q)] = q[:n_coeffs]
        all_sf[i, :len(sf)] = sf[:num_sfb]
        all_gain[i] = g

    with Timer() as t:
        if is_adts:
            mdct = dequantize_iso_cpu(all_q, all_sf, sample_rate)
        else:
            mdct = dequantize_cpu(all_q, all_sf, all_gain, sample_rate)
    if timings is not None:
        timings["dequantization"] = timings.get("dequantization", 0) + t.elapsed

    with Timer() as t:
        basis = MDCTBasis.create(config.frame_size)
        time_frames = imdct_cpu(mdct, basis)
    if timings is not None:
        timings["imdct"] = timings.get("imdct", 0) + t.elapsed

    with Timer() as t:
        window = get_window(config.window_type, config.frame_size)
        pcm = overlap_add(time_frames, config.hop_size, window, config.output_length)
    if timings is not None:
        timings["overlap_add"] = timings.get("overlap_add", 0) + t.elapsed

    return pcm


def _decode_cpu(bitstream: bytes, config: DecoderConfig) -> DecoderResult:
    timings = {}

    with Timer() as t:
        parsed_frames, sample_rate, num_sfb, is_adts, num_channels = _parse_frames(bitstream)

        if not parsed_frames:
            return DecoderResult(
                pcm=np.array([], dtype=np.float32),
                sample_rate=44100,
                num_frames=0,
                timings=timings,
            )

        if is_adts and num_channels == 2:
            frames_l, frames_r = [], []
            for (payload,) in parsed_frames:
                ch0, ch1 = decode_cpe_iso(payload, sample_rate)
                frames_l.append(ch0)
                frames_r.append(ch1)
            timings["huffman_bitstream"] = t.elapsed

            pcm_l = _decode_channel(frames_l, num_sfb, config, sample_rate, True, timings)
            pcm_r = _decode_channel(frames_r, num_sfb, config, sample_rate, True, timings)
            pcm = np.column_stack([pcm_l, pcm_r])

            return DecoderResult(
                pcm=pcm, sample_rate=sample_rate,
                num_frames=len(parsed_frames), timings=timings,
            )

        decoded_frames = []
        for (payload,) in parsed_frames:
            if is_adts:
                quantized, scalefactors, global_gain = decode_raw_data_block_iso(
                    payload, sample_rate
                )
            else:
                quantized, scalefactors, global_gain = decode_spectral_data(
                    payload, sample_rate
                )
            decoded_frames.append((quantized, scalefactors, global_gain))
    timings["huffman_bitstream"] = t.elapsed

    pcm = _decode_channel(decoded_frames, num_sfb, config, sample_rate, is_adts, timings)

    return DecoderResult(
        pcm=pcm,
        sample_rate=sample_rate,
        num_frames=len(decoded_frames),
        timings=timings,
    )


def _decode_gpu(bitstream: bytes, config: DecoderConfig) -> DecoderResult:
    timings = {}

    with Timer() as t:
        parsed_frames, sample_rate, num_sfb, is_adts, num_channels = _parse_frames(bitstream)

        if not parsed_frames:
            return DecoderResult(
                pcm=np.array([], dtype=np.float32),
                sample_rate=44100,
                num_frames=0,
                timings=timings,
            )

        n_coeffs = config.frame_size // 2

        if is_adts and num_channels == 2:
            frames_l, frames_r = [], []
            for (payload,) in parsed_frames:
                ch0, ch1 = decode_cpe_iso(payload, sample_rate)
                frames_l.append(ch0)
                frames_r.append(ch1)
            timings["huffman_bitstream"] = t.elapsed
            pcm_l = _decode_channel(frames_l, num_sfb, config, sample_rate, True, timings)
            pcm_r = _decode_channel(frames_r, num_sfb, config, sample_rate, True, timings)
            pcm = np.column_stack([pcm_l, pcm_r])
            return DecoderResult(pcm=pcm, sample_rate=sample_rate,
                                num_frames=len(parsed_frames), timings=timings)

        if is_adts:
            decoded_frames = []
            for (payload,) in parsed_frames:
                quantized, scalefactors, global_gain = decode_raw_data_block_iso(
                    payload, sample_rate
                )
                decoded_frames.append((quantized, scalefactors, global_gain))
            all_quantized = np.zeros((len(decoded_frames), n_coeffs), dtype=np.int32)
            all_sf = np.zeros((len(decoded_frames), num_sfb), dtype=np.int32)
            all_gain = np.zeros(len(decoded_frames), dtype=np.int32)
            for i, (q, sf, g) in enumerate(decoded_frames):
                all_quantized[i, : len(q)] = q[:n_coeffs]
                all_sf[i, : len(sf)] = sf[:num_sfb]
                all_gain[i] = g
        else:
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
                all_sf = np.zeros((len(decoded_frames), num_sfb), dtype=np.int32)
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
        q_mx = mx.array(all_quantized)
        q_float = q_mx.astype(mx.float32)
        signs = mx.sign(q_float)
        abs_q = mx.abs(q_float)

        if is_adts:
            sfb_map = np.zeros(n_coeffs, dtype=np.int32)
            for sb in range(min(num_sfb, len(sfb_offsets) - 1)):
                lo, hi = sfb_offsets[sb], min(sfb_offsets[sb + 1], n_coeffs)
                sfb_map[lo:hi] = sb
            per_coeff_sf = all_sf[:, sfb_map].astype(np.float32)
            scale = np.power(2.0, (per_coeff_sf - 200.0) / 4.0)
            mdct_mx = signs * mx.power(abs_q, 4.0 / 3.0) * mx.array(scale)
        else:
            all_inv_gains = np.zeros((num_frames, n_coeffs), dtype=np.float32)
            for i in range(num_frames):
                _, inv_gains = precompute_sf_gains(
                    int(all_gain[i]), all_sf[i], sfb_offsets, n_coeffs,
                )
                all_inv_gains[i] = inv_gains
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
