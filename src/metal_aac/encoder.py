r"""AAC-LC encoder pipeline with CPU and GPU backends.

Pipeline:
    PCM -> Frame/Window -> MDCT -> Psychoacoustic -> Quantize -> Huffman -> Bitstream
           \___________ GPU (MLX) ___________/      \____ CPU ____/

The GPU path accelerates the compute-intensive stages (MDCT, psychoacoustic
model, quantization formula). Huffman coding and bitstream packing remain
on CPU as they are inherently sequential.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from metal_aac.core.bitstream import BitstreamWriter
from metal_aac.core.huffman import (
    encode_frames_metal,
    encode_frames_parallel,
    encode_spectral_data,
)
from metal_aac.core.mdct import (
    MDCTBasis,
    MDCTBasisGPU,
    frame_signal,
    frame_signal_mlx,
    mdct_cpu,
    mdct_gpu,
)
from metal_aac.core.psychoacoustic import (
    PsychoacousticTables,
    PsychoacousticTablesGPU,
    psychoacoustic_cpu,
    psychoacoustic_gpu,
)
from metal_aac.core.quantization import quantize_batch_gpu, quantize_cpu
from metal_aac.metrics.throughput import Timer
from metal_aac.tables.scalefactor_bands import get_num_sfb
from metal_aac.tables.windows import get_window


@dataclass
class EncoderConfig:
    sample_rate: int = 44100
    frame_size: int = 2048
    hop_size: int = 1024
    window_type: str = "kbd"
    target_bitrate_kbps: float = 128.0
    use_gpu: bool = True

    @property
    def n_coeffs(self) -> int:
        return self.frame_size // 2

    @property
    def target_bits_per_frame(self) -> int:
        frame_duration = self.hop_size / self.sample_rate
        return int(self.target_bitrate_kbps * 1000 * frame_duration)


@dataclass
class EncoderResult:
    bitstream: bytes
    num_frames: int
    actual_bitrate_kbps: float
    audio_duration: float
    timings: dict[str, float] = field(default_factory=dict)


def encode(pcm: np.ndarray, config: EncoderConfig | None = None) -> EncoderResult:
    """Encode PCM audio to AAC bitstream.

    pcm: (num_samples,) float32, mono
    """
    if config is None:
        config = EncoderConfig()

    if config.use_gpu and HAS_MLX:
        return _encode_gpu(pcm, config)
    return _encode_cpu(pcm, config)


def _encode_cpu(pcm: np.ndarray, config: EncoderConfig) -> EncoderResult:
    timings = {}
    window = get_window(config.window_type, config.frame_size)
    audio_duration = len(pcm) / config.sample_rate

    # Frame and window
    with Timer() as t:
        frames = frame_signal(pcm, config.frame_size, config.hop_size, window)
    timings["framing"] = t.elapsed

    num_frames = len(frames)

    # MDCT
    with Timer() as t:
        basis = MDCTBasis.create(config.frame_size)
        mdct_coeffs = mdct_cpu(frames, basis)
    timings["mdct"] = t.elapsed

    # Psychoacoustic model
    with Timer() as t:
        psy_tables = PsychoacousticTables.create(
            config.sample_rate, config.frame_size
        )
        masking = psychoacoustic_cpu(frames, psy_tables)
    timings["psychoacoustic"] = t.elapsed

    # Quantization + Huffman + bitstream (per-frame, sequential)
    with Timer() as t:
        quant_result = quantize_cpu(
            mdct_coeffs,
            masking,
            config.target_bits_per_frame,
            config.sample_rate,
        )
    timings["quantization"] = t.elapsed

    with Timer() as t:
        writer = BitstreamWriter()
        num_sfb = get_num_sfb(config.sample_rate)
        for i in range(num_frames):
            spectral_bytes = encode_spectral_data(
                quant_result.quantized[i],
                quant_result.scalefactors[i],
                int(quant_result.global_gain[i]),
                config.sample_rate,
            )
            writer.write_frame(spectral_bytes, config.sample_rate, 1, num_sfb)
    timings["huffman_bitstream"] = t.elapsed

    bitstream = writer.get_bytes()
    actual_bitrate = writer.get_bitrate(audio_duration) if audio_duration > 0 else 0

    return EncoderResult(
        bitstream=bitstream,
        num_frames=num_frames,
        actual_bitrate_kbps=actual_bitrate,
        audio_duration=audio_duration,
        timings=timings,
    )


def _encode_gpu(pcm: np.ndarray, config: EncoderConfig) -> EncoderResult:
    timings = {}
    window = get_window(config.window_type, config.frame_size)
    audio_duration = len(pcm) / config.sample_rate

    # Frame and window on GPU (MLX gather, 3x faster than numpy loop)
    with Timer() as t:
        pcm_mx = mx.array(pcm)
        window_mx = mx.array(window)
        frames_mx = frame_signal_mlx(
            pcm_mx, config.frame_size, config.hop_size, window_mx
        )
        mx.eval(frames_mx)
    timings["framing"] = t.elapsed

    num_frames = frames_mx.shape[0]

    # MDCT on GPU
    with Timer() as t:
        basis_gpu = MDCTBasisGPU(config.frame_size)
        mdct_mx = mdct_gpu(frames_mx, basis_gpu)
        mx.eval(mdct_mx)
    timings["mdct"] = t.elapsed

    # Psychoacoustic model on GPU
    with Timer() as t:
        psy_tables_cpu = PsychoacousticTables.create(
            config.sample_rate, config.frame_size
        )
        psy_tables_gpu = PsychoacousticTablesGPU(psy_tables_cpu)
        masking_mx = psychoacoustic_gpu(frames_mx, psy_tables_gpu)
        mx.eval(masking_mx)
    timings["psychoacoustic"] = t.elapsed

    # Quantization + Huffman: try Metal for both, fallback to MLX+CPU
    try:
        from metal_aac.core.metal_bridge import MetalHuffman

        metal = MetalHuffman.shared()

        # Metal quantization (single GPU dispatch, no Python loop)
        with Timer() as t:
            mdct_np = np.array(mdct_mx)
            masking_np = np.array(masking_mx)
            q, sf, gg, tb = metal.quantize(
                mdct_np, masking_np,
                config.target_bits_per_frame,
                config.sample_rate,
            )
        timings["quantization"] = t.elapsed

        # Metal Huffman encoding
        with Timer() as t:
            encoded_frames = metal.encode_frames(q, sf, gg)
    except (OSError, RuntimeError, FileNotFoundError):
        with Timer() as t:
            quant_result = quantize_batch_gpu(
                mdct_mx, masking_mx,
                config.target_bits_per_frame,
                config.sample_rate,
            )
        timings["quantization"] = t.elapsed
        q = quant_result.quantized
        sf = quant_result.scalefactors
        gg = quant_result.global_gain

        with Timer() as t:
            encoded_frames = encode_frames_parallel(
                q, sf, gg, config.sample_rate,
            )

    writer = BitstreamWriter()
    num_sfb = get_num_sfb(config.sample_rate)
    for spectral_bytes in encoded_frames:
        writer.write_frame(spectral_bytes, config.sample_rate, 1, num_sfb)
    timings["huffman_bitstream"] = t.elapsed

    bitstream = writer.get_bytes()
    actual_bitrate = writer.get_bitrate(audio_duration) if audio_duration > 0 else 0

    return EncoderResult(
        bitstream=bitstream,
        num_frames=num_frames,
        actual_bitrate_kbps=actual_bitrate,
        audio_duration=audio_duration,
        timings=timings,
    )
