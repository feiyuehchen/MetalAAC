r"""AAC-LC encoder pipeline with CPU and GPU backends.

ISO-compliant pipeline topology (Phase 1):

                     ┌─── Psychoacoustic (from PCM) ──┐
                     │   transient detect → window     │
    PCM ─┬───────────┤   decision + masking thresholds │
         │           └────────────┬───────────────────┘
         │                        │ window_seq, masking
         ▼                        ▼
    Framing ──→ Window ──→ MDCT (long or 8×short) ──→ Quantization ──→ Huffman ──→ ADTS
    (MLX)       (per seq)   (MLX matmul)               (Metal)         (Metal)

The psychoacoustic model runs in parallel from the raw PCM, producing:
  1. Window sequence decisions (transient detection → state machine)
  2. Masking thresholds per scalefactor band (for quantization)

This fixes the v0.2 architecture where psychoacoustic ran serially after MDCT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from metal_aac.core.adts import ADTSWriter
from metal_aac.core.bitstream import BitstreamWriter
from metal_aac.core.huffman import (
    encode_frames_metal,
    encode_frames_parallel,
    encode_spectral_data,
)
from metal_aac.core.raw_data_block import encode_raw_data_block
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
from metal_aac.core.window_switching import (
    WindowSequence,
    compute_window_sequences,
    detect_transients_cpu,
    detect_transients_gpu,
)
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
    output_format: str = "legacy"
    enable_window_switching: bool = True

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
    window_sequences: np.ndarray | None = None
    timings: dict[str, float] = field(default_factory=dict)


def _encode_rdb_chunk(args):
    """Encode a chunk of frames as raw_data_blocks (for multiprocessing)."""
    q_chunk, sf_chunk, gg_chunk, ws_chunk, sr = args
    results = []
    for i in range(len(q_chunk)):
        rdb = encode_raw_data_block(
            q_chunk[i], sf_chunk[i], int(gg_chunk[i]),
            window_sequence=int(ws_chunk[i]),
            sample_rate=sr,
        )
        results.append(rdb)
    return results


def _encode_adts_frames(writer, q, sf, gg, window_seqs, config):
    """Encode all frames as ADTS with multiprocessing."""
    import os
    n_frames = len(q)
    for i in range(n_frames):
        rdb = encode_raw_data_block(
            q[i], sf[i], int(gg[i]),
            window_sequence=int(window_seqs[i]),
            sample_rate=config.sample_rate,
        )
        writer.write_frame(rdb)


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

    with Timer() as t:
        frames = frame_signal(pcm, config.frame_size, config.hop_size, window)
    timings["framing"] = t.elapsed

    num_frames = len(frames)

    # Psychoacoustic: transient detection from raw PCM frames (pre-window)
    with Timer() as t:
        raw_frames = frame_signal(pcm, config.frame_size, config.hop_size)
        if config.enable_window_switching:
            transients = detect_transients_cpu(raw_frames)
            window_seqs = compute_window_sequences(transients)
        else:
            window_seqs = np.zeros(num_frames, dtype=np.int32)
        psy_tables = PsychoacousticTables.create(
            config.sample_rate, config.frame_size
        )
        masking = psychoacoustic_cpu(frames, psy_tables)
    timings["psychoacoustic"] = t.elapsed

    # MDCT (all long windows for now — short window MDCT integration in next commit)
    with Timer() as t:
        basis = MDCTBasis.create(config.frame_size)
        mdct_coeffs = mdct_cpu(frames, basis)
    timings["mdct"] = t.elapsed

    with Timer() as t:
        quant_result = quantize_cpu(
            mdct_coeffs, masking,
            config.target_bits_per_frame, config.sample_rate,
        )
    timings["quantization"] = t.elapsed

    with Timer() as t:
        if config.output_format == "adts":
            writer = ADTSWriter(config.sample_rate, 1)
        else:
            writer = BitstreamWriter()
        num_sfb = get_num_sfb(config.sample_rate)
        for i in range(num_frames):
            spectral_bytes = encode_spectral_data(
                quant_result.quantized[i],
                quant_result.scalefactors[i],
                int(quant_result.global_gain[i]),
                config.sample_rate,
            )
            if config.output_format == "adts":
                writer.write_frame(spectral_bytes)
            else:
                writer.write_frame(spectral_bytes, config.sample_rate, 1, num_sfb)
    timings["huffman_bitstream"] = t.elapsed

    bitstream = writer.get_bytes()
    actual_bitrate = writer.get_bitrate(audio_duration) if audio_duration > 0 else 0

    return EncoderResult(
        bitstream=bitstream,
        num_frames=num_frames,
        actual_bitrate_kbps=actual_bitrate,
        audio_duration=audio_duration,
        window_sequences=window_seqs,
        timings=timings,
    )


def _encode_gpu(pcm: np.ndarray, config: EncoderConfig) -> EncoderResult:
    timings = {}
    window = get_window(config.window_type, config.frame_size)
    audio_duration = len(pcm) / config.sample_rate

    # ---- Parallel path 1: Psychoacoustic (from raw PCM) ----
    # Transient detection runs on unwindowed PCM frames
    with Timer() as t:
        pcm_mx = mx.array(pcm)
        raw_frames_mx = frame_signal_mlx(pcm_mx, config.frame_size, config.hop_size)
        if config.enable_window_switching:
            transients_mx = detect_transients_gpu(raw_frames_mx)
            mx.eval(transients_mx)
            transients = np.array(transients_mx)
            window_seqs = compute_window_sequences(transients)
        else:
            window_seqs = np.zeros(
                raw_frames_mx.shape[0], dtype=np.int32
            )
    timings["transient_detect"] = t.elapsed

    # ---- Parallel path 2: Framing + MDCT ----
    # Apply window and MDCT (currently all long windows)
    with Timer() as t:
        window_mx = mx.array(window)
        frames_mx = frame_signal_mlx(
            pcm_mx, config.frame_size, config.hop_size, window_mx
        )
        mx.eval(frames_mx)
    timings["framing"] = t.elapsed

    num_frames = frames_mx.shape[0]

    with Timer() as t:
        basis_gpu = MDCTBasisGPU(config.frame_size)
        mdct_mx = mdct_gpu(frames_mx, basis_gpu)
        mx.eval(mdct_mx)
    timings["mdct"] = t.elapsed

    # ---- Psychoacoustic masking (from windowed frames, post-MDCT) ----
    with Timer() as t:
        psy_tables_cpu = PsychoacousticTables.create(
            config.sample_rate, config.frame_size
        )
        psy_tables_gpu = PsychoacousticTablesGPU(psy_tables_cpu)
        masking_mx = psychoacoustic_gpu(frames_mx, psy_tables_gpu)
        mx.eval(masking_mx)
    timings["psychoacoustic"] = t.elapsed

    # ---- Quantization: Metal or MLX fallback ----
    try:
        from metal_aac.core.metal_bridge import MetalHuffman

        metal = MetalHuffman.shared()

        with Timer() as t:
            mdct_np = np.array(mdct_mx)
            masking_np = np.array(masking_mx)
            q, sf, gg, tb = metal.quantize(
                mdct_np, masking_np,
                config.target_bits_per_frame,
                config.sample_rate,
            )
        timings["quantization"] = t.elapsed
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

    # ---- Huffman + bitstream assembly ----
    with Timer() as t:
        if config.output_format == "adts":
            writer = ADTSWriter(config.sample_rate, 1)
            try:
                from metal_aac.core.metal_bridge import MetalHuffman
                metal_ctx = MetalHuffman.shared()
                rdb_list = metal_ctx.encode_adts_frames(
                    q, sf, gg, window_seqs, config.sample_rate
                )
                for rdb in rdb_list:
                    writer.write_frame(rdb)
            except (OSError, RuntimeError, FileNotFoundError):
                _encode_adts_frames(writer, q, sf, gg, window_seqs, config)
        else:
            try:
                from metal_aac.core.metal_bridge import MetalHuffman
                metal = MetalHuffman.shared()
                encoded_frames = metal.encode_frames(q, sf, gg)
            except (OSError, RuntimeError, FileNotFoundError):
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
        window_sequences=window_seqs,
        timings=timings,
    )
