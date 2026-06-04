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
from metal_aac.core.raw_data_block import encode_raw_data_block, encode_raw_data_block_iso
from metal_aac.core.mdct import (
    MDCTBasis,
    MDCTBasisGPU,
    frame_signal,
    frame_signal_mlx,
    mdct_cpu,
    mdct_gpu,
    mdct_short_gpu,
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


def _encode_adts_frames(writer, q, sf, gg, window_seqs, config):
    """CPU fallback: encode all frames as ADTS raw_data_blocks."""
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
            for i in range(num_frames):
                rdb = encode_raw_data_block(
                    quant_result.quantized[i],
                    quant_result.scalefactors[i],
                    int(quant_result.global_gain[i]),
                    window_sequence=int(window_seqs[i]),
                    sample_rate=config.sample_rate,
                )
                writer.write_frame(rdb)
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
    with Timer() as t:
        window_mx = mx.array(window)
        frames_mx = frame_signal_mlx(
            pcm_mx, config.frame_size, config.hop_size, window_mx
        )
        mx.eval(frames_mx)
    timings["framing"] = t.elapsed

    num_frames = frames_mx.shape[0]

    # MDCT: long windows for most frames, short (8x256) for transient frames
    with Timer() as t:
        has_short = config.enable_window_switching and np.any(window_seqs == 2)
        if has_short:
            long_mask = (window_seqs != 2)
            short_mask = (window_seqs == 2)
            basis_long = MDCTBasisGPU(config.frame_size)
            basis_short = MDCTBasisGPU(256)

            mdct_np = np.zeros((num_frames, config.n_coeffs), dtype=np.float32)

            if np.any(long_mask):
                long_idx = np.where(long_mask)[0]
                long_frames = mx.array(np.array(frames_mx)[long_idx])
                long_mdct = np.array(mdct_gpu(long_frames, basis_long))
                mx.eval(mdct_gpu(long_frames, basis_long))
                mdct_np[long_idx] = long_mdct

            if np.any(short_mask):
                short_idx = np.where(short_mask)[0]
                short_frames = mx.array(np.array(frames_mx)[short_idx])
                short_mdct = mdct_short_gpu(short_frames, basis_short)
                mx.eval(short_mdct)
                # Flatten 8x128 -> 1024 so quantizer sees uniform shape
                mdct_np[short_idx] = np.array(short_mdct).reshape(-1, config.n_coeffs)

            mdct_mx = mx.array(mdct_np)
            mx.eval(mdct_mx)
        else:
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

    # ---- Quantization ----
    if config.output_format == "adts":
        # ISO-native quantizer: sf is the only parameter per band.
        # No mapping needed — sf goes directly to ADTS bitstream.
        with Timer() as t:
            quant_result = quantize_batch_gpu(
                mdct_mx, masking_mx,
                config.target_bits_per_frame,
                config.sample_rate,
            )
        timings["quantization"] = t.elapsed
        q = quant_result.quantized
        sf = quant_result.scalefactors   # direct ISO sf values (100-255)
        gg = quant_result.global_gain    # mean of ISO SFs for DPCM anchor
    else:
        # Legacy quantizer for internal round-trip
        try:
            from metal_aac.core.metal_bridge import MetalHuffman
            metal = MetalHuffman.shared()
            with Timer() as t:
                mdct_np = np.array(mdct_mx)
                masking_np = np.array(masking_mx)
                q, sf, gg, _ = metal.quantize(
                    mdct_np, masking_np,
                    config.target_bits_per_frame, config.sample_rate,
                )
            timings["quantization"] = t.elapsed
        except (OSError, RuntimeError, FileNotFoundError):
            # CPU fallback with old formula
            with Timer() as t:
                quant_result = quantize_cpu(
                    np.array(mdct_mx), np.array(masking_mx),
                    config.target_bits_per_frame, config.sample_rate,
                )
            timings["quantization"] = t.elapsed
            q = quant_result.quantized
            sf = quant_result.scalefactors
            gg = quant_result.global_gain

    # ---- Huffman + bitstream assembly ----
    with Timer() as t:
        if config.output_format == "adts":
            writer = ADTSWriter(config.sample_rate, 1)
            # ISO path: sf are already ISO convention, use _iso encoder
            for i in range(len(q)):
                rdb = encode_raw_data_block_iso(
                    q[i], sf[i], int(gg[i]),
                    window_sequence=int(window_seqs[i]),
                    sample_rate=config.sample_rate,
                )
                writer.write_frame(rdb)
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
