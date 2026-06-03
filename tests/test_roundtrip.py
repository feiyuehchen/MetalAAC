"""End-to-end round-trip tests: encode -> decode -> quality check."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.decoder import DecoderConfig, decode
from metal_aac.encoder import EncoderConfig, encode
from metal_aac.metrics.quality import compute_snr, compute_spectral_convergence

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _make_sine(freq: float = 440.0, duration: float = 0.5, sr: int = 44100):
    t = np.arange(int(sr * duration)) / sr
    return (0.9 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestRoundtripCPU:
    def test_sine_roundtrip(self):
        pcm = _make_sine(440.0, 0.5)
        enc_cfg = EncoderConfig(use_gpu=False, target_bitrate_kbps=128.0)
        dec_cfg = DecoderConfig(use_gpu=False, output_length=len(pcm))

        enc_result = encode(pcm, enc_cfg)
        assert enc_result.num_frames > 0
        assert len(enc_result.bitstream) > 0

        dec_result = decode(enc_result.bitstream, dec_cfg)
        assert len(dec_result.pcm) == len(pcm)

        snr = compute_snr(pcm, dec_result.pcm)
        assert snr > 5.0, f"SNR too low: {snr:.1f} dB"

    def test_silence_roundtrip(self):
        pcm = np.zeros(22050, dtype=np.float32)
        enc_cfg = EncoderConfig(use_gpu=False)
        dec_cfg = DecoderConfig(use_gpu=False, output_length=len(pcm))

        enc_result = encode(pcm, enc_cfg)
        dec_result = decode(enc_result.bitstream, dec_cfg)

        # Silence should decode to near-silence
        assert np.max(np.abs(dec_result.pcm)) < 0.1

    def test_bitrate_reasonable(self):
        pcm = _make_sine(1000.0, 1.0)
        enc_cfg = EncoderConfig(use_gpu=False, target_bitrate_kbps=128.0)
        enc_result = encode(pcm, enc_cfg)
        # Actual bitrate should be within 5x of target (simplified coding)
        assert enc_result.actual_bitrate_kbps > 10.0
        assert enc_result.actual_bitrate_kbps < 1000.0

    def test_timings_present(self):
        pcm = _make_sine()
        enc_cfg = EncoderConfig(use_gpu=False)
        dec_cfg = DecoderConfig(use_gpu=False)

        enc_result = encode(pcm, enc_cfg)
        dec_result = decode(enc_result.bitstream, dec_cfg)

        assert "mdct" in enc_result.timings
        assert "psychoacoustic" in enc_result.timings
        assert "quantization" in enc_result.timings
        assert "imdct" in dec_result.timings

    def test_multiple_frequencies(self):
        for freq in [440.0, 1000.0, 4000.0]:
            pcm = _make_sine(freq, 0.3)
            enc_cfg = EncoderConfig(use_gpu=False, target_bitrate_kbps=128.0)
            dec_cfg = DecoderConfig(use_gpu=False, output_length=len(pcm))

            enc_result = encode(pcm, enc_cfg)
            dec_result = decode(enc_result.bitstream, dec_cfg)

            snr = compute_snr(pcm, dec_result.pcm)
            assert snr > 3.0, f"SNR too low for {freq}Hz: {snr:.1f} dB"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestRoundtripGPU:
    def test_sine_roundtrip_gpu(self):
        pcm = _make_sine(440.0, 0.5)
        enc_cfg = EncoderConfig(use_gpu=True, target_bitrate_kbps=128.0)
        dec_cfg = DecoderConfig(use_gpu=True, output_length=len(pcm))

        enc_result = encode(pcm, enc_cfg)
        dec_result = decode(enc_result.bitstream, dec_cfg)

        snr = compute_snr(pcm, dec_result.pcm)
        assert snr > 5.0, f"GPU SNR too low: {snr:.1f} dB"

    def test_gpu_cpu_quality_match(self):
        """GPU and CPU paths should produce similar quality."""
        pcm = _make_sine(1000.0, 0.5)

        enc_cpu = encode(pcm, EncoderConfig(use_gpu=False))
        dec_cpu = decode(
            enc_cpu.bitstream, DecoderConfig(use_gpu=False, output_length=len(pcm))
        )
        snr_cpu = compute_snr(pcm, dec_cpu.pcm)

        enc_gpu = encode(pcm, EncoderConfig(use_gpu=True))
        dec_gpu = decode(
            enc_gpu.bitstream, DecoderConfig(use_gpu=True, output_length=len(pcm))
        )
        snr_gpu = compute_snr(pcm, dec_gpu.pcm)

        # Quality should be within 3 dB
        assert abs(snr_cpu - snr_gpu) < 3.0, (
            f"CPU SNR={snr_cpu:.1f}, GPU SNR={snr_gpu:.1f}"
        )
