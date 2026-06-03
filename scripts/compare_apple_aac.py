#!/usr/bin/env python3
"""Compare AppleAAC with Apple's built-in AAC encoder (afconvert).

Uses macOS CoreAudio via afconvert to encode/decode with Apple's AAC,
then measures quality and throughput against our implementation.

Usage:
    python scripts/compare_metal_aac.py
    python scripts/compare_metal_aac.py --signals sine_440 chirp music_1min
    python scripts/compare_metal_aac.py --bitrate 256
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.data.synthetic import generate_test_signals
from metal_aac.decoder import DecoderConfig, decode
from metal_aac.encoder import EncoderConfig, encode
from metal_aac.metrics.quality import compute_snr, compute_spectral_convergence
from metal_aac.metrics.throughput import Timer, compute_rtf


def write_wav(path: str, pcm: np.ndarray, sample_rate: int = 44100):
    """Write a mono float32 PCM signal to a 16-bit WAV file."""
    pcm_int16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
    n_samples = len(pcm_int16)
    data_size = n_samples * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                            sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_int16.tobytes())


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a 16-bit mono WAV file back to float32."""
    with open(path, "rb") as f:
        riff = f.read(4)
        assert riff == b"RIFF"
        f.read(4)  # file size
        wave = f.read(4)
        assert wave == b"WAVE"

        sample_rate = 44100
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
            elif chunk_id == b"data":
                raw = f.read(chunk_size)
                pcm_int16 = np.frombuffer(raw, dtype=np.int16)
                return pcm_int16.astype(np.float32) / 32767.0, sample_rate
            else:
                f.read(chunk_size)

    return np.array([], dtype=np.float32), sample_rate


def encode_with_metal_aac(
    pcm: np.ndarray,
    sample_rate: int,
    bitrate: int,
    tmpdir: str,
) -> tuple[float, float, np.ndarray]:
    """Encode/decode with Apple's afconvert.

    Returns (encode_time, file_size_bytes, decoded_pcm).
    """
    wav_path = os.path.join(tmpdir, "input.wav")
    m4a_path = os.path.join(tmpdir, "output.m4a")
    out_wav_path = os.path.join(tmpdir, "decoded.wav")

    write_wav(wav_path, pcm, sample_rate)

    # Encode
    with Timer() as t_enc:
        subprocess.run(
            [
                "afconvert",
                "-f", "m4af",
                "-d", "aac",
                "-b", str(bitrate),
                "-s", "3",  # best quality
                wav_path,
                m4a_path,
            ],
            check=True,
            capture_output=True,
        )
    encode_time = t_enc.elapsed

    file_size = os.path.getsize(m4a_path)

    # Decode back
    subprocess.run(
        [
            "afconvert",
            "-f", "WAVE",
            "-d", "LEI16",
            m4a_path,
            out_wav_path,
        ],
        check=True,
        capture_output=True,
    )

    decoded_pcm, _ = read_wav(out_wav_path)
    return encode_time, file_size, decoded_pcm


def compare_signal(
    name: str,
    pcm: np.ndarray,
    sample_rate: int,
    bitrate_kbps: float,
    tmpdir: str,
) -> dict:
    """Compare our encoder vs Apple AAC for a single signal."""
    audio_duration = len(pcm) / sample_rate
    bitrate_bps = int(bitrate_kbps * 1000)
    result = {"signal": name, "duration": audio_duration}

    # --- Our encoder (CPU) ---
    with Timer() as t_ours_cpu:
        enc_cpu = encode(pcm, EncoderConfig(
            sample_rate=sample_rate, use_gpu=False,
            target_bitrate_kbps=bitrate_kbps))
        dec_cpu = decode(enc_cpu.bitstream, DecoderConfig(
            use_gpu=False, output_length=len(pcm)))
    result["ours_cpu_time"] = t_ours_cpu.elapsed
    result["ours_cpu_rtf"] = compute_rtf(t_ours_cpu.elapsed, audio_duration)
    result["ours_cpu_snr"] = compute_snr(pcm, dec_cpu.pcm)
    result["ours_cpu_sc"] = compute_spectral_convergence(pcm, dec_cpu.pcm)
    result["ours_cpu_bitrate"] = enc_cpu.actual_bitrate_kbps

    # --- Our encoder (GPU) ---
    try:
        with Timer() as t_ours_gpu:
            enc_gpu = encode(pcm, EncoderConfig(
                sample_rate=sample_rate, use_gpu=True,
                target_bitrate_kbps=bitrate_kbps))
            dec_gpu = decode(enc_gpu.bitstream, DecoderConfig(
                use_gpu=True, output_length=len(pcm)))
        result["ours_gpu_time"] = t_ours_gpu.elapsed
        result["ours_gpu_rtf"] = compute_rtf(t_ours_gpu.elapsed, audio_duration)
        result["ours_gpu_snr"] = compute_snr(pcm, dec_gpu.pcm)
    except Exception as e:
        result["ours_gpu_error"] = str(e)

    # --- Apple AAC ---
    try:
        enc_time, fsize, apple_pcm = encode_with_metal_aac(
            pcm, sample_rate, bitrate_bps, tmpdir)
        result["apple_encode_time"] = enc_time
        result["apple_rtf"] = compute_rtf(enc_time, audio_duration)
        result["apple_file_size"] = fsize
        result["apple_bitrate"] = fsize * 8 / (audio_duration * 1000)

        min_len = min(len(pcm), len(apple_pcm))
        result["apple_snr"] = compute_snr(pcm[:min_len], apple_pcm[:min_len])
        result["apple_sc"] = compute_spectral_convergence(
            pcm[:min_len], apple_pcm[:min_len])
    except FileNotFoundError:
        result["apple_error"] = "afconvert not found (not macOS?)"
    except subprocess.CalledProcessError as e:
        result["apple_error"] = f"afconvert failed: {e.stderr.decode()}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare with Apple AAC")
    parser.add_argument("--signals", type=str, nargs="*", default=None)
    parser.add_argument("--bitrate", type=float, default=128.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Generating test signals...")
    all_signals = generate_test_signals()

    if args.signals:
        signals = {k: v for k, v in all_signals.items() if k in args.signals}
    else:
        # Default: skip very long signals for quick comparison
        skip = {"music_5min"}
        signals = {k: v for k, v in all_signals.items() if k not in skip}

    print(f"\nComparing {len(signals)} signals at {args.bitrate} kbps")
    print(f"{'Signal':20s} {'Dur':>5s} | {'Ours CPU':>10s} {'Ours GPU':>10s} {'Apple':>10s} | {'Ours SNR':>10s} {'Apple SNR':>10s}")
    print("-" * 90)

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, sig in signals.items():
            r = compare_signal(
                name, sig["pcm"], sig["sample_rate"], args.bitrate, tmpdir)
            results.append(r)

            dur_str = f"{r['duration']:.1f}s"
            ours_cpu = f"{r.get('ours_cpu_rtf', -1):.4f}"
            ours_gpu = f"{r.get('ours_gpu_rtf', -1):.4f}" if "ours_gpu_rtf" in r else "N/A"
            apple_rtf = f"{r.get('apple_rtf', -1):.4f}" if "apple_rtf" in r else "N/A"
            ours_snr = f"{r.get('ours_cpu_snr', 0):.1f}"
            apple_snr = f"{r.get('apple_snr', 0):.1f}" if "apple_snr" in r else "N/A"

            print(f"{name:20s} {dur_str:>5s} | {ours_cpu:>10s} {ours_gpu:>10s} {apple_rtf:>10s} | {ours_snr:>10s} {apple_snr:>10s}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
