#!/usr/bin/env python3
"""Compare MetalAAC vs ffmpeg (native + AudioToolbox) vs PyAV vs pydub.

Tests all available AAC encoders on the same signals and reports
throughput and quality side by side.

Usage:
    python scripts/compare_all_encoders.py
    python scripts/compare_all_encoders.py --durations 10 60 300
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.encoder import EncoderConfig, encode
from metal_aac.decoder import DecoderConfig, decode
from metal_aac.metrics.quality import compute_snr, compute_spectral_convergence
from metal_aac.metrics.throughput import Timer


def write_wav(path: str, pcm: np.ndarray, sr: int = 44100):
    p16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
    ds = len(p16) * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + ds))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", ds))
        f.write(p16.tobytes())


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with open(path, "rb") as f:
        f.read(4)  # RIFF
        f.read(4)  # size
        f.read(4)  # WAVE
        sr = 44100
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt = f.read(chunk_size)
                sr = struct.unpack("<I", fmt[4:8])[0]
            elif chunk_id == b"data":
                raw = f.read(chunk_size)
                return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0, sr
            else:
                f.read(chunk_size)
    return np.array([], dtype=np.float32), sr


def gen_signal(dur: float, sr: int = 44100) -> np.ndarray:
    rng = np.random.RandomState(42)
    n = int(sr * dur)
    t = np.arange(n) / sr
    nf = rng.randint(3, 8)
    freqs = rng.uniform(100, 4000, nf)
    amps = rng.uniform(0.05, 0.3, nf)
    sig = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
              for f, a in zip(freqs, amps))
    return (sig / (np.max(np.abs(sig)) + 1e-10) * 0.7).astype(np.float32)


# ---- Encoder wrappers ----

def bench_metal_aac(pcm, sr, bitrate, n_runs):
    encode(pcm[:sr * 2], EncoderConfig(use_gpu=True))
    times = []
    for _ in range(n_runs):
        with Timer() as t:
            enc = encode(pcm, EncoderConfig(
                sample_rate=sr, use_gpu=True, target_bitrate_kbps=bitrate))
            dec = decode(enc.bitstream, DecoderConfig(
                use_gpu=True, output_length=len(pcm)))
        times.append(t.elapsed)
    snr = compute_snr(pcm, dec.pcm)
    sc = compute_spectral_convergence(pcm, dec.pcm)
    return np.mean(times), np.std(times), snr, sc


def bench_afconvert(pcm, sr, bitrate, tmpdir, n_runs):
    wav_path = os.path.join(tmpdir, "in.wav")
    m4a_path = os.path.join(tmpdir, "out.m4a")
    dec_path = os.path.join(tmpdir, "dec.wav")
    write_wav(wav_path, pcm, sr)

    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac",
                    "-b", str(int(bitrate * 1000)), "-s", "3",
                    wav_path, m4a_path], check=True, capture_output=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16",
                    m4a_path, dec_path], check=True, capture_output=True)
    dec_pcm, _ = read_wav(dec_path)
    min_len = min(len(pcm), len(dec_pcm))
    snr = compute_snr(pcm[:min_len], dec_pcm[:min_len])
    sc = compute_spectral_convergence(pcm[:min_len], dec_pcm[:min_len])

    for p in [m4a_path, dec_path]:
        if os.path.exists(p):
            os.unlink(p)

    times = []
    for _ in range(n_runs):
        with Timer() as t:
            subprocess.run(["afconvert", "-f", "m4af", "-d", "aac",
                            "-b", str(int(bitrate * 1000)), "-s", "3",
                            wav_path, m4a_path], check=True, capture_output=True)
        times.append(t.elapsed)
        os.unlink(m4a_path)
    return np.mean(times), np.std(times), snr, sc


def bench_ffmpeg_native(pcm, sr, bitrate, tmpdir, n_runs):
    """ffmpeg with built-in AAC encoder (libfdk_aac or native)."""
    wav_path = os.path.join(tmpdir, "in.wav")
    m4a_path = os.path.join(tmpdir, "out_ff.m4a")
    dec_path = os.path.join(tmpdir, "dec_ff.wav")
    write_wav(wav_path, pcm, sr)

    # Encode + decode once for quality
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac",
                    "-b:a", f"{int(bitrate)}k", m4a_path],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", m4a_path, dec_path],
                   check=True, capture_output=True)
    dec_pcm, _ = read_wav(dec_path)
    min_len = min(len(pcm), len(dec_pcm))
    snr = compute_snr(pcm[:min_len], dec_pcm[:min_len])
    sc = compute_spectral_convergence(pcm[:min_len], dec_pcm[:min_len])

    for p in [m4a_path, dec_path]:
        if os.path.exists(p):
            os.unlink(p)

    times = []
    for _ in range(n_runs):
        with Timer() as t:
            subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac",
                            "-b:a", f"{int(bitrate)}k", m4a_path],
                           check=True, capture_output=True)
        times.append(t.elapsed)
        os.unlink(m4a_path)
    return np.mean(times), np.std(times), snr, sc


def bench_ffmpeg_at(pcm, sr, bitrate, tmpdir, n_runs):
    """ffmpeg with AudioToolbox AAC encoder (macOS only)."""
    wav_path = os.path.join(tmpdir, "in.wav")
    m4a_path = os.path.join(tmpdir, "out_at.m4a")
    dec_path = os.path.join(tmpdir, "dec_at.wav")
    write_wav(wav_path, pcm, sr)

    try:
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac_at",
                        "-b:a", f"{int(bitrate)}k", m4a_path],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return None, None, None, None

    subprocess.run(["ffmpeg", "-y", "-i", m4a_path, dec_path],
                   check=True, capture_output=True)
    dec_pcm, _ = read_wav(dec_path)
    min_len = min(len(pcm), len(dec_pcm))
    snr = compute_snr(pcm[:min_len], dec_pcm[:min_len])
    sc = compute_spectral_convergence(pcm[:min_len], dec_pcm[:min_len])

    for p in [m4a_path, dec_path]:
        if os.path.exists(p):
            os.unlink(p)

    times = []
    for _ in range(n_runs):
        with Timer() as t:
            subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac_at",
                            "-b:a", f"{int(bitrate)}k", m4a_path],
                           check=True, capture_output=True)
        times.append(t.elapsed)
        os.unlink(m4a_path)
    return np.mean(times), np.std(times), snr, sc


def bench_pyav(pcm, sr, bitrate, n_runs):
    """PyAV (Python ffmpeg bindings) AAC encoder."""
    try:
        import av
    except ImportError:
        return None, None, None, None

    def _encode_decode(pcm_data, sample_rate, target_bitrate):
        buf = io.BytesIO()
        out_container = av.open(buf, mode="w", format="adts")
        stream = out_container.add_stream("aac", rate=sample_rate)
        stream.bit_rate = int(target_bitrate * 1000)
        stream.layout = "mono"

        p16 = np.clip(pcm_data * 32767, -32768, 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(
            p16.reshape(1, -1), format="s16", layout="mono"
        )
        frame.rate = sample_rate

        chunk_size = 1024
        for i in range(0, len(p16), chunk_size):
            chunk = p16[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            frame = av.AudioFrame.from_ndarray(
                chunk.reshape(1, -1), format="s16", layout="mono"
            )
            frame.rate = sample_rate
            for packet in stream.encode(frame):
                out_container.mux(packet)

        for packet in stream.encode(None):
            out_container.mux(packet)
        out_container.close()

        buf.seek(0)
        in_container = av.open(buf, format="aac")
        decoded = []
        for frame in in_container.decode(audio=0):
            arr = frame.to_ndarray()
            decoded.append(arr.flatten())
        in_container.close()
        if decoded:
            return np.concatenate(decoded).astype(np.float32) / 32767.0
        return np.zeros_like(pcm_data)

    dec_pcm = _encode_decode(pcm, sr, bitrate)
    min_len = min(len(pcm), len(dec_pcm))
    snr = compute_snr(pcm[:min_len], dec_pcm[:min_len])
    sc = compute_spectral_convergence(pcm[:min_len], dec_pcm[:min_len])

    times = []
    for _ in range(n_runs):
        with Timer() as t:
            _encode_decode(pcm, sr, bitrate)
        times.append(t.elapsed)
    return np.mean(times), np.std(times), snr, sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--durations", type=int, nargs="*", default=[10, 60, 300])
    parser.add_argument("--bitrate", type=float, default=128.0)
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    sr = 44100
    encoders = [
        ("MetalAAC (ours)", bench_metal_aac),
        ("afconvert", None),
        ("ffmpeg native", None),
        ("ffmpeg AudioToolbox", None),
        ("PyAV", bench_pyav),
    ]

    print(f"Comparing AAC encoders at {args.bitrate} kbps, {args.n_runs} runs each")
    print()

    for dur in args.durations:
        pcm = gen_signal(dur, sr)
        print(f"{'='*90}")
        print(f"Duration: {dur}s  ({len(pcm):,} samples)")
        print(f"{'='*90}")
        print(f"{'Encoder':<25s} {'Time (ms)':>12s} {'RTF':>10s} {'SNR (dB)':>10s} {'SC':>8s}")
        print(f"{'-'*70}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # MetalAAC
            m, s, snr, sc = bench_metal_aac(pcm, sr, args.bitrate, args.n_runs)
            print(f"{'MetalAAC (GPU+Metal)':<25s} {m*1000:>8.1f}±{s*1000:.1f} {m/dur:>10.6f} {snr:>10.1f} {sc:>8.4f}")
            metal_time = m

            # afconvert
            m, s, snr, sc = bench_afconvert(pcm, sr, args.bitrate, tmpdir, args.n_runs)
            print(f"{'Apple afconvert':<25s} {m*1000:>8.1f}±{s*1000:.1f} {m/dur:>10.6f} {snr:>10.1f} {sc:>8.4f}")

            # ffmpeg native AAC
            m, s, snr, sc = bench_ffmpeg_native(pcm, sr, args.bitrate, tmpdir, args.n_runs)
            print(f"{'ffmpeg (native aac)':<25s} {m*1000:>8.1f}±{s*1000:.1f} {m/dur:>10.6f} {snr:>10.1f} {sc:>8.4f}")

            # ffmpeg AudioToolbox
            m, s, snr, sc = bench_ffmpeg_at(pcm, sr, args.bitrate, tmpdir, args.n_runs)
            if m is not None:
                print(f"{'ffmpeg (aac_at)':<25s} {m*1000:>8.1f}±{s*1000:.1f} {m/dur:>10.6f} {snr:>10.1f} {sc:>8.4f}")
            else:
                print(f"{'ffmpeg (aac_at)':<25s} {'N/A':>12s}")

            # PyAV
            m, s, snr, sc = bench_pyav(pcm, sr, args.bitrate, args.n_runs)
            if m is not None:
                print(f"{'PyAV (Python ffmpeg)':<25s} {m*1000:>8.1f}±{s*1000:.1f} {m/dur:>10.6f} {snr:>10.1f} {sc:>8.4f}")
            else:
                print(f"{'PyAV (Python ffmpeg)':<25s} {'N/A':>12s}")

        print()


if __name__ == "__main__":
    main()
