#!/usr/bin/env python3
"""Decode a bitstream back to PCM and measure quality.

Usage:
    python scripts/decode.py --input output.bin --signal sine_440
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.data.synthetic import generate_test_signals
from metal_aac.decoder import DecoderConfig, decode
from metal_aac.metrics.quality import compute_snr, compute_spectral_convergence


def main():
    parser = argparse.ArgumentParser(description="Decode a bitstream")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--signal", type=str, default=None, help="Compare to this test signal")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    bitstream = Path(args.input).read_bytes()
    config = DecoderConfig(use_gpu=not args.no_gpu)

    if args.signal:
        signals = generate_test_signals()
        if args.signal in signals:
            config.output_length = len(signals[args.signal]["pcm"])

    print(f"Decoding '{args.input}' ({len(bitstream)} bytes)")
    result = decode(bitstream, config)

    print(f"  Frames: {result.num_frames}")
    print(f"  Sample rate: {result.sample_rate}")
    print(f"  Output samples: {len(result.pcm)}")
    print(f"  Timings: {result.timings}")

    if args.signal:
        signals = generate_test_signals()
        if args.signal in signals:
            original = signals[args.signal]["pcm"]
            snr = compute_snr(original, result.pcm)
            sc = compute_spectral_convergence(original, result.pcm)
            print(f"\n  Quality vs '{args.signal}':")
            print(f"    SNR: {snr:.1f} dB")
            print(f"    Spectral Convergence: {sc:.4f}")


if __name__ == "__main__":
    main()
