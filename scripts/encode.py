#!/usr/bin/env python3
"""Encode a test signal and save the bitstream.

Usage:
    python scripts/encode.py --signal sine_440 --output output.bin
    python scripts/encode.py --signal chirp --bitrate 64 --no-gpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.data.synthetic import generate_test_signals
from metal_aac.encoder import EncoderConfig, encode


def main():
    parser = argparse.ArgumentParser(description="Encode a test signal")
    parser.add_argument("--signal", type=str, default="sine_440")
    parser.add_argument("--output", type=str, default="output.bin")
    parser.add_argument("--bitrate", type=float, default=128.0)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    signals = generate_test_signals()
    if args.signal not in signals:
        print(f"Unknown signal: {args.signal}")
        print(f"Available: {list(signals.keys())}")
        sys.exit(1)

    sig = signals[args.signal]
    config = EncoderConfig(
        sample_rate=sig["sample_rate"],
        use_gpu=not args.no_gpu,
        target_bitrate_kbps=args.bitrate,
    )

    print(f"Encoding '{args.signal}' ({sig['duration']:.1f}s, {sig['sample_rate']} Hz)")
    print(f"  Target: {args.bitrate} kbps, GPU: {config.use_gpu}")

    result = encode(sig["pcm"], config)

    Path(args.output).write_bytes(result.bitstream)
    print(f"  Output: {args.output} ({len(result.bitstream)} bytes)")
    print(f"  Frames: {result.num_frames}")
    print(f"  Actual bitrate: {result.actual_bitrate_kbps:.1f} kbps")
    print(f"  Timings: {result.timings}")


if __name__ == "__main__":
    main()
