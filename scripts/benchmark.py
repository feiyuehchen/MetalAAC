#!/usr/bin/env python3
"""Run the full benchmark suite and output results.

Usage:
    python scripts/benchmark.py [--output outputs/benchmark.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.benchmark import run_benchmark
from metal_aac.data.synthetic import generate_test_signals


def main():
    parser = argparse.ArgumentParser(description="Run AppleAAC benchmark")
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON path"
    )
    parser.add_argument(
        "--signals",
        type=str,
        nargs="*",
        default=None,
        help="Specific signals to benchmark (default: all)",
    )
    parser.add_argument(
        "--bitrate", type=float, default=128.0, help="Target bitrate in kbps"
    )
    parser.add_argument(
        "--n-runs", type=int, default=10, help="Number of timed runs"
    )
    args = parser.parse_args()

    print("Generating test signals...")
    signals = generate_test_signals()

    if args.signals:
        signals = {k: v for k, v in signals.items() if k in args.signals}
        if not signals:
            print(f"No matching signals. Available: {list(generate_test_signals().keys())}")
            sys.exit(1)

    print(f"Running benchmark ({len(signals)} signals, {args.bitrate} kbps target)...\n")
    result = run_benchmark(
        signals,
        target_bitrate=args.bitrate,
        n_runs=args.n_runs,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
