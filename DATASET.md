# Dataset Specification

## Overview
- **Name**: AppleAAC Synthetic Test Signals
- **Version**: v1 (2026-06-03)
- **Source**: Programmatically generated (deterministic)
- **License**: N/A (synthetic)

## Task Formulation
- **Input**: Raw PCM audio (mono, float32, normalized to [-1, 1])
- **Output**: AAC-encoded bitstream; decoded PCM for round-trip quality measurement
- **Modeling target**: GPU-accelerated AAC-LC encoding/decoding on Apple Silicon

## Test Signals

All signals are generated deterministically by `src/apple_aac/data/synthetic.py::generate_test_signals()`.
Parameters are fixed; re-running must produce bit-identical output.

| Signal ID | Description | Sample Rate | Duration | Channels |
|-----------|-------------|-------------|----------|----------|
| sine_440 | 440 Hz sine wave | 44100 | 1.0 s | 1 |
| sine_1k | 1000 Hz sine wave | 44100 | 1.0 s | 1 |
| sine_4k | 4000 Hz sine wave | 44100 | 1.0 s | 1 |
| multitone | 440 + 1000 + 4000 Hz | 44100 | 1.0 s | 1 |
| chirp | Linear sweep 20-20000 Hz | 44100 | 2.0 s | 1 |
| white_noise | Gaussian white noise (seed=42) | 44100 | 1.0 s | 1 |
| pink_noise | 1/f noise (seed=42) | 44100 | 1.0 s | 1 |
| silence | All zeros | 44100 | 0.5 s | 1 |
| impulse | Single sample = 1.0 at t=0.5s | 44100 | 1.0 s | 1 |
| long_music | Synthetic harmonic sequence | 44100 | 10.0 s | 1 |
| music_1min | Synthetic harmonic sequence (1min) | 44100 | 60.0 s | 1 |
| music_5min | Synthetic harmonic sequence (5min) | 44100 | 300.0 s | 1 |

## Generation Interface

```python
from apple_aac.data.synthetic import generate_test_signals

signals = generate_test_signals()
# signals: dict[str, dict] where each entry has:
#   "pcm": np.ndarray (float32, shape=(num_samples,))
#   "sample_rate": int
#   "duration": float
#   "description": str
```

## Statistics
- Total audio: ~378.5 seconds (~6.3 minutes)
- Sample rate: 44100 Hz (all signals)
- Bit depth: float32 normalized
- Total samples: ~16,693,350

## Known Issues / Caveats
- Synthetic signals only; real-world audio testing deferred to v2
- All mono; stereo/multichannel deferred to v2
- 1-hour signal not included by default (604 MB memory); use `generate_test_signals(include_1hr=True)` for throughput scaling tests

## Version History
- **v1** (this version, 2026-06-03): initial synthetic test signal set (10 base + 2 extended duration)
