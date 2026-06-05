# MetalAAC

GPU-accelerated AAC-LC encoder/decoder for Apple Silicon, using MLX + Metal compute shaders + native C.

Produces ISO/IEC 14496-3 compliant ADTS output decodable by ffmpeg/VLC. Supports mono and stereo with M/S coding.

## Current State (v0.10.3)

| Metric | Value |
|--------|-------|
| Encode 60s mono | 505 ms (RTF 0.008) |
| Decode 60s mono (GPU) | 74 ms (RTF 0.001) |
| SNR (440 Hz sine, ffmpeg roundtrip) | 51.6 dB |
| Bitrate accuracy | 125-136 kbps at 128 target |
| Stereo | M/S coding, 51.7 dB per channel |
| Tests | 82 passing |

## Setup

Requires macOS with Apple Silicon (M1/M2/M3/M4) and Python 3.11+.

```bash
git clone git@github.com:feiyuehchen/MetalAAC.git
cd MetalAAC

# Build native library (Metal shaders + C Huffman decoder)
cd metal && make && cd ..

# Install Python package
python -m venv .venv && source .venv/bin/activate
pip install mlx numpy pyyaml pytest
pip install -e ".[dev]"

# Verify
python -m pytest tests/
```

## Usage

### Encode

```python
import numpy as np
from metal_aac.encoder import encode, EncoderConfig

# Mono
pcm = np.random.randn(44100 * 5).astype(np.float32) * 0.5
result = encode(pcm, EncoderConfig(
    sample_rate=44100,
    target_bitrate_kbps=128.0,
    output_format="adts",
))
with open("output.aac", "wb") as f:
    f.write(result.bitstream)

# Stereo (M/S coding automatic)
stereo = np.column_stack([left_channel, right_channel])
result = encode(stereo, EncoderConfig(output_format="adts"))
```

### Decode

```python
from metal_aac.decoder import decode, DecoderConfig

result = decode(bitstream, DecoderConfig(use_gpu=True))
pcm = result.pcm          # (N,) mono or (N, 2) stereo
sr = result.sample_rate    # 44100
```

### Benchmark

```bash
python scripts/benchmark.py
python scripts/compare_all_encoders.py --durations 10 60 300
```

## Architecture

```
Encoder:
  PCM -> Framing -> MDCT -> Psychoacoustic -> M/S Stereo -> Quantization -> Huffman -> ADTS
         MLX GPU   MLX GPU  MLX GPU          MLX GPU       MLX GPU         Metal GPU

Decoder:
  ADTS -> Huffman parse -> Dequantize -> Inverse M/S -> IMDCT -> Overlap-Add -> PCM
          Native C+GCD    NumPy/MLX    NumPy          MLX GPU   NumPy
```

### Encoder stages (60s mono)

| Stage | Time | Accelerator |
|-------|------|-------------|
| Transient detect | 166 ms | MLX GPU |
| MDCT | 41 ms | MLX GPU (batch matmul) |
| Psychoacoustic | 8 ms | MLX GPU (FFT + matmul) |
| Quantization | 309 ms | MLX GPU (binary search, 2-pass) |
| Huffman + ADTS | 55 ms | Metal GPU |
| **Total** | **583 ms** | |

### Decoder stages (60s mono)

| Stage | Time | Accelerator |
|-------|------|-------------|
| Huffman parse | ~0 ms | Native C + GCD (LUT decode) |
| Dequantize | 45 ms | NumPy vectorized |
| IMDCT | 22 ms | MLX GPU |
| Overlap-add | 4 ms | NumPy |
| **Total** | **74 ms** (GPU) | |

## Project structure

```
MetalAAC/
├── metal/                          # Native code (ObjC + Metal shaders)
│   ├── huffman_kernels.metal       # 8 Metal compute kernels
│   ├── metal_huffman.h/.m          # ObjC host + C ISO decoder (GCD parallel)
│   └── Makefile
│
├── src/metal_aac/
│   ├── core/
│   │   ├── mdct.py                 # MDCT/IMDCT (CPU + MLX GPU)
│   │   ├── psychoacoustic.py       # Bark-scale masking (CPU + GPU)
│   │   ├── quantization.py         # ISO quantizer (clipping-aware per-band SF)
│   │   ├── raw_data_block.py       # ISO bitstream encoder/decoder (SCE + CPE)
│   │   ├── adts.py                 # ADTS header writer/reader
│   │   ├── metal_bridge.py         # ctypes wrapper for native library
│   │   └── window_switching.py     # Transient detection + state machine
│   ├── encoder.py                  # Encode pipeline (mono + stereo)
│   ├── decoder.py                  # Decode pipeline (mono + stereo)
│   └── tables/                     # Huffman codebooks, SFB tables, windows
│
├── DATASET.md                      # Test signal specification
├── BENCHMARK.md                    # Metric definitions + results
├── CHANGELOG.md                    # v0.1.0 -- v0.10.3
└── tests/                          # 82 tests
```

## Version history

| Version | Key change |
|---------|-----------|
| v0.1.0 | Python baseline |
| v0.2.0 | Metal GPU pipeline (5.2x faster than Apple, legacy format) |
| v0.3.0 | ISO Huffman + ADTS, ffmpeg decodable |
| v0.4.0 | ISO SF calibration (SNR 13 dB) |
| v0.5.0 | MDCT 2/N normalization (SNR 25 dB) |
| v0.6.0 | ISO-native quantizer (SNR 47 dB) |
| v0.6.1 | Metal ADTS kernel uses ISO SFs directly |
| v0.7.0 | Clipping-aware rate allocation (10 -> 107 kbps) |
| v0.8.0 | ADTS decoder (no ffmpeg dependency) |
| v0.8.1 | Adaptive bitrate calibration |
| v0.9.0 | Stereo CPE encoding/decoding |
| v0.10.0 | M/S stereo + bit reservoir |
| v0.10.1 | Code review: 9 bug fixes |
| v0.10.2 | LUT Huffman + vectorized dequant |
| v0.10.3 | Native C+GCD decoder (23x faster) |

## Research workflow

This project follows the [AI Research Code Workflow](../ai-research-code-workflow.md):
- **DATASET.md** and **BENCHMARK.md** are the research contracts
- Version bumps follow research-adapted SemVer
- Changes to metric definitions trigger MAJOR bumps
- All benchmark results accumulate in BENCHMARK.md Part B

## License

MIT
