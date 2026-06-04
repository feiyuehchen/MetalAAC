# MetalAAC

GPU-accelerated AAC-LC encoder for Apple Silicon, using MLX and Metal compute shaders.

**Produces ISO/IEC 14496-3 compliant ADTS output decodable by ffmpeg/VLC.** 5.2x faster than Apple's `afconvert` and 18x faster than ffmpeg at encoding on M3 Pro.

## Benchmark (300s audio, 128 kbps, encode-only)

| Encoder | Time | vs MetalAAC |
|---------|------|-------------|
| **MetalAAC** | **102 ms** | — |
| Apple afconvert | 526 ms | 5.2x slower |
| ffmpeg (aac_at) | 1,642 ms | 16x slower |
| ffmpeg (native aac) | 1,862 ms | 18x slower |

All numbers on Apple M3 Pro, best of 5 runs. Full results with 50-run std in [BENCHMARK.md](BENCHMARK.md).

### Quality (encode → ffmpeg decode → compare)

| Signal | SNR | Amplitude ratio |
|--------|-----|-----------------|
| 440 Hz sine | 13.3 dB | 0.99x |
| 1 kHz sine | 13.1 dB | 0.95x |
| Multi-tone | 7.0 dB | 0.90x |

## How it works

```
                    ┌── Transient detect (MLX GPU) ──→ window decision
PCM ──┬─────────────┤
      │             └── Masking thresholds (MLX GPU FFT+matmul)
      │                                        │
      └──→ Framing ──→ MDCT ──────────────────┘──→ Quantize ──→ Huffman ──→ ADTS
           MLX GPU     MLX GPU                     Metal GPU    Metal GPU
           (gather)    (matmul)                     (binary      (ISO codebook
                                                    search +     LUT + atomic
                                                    max|q|       scatter-write)
                                                    reduction)
```

The psychoacoustic model runs **parallel** from raw PCM (not serial after MDCT), matching the standard AAC encoder architecture (Brandenburg, AES-17).

**Key insight:** MLX is ideal for array-level operations (MDCT, psychoacoustic model), but stages requiring per-thread control flow (Huffman bit packing, quantization binary search) need Metal compute shaders. Moving both to Metal eliminated the Python-to-GPU round-trip overhead that dominated the pipeline.

## Setup

Requires macOS with Apple Silicon (M1/M2/M3/M4) and Python 3.11+.

```bash
# Clone
git clone git@github.com:feiyuehchen/MetalAAC.git
cd MetalAAC

# Build Metal shaders
cd metal && make && cd ..

# Install Python package
python -m venv .venv
source .venv/bin/activate
pip install mlx numpy pyyaml pytest
pip install -e ".[dev]"

# Verify
python -m pytest tests/
```

## Usage

### Encode / Decode

```python
from metal_aac.encoder import EncoderConfig, encode
from metal_aac.decoder import DecoderConfig, decode

# Encode to ISO-compliant ADTS (decodable by ffmpeg/VLC)
result = encode(pcm_float32, EncoderConfig(
    sample_rate=44100,
    target_bitrate_kbps=128.0,
    use_gpu=True,
    output_format="adts",  # ISO AAC-LC ADTS output
))

# Write to .aac file
with open("output.aac", "wb") as f:
    f.write(result.bitstream)

# Verify with ffmpeg:
#   ffmpeg -i output.aac -f f32le decoded.raw

# Internal round-trip (legacy format, for testing)
result = encode(pcm_float32, EncoderConfig(use_gpu=True))
decoded = decode(result.bitstream, DecoderConfig(
    use_gpu=True, output_length=len(pcm_float32),
))
reconstructed = decoded.pcm
```

### Benchmark

```bash
# Per-stage CPU vs GPU comparison
python scripts/benchmark.py --signals sine_440 chirp long_music

# Full 50-run benchmark across durations
python scripts/benchmark_full.py --n-runs 50

# Compare against Apple AAC and ffmpeg
python scripts/compare_all_encoders.py --durations 10 60 300
```

### Metal Huffman (direct API)

```python
from metal_aac.core.metal_bridge import MetalHuffman

metal = MetalHuffman.shared()

# Encode: quantized (B, 1024) int32 → list of packed byte arrays
frames = metal.encode_frames(quantized, scalefactors, global_gains)

# Decode: list of byte arrays → quantized (B, 1024) int32
quantized, scalefactors, gains = metal.decode_frames(frame_payloads)

# Full quantization on GPU (binary search + bit estimation)
q, sf, gg, bits = metal.quantize(
    mdct_coeffs, masking_thresholds,
    target_bits_per_frame=2972,
)
```

## Project structure

```
MetalAAC/
├── DATASET.md              # Test signal specification (research contract)
├── BENCHMARK.md            # Metric definitions + all results (research contract)
├── CHANGELOG.md            # Version history
│
├── metal/                  # GPU compute shaders (Objective-C + MSL)
│   ├── huffman_kernels.metal   # 7 Metal compute kernels
│   ├── metal_huffman.h         # C API
│   ├── metal_huffman.m         # ObjC host code
│   └── Makefile
│
├── src/metal_aac/          # Python package
│   ├── core/
│   │   ├── mdct.py             # MDCT/IMDCT (CPU NumPy + GPU MLX)
│   │   ├── psychoacoustic.py   # Psychoacoustic model (CPU + GPU)
│   │   ├── quantization.py     # Quantization (CPU + MLX + Metal)
│   │   ├── huffman.py          # Entropy coding (CPU + Metal)
│   │   ├── bitstream.py        # Legacy frame format reader/writer
│   │   ├── adts.py             # ISO ADTS header writer/reader
│   │   ├── raw_data_block.py   # ISO raw_data_block encoder (SCE)
│   │   ├── window_switching.py # Transient detection + state machine
│   │   └── metal_bridge.py     # ctypes wrapper for Metal dylib
│   ├── encoder.py              # Encoder pipeline (parallel psychoacoustic)
│   ├── decoder.py              # Decoder pipeline
│   ├── data/synthetic.py       # Deterministic test signal generation
│   ├── metrics/                # SNR, spectral convergence, RTF
│   └── tables/                 # SFB tables, window functions, Huffman codebooks
│
├── scripts/                # Benchmarking and comparison tools
└── tests/                  # 82 tests (contract + correctness + performance)
```

## Metal kernels

| Kernel | Grid | Purpose |
|--------|------|---------|
| `kernel_quantize` | B groups × 1024 threads | Binary search + parallel max\|q\| reduction |
| `kernel_compute_scalefactors` | B groups × 49 threads | SMR → scalefactor mapping |
| `kernel_encode_raw_data_block` | B groups × 1024 threads | Full ISO SCE: sections + SF DPCM + spectral (11 codebook LUTs) |
| `kernel_compute_codewords` | (N, B) | Exp-Golomb codeword + length (legacy path) |
| `kernel_prefix_sum` | B groups × 1024 threads | Blelloch exclusive scan (legacy path) |
| `kernel_scatter_write` | B groups × 1024 threads | Atomic OR bit packing (legacy path) |
| `kernel_decode_frames` | B threads | Sequential exp-Golomb decode per frame |
| `kernel_zero_output` | total bytes | Buffer initialization |

## Optimization history

| Version | 300s encode | vs Apple | Key change |
|---------|-------------|----------|------------|
| v0.1.0 CPU | 44,789 ms | 85x slower | Pure Python baseline |
| + MLX GPU | 1,754 ms | 3.3x slower | Batch MDCT + psychoacoustic on GPU |
| + Metal Huffman | 699 ms | 1.3x slower | 4-kernel exp-Golomb pipeline |
| + Metal quantize | 121 ms | 4.4x faster | Binary search in single dispatch |
| + MLX framing | 102 ms | 5.2x faster | GPU gather indexing |
| **v0.3.0 ISO ADTS** | **39 ms** | **13.5x faster** | Metal ISO codebook kernel |
| **v0.4.0 quality** | 39 ms | 13.5x faster | ISO SF calibration (SNR 13 dB) |

## Research workflow

This project follows the [AI Research Code Workflow](https://github.com/feiyuehchen/MetalAAC/blob/main/CHANGELOG.md):
- **DATASET.md** and **BENCHMARK.md** are the research contracts
- Version bumps follow research-adapted SemVer
- Changes to metric definitions or dataset spec trigger MAJOR bumps
- All benchmark results accumulate in BENCHMARK.md Part B

## License

MIT
