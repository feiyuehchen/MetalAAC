# MetalAAC

GPU-accelerated AAC encoder/decoder for Apple Silicon, using MLX and Metal compute shaders.

**5.2x faster than Apple's `afconvert`** and **18x faster than ffmpeg** at encoding 5 minutes of audio on M3 Pro.

## Benchmark (300s audio, 128 kbps, encode-only)

| Encoder | Time | vs MetalAAC |
|---------|------|-------------|
| **MetalAAC** | **102 ms** | — |
| Apple afconvert | 526 ms | 5.2x slower |
| ffmpeg (aac_at) | 1,642 ms | 16x slower |
| ffmpeg (native aac) | 1,862 ms | 18x slower |

All numbers on Apple M3 Pro, best of 5 runs. Full results with 50-run std in [BENCHMARK.md](BENCHMARK.md).

## How it works

Each AAC pipeline stage runs on the accelerator best suited for it:

```
PCM ─→ Framing ─→ MDCT ─→ Psychoacoustic ─→ Quantization ─→ Huffman ─→ Bitstream
       MLX GPU    MLX GPU   MLX GPU           Metal GPU       Metal GPU
       (gather)   (matmul)  (FFT+matmul)      (binary search  (prefix sum +
                                               + parallel      atomic scatter
                                               reduction)      write)
```

The key insight: MLX is ideal for array-level operations (MDCT, psychoacoustic model), but stages requiring per-thread control flow (Huffman bit packing, quantization binary search) need Metal compute shaders. Moving both to Metal eliminated the Python-to-GPU round-trip overhead that dominated the pipeline.

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

# Encode
result = encode(pcm_float32, EncoderConfig(
    sample_rate=44100,
    target_bitrate_kbps=128.0,
    use_gpu=True,  # MLX + Metal path
))
bitstream = result.bitstream
print(f"{result.num_frames} frames, {result.actual_bitrate_kbps:.1f} kbps")
print(f"Timings: {result.timings}")

# Decode
decoded = decode(bitstream, DecoderConfig(
    use_gpu=True,
    output_length=len(pcm_float32),
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
│   │   ├── bitstream.py        # Frame format reader/writer
│   │   └── metal_bridge.py     # ctypes wrapper for Metal dylib
│   ├── encoder.py              # Encoder pipeline
│   ├── decoder.py              # Decoder pipeline
│   ├── data/synthetic.py       # Deterministic test signal generation
│   ├── metrics/                # SNR, spectral convergence, RTF
│   └── tables/                 # SFB tables, window functions
│
├── scripts/                # Benchmarking and comparison tools
└── tests/                  # 44 tests (contract + correctness + round-trip)
```

## Metal kernels

| Kernel | Grid | Purpose |
|--------|------|---------|
| `kernel_compute_codewords` | (N, B) | Exp-Golomb codeword + length via `clz()` |
| `kernel_prefix_sum` | B groups × 1024 threads | Blelloch exclusive scan in shared memory |
| `kernel_scatter_write` | B groups × 1024 threads | Atomic OR on uint32 words for bit packing |
| `kernel_decode_frames` | B threads | Sequential exp-Golomb decode per frame |
| `kernel_quantize` | B groups × 1024 threads | Binary search with parallel reduction |
| `kernel_compute_scalefactors` | B groups × 49 threads | SMR → scalefactor mapping |
| `kernel_zero_output` | total bytes | Buffer initialization |

## Optimization history

| Version | 300s encode | vs Apple | Key change |
|---------|-------------|----------|------------|
| v0.1.0 CPU | 44,789 ms | 85x slower | Pure Python baseline |
| + MLX GPU | 1,754 ms | 3.3x slower | Batch MDCT + psychoacoustic on GPU |
| + Metal Huffman | 699 ms | 1.3x slower | 4-kernel Huffman pipeline |
| + Metal quantize | 121 ms | 4.4x faster | Binary search in single dispatch |
| **+ MLX framing** | **102 ms** | **5.2x faster** | GPU gather indexing |

## Research workflow

This project follows the [AI Research Code Workflow](https://github.com/feiyuehchen/MetalAAC/blob/main/CHANGELOG.md):
- **DATASET.md** and **BENCHMARK.md** are the research contracts
- Version bumps follow research-adapted SemVer
- Changes to metric definitions or dataset spec trigger MAJOR bumps
- All benchmark results accumulate in BENCHMARK.md Part B

## License

MIT
