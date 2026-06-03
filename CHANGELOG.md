# Changelog

This project follows a research-adapted SemVer (see ai-research-code-workflow.md).
DATASET.md and BENCHMARK.md changes trigger MAJOR bumps.

## [Unreleased]

## [0.2.0] - 2026-06-03

### Added
- Metal compute shader for Huffman encoding (4 kernels: codeword computation,
  Blelloch prefix sum, atomic scatter-write, zero-init)
- Metal compute shader for Huffman decoding (one-thread-per-frame exp-Golomb decode)
- Metal compute shader for quantization (binary search with parallel bit-count
  reduction in threadgroup shared memory)
- Metal compute shader for scalefactor computation from masking thresholds
- Objective-C host code with `CFBridgingRetain`/`CFBridgingRelease` for C struct compatibility
- Python ctypes bridge (`metal_bridge.py`) with singleton `MetalHuffman` context
- MLX-accelerated framing via GPU gather indexing (3x faster than numpy loop)
- Cross-encoder comparison script (`compare_all_encoders.py`): MetalAAC vs
  Apple afconvert vs ffmpeg native aac vs ffmpeg aac_at vs PyAV
- Extended test signals: 1-minute and 5-minute durations, optional 1-hour
- Full 50-run benchmark with std for all durations up to 3600s

### Changed
- Quantization binary search reduced from 20 to 8 iterations (mathematically
  sufficient for [0, 255] gain range; 2.4x speedup, <0.2% bit-count difference)
- GPU encoder path now uses Metal for quantization + Huffman instead of
  MLX + multiprocess CPU
- Package renamed from `apple_aac` to `metal_aac`

### Performance
- **5.2x faster than Apple afconvert** at 300s (102ms vs 529ms encode-only)
- **18x faster than ffmpeg native aac** at 300s (102ms vs 1862ms)
- **439x faster than v0.1.0 CPU baseline** (44.8s vs 102ms for 300s)
- Metal Huffman: 70-886x speedup over Python bit-packing
- Metal quantization: 23x speedup over MLX (eliminates `mx.eval()` overhead)
- Pipeline balanced: no single stage exceeds 30% of total time
- Results: BENCHMARK.md, "End-to-End Encoder: Full Optimization History"

## [0.1.0] - 2026-06-03

### Added
- Project skeleton with DATASET.md and BENCHMARK.md contracts
- Synthetic test signal generation (9 signal types)
- MDCT/IMDCT with both CPU (NumPy) and GPU (MLX) backends
- Simplified psychoacoustic model (power spectrum, Bark-scale critical bands, spreading function)
- AAC quantization formula with rate control loop
- Huffman-style entropy coding (exp-Golomb, simplified codebooks)
- Bitstream writer/reader for custom frame format
- End-to-end encoder and decoder pipelines
- Benchmark framework comparing CPU vs GPU per stage
- Quality metrics: SNR, Spectral Convergence
- Throughput metrics: RTF, frame throughput, GPU speedup
- Contract tests for DATASET.md and BENCHMARK.md
- Round-trip correctness tests (44 tests)
