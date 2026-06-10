# Benchmark Specification

---

# Part A -- Definitions (changes require MAJOR bump)

## Evaluation Metrics

### Real-Time Factor (RTF)
- **Formula**: `RTF = processing_time / audio_duration`
- **Implementation**: `src/apple_aac/metrics/throughput.py::compute_rtf()`
- **Reduction**: Mean across all test signals
- **Notes**: RTF < 1.0 means faster than real-time. Measured per-stage and end-to-end.

```python
def compute_rtf(
    processing_time_sec: float,
    audio_duration_sec: float,
) -> float
```

### Signal-to-Noise Ratio (SNR)
- **Formula**: `SNR = 10 * log10(sum(x^2) / sum((x - x_hat)^2))` in dB
- **Implementation**: `src/apple_aac/metrics/quality.py::compute_snr()`
- **Reduction**: Per-signal, then mean across test set
- **Notes**: Measured on round-trip (encode -> decode). Higher is better.

```python
def compute_snr(
    original: np.ndarray,   # (T,) float32
    reconstructed: np.ndarray,  # (T,) float32
) -> float
```

### Spectral Convergence (SC)
- **Formula**: `SC = ||STFT(x) - STFT(x_hat)||_F / ||STFT(x)||_F`
- **Implementation**: `src/apple_aac/metrics/quality.py::compute_spectral_convergence()`
- **Reduction**: Per-signal, then mean across test set
- **Notes**: Lower is better. Uses 2048-point STFT with hop=512, Hann window.

```python
def compute_spectral_convergence(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> float
```

### Frame Throughput
- **Formula**: `throughput = num_frames / processing_time` in frames/sec
- **Implementation**: `src/apple_aac/metrics/throughput.py::compute_frame_throughput()`
- **Reduction**: Mean across 10 repeated measurements (excluding first as warmup)
- **Notes**: Measured per-stage (MDCT, psychoacoustic, quantization, Huffman, total).

```python
def compute_frame_throughput(
    num_frames: int,
    processing_time_sec: float,
) -> float
```

### GPU Speedup
- **Formula**: `speedup = cpu_time / gpu_time`
- **Implementation**: `src/apple_aac/metrics/throughput.py::compute_gpu_speedup()`
- **Reduction**: Per-stage, per-signal, then mean
- **Notes**: Speedup > 1.0 means GPU is faster. Measured for parallelizable stages only.

```python
def compute_gpu_speedup(
    cpu_time_sec: float,
    gpu_time_sec: float,
) -> float
```

### Peak Memory Usage
- **Formula**: Peak resident memory during encoding (MB)
- **Implementation**: `src/apple_aac/metrics/throughput.py::measure_peak_memory()`
- **Reduction**: Max across test signals
- **Notes**: Measured separately for CPU and GPU paths.

```python
def measure_peak_memory(func: Callable, *args) -> tuple[float, Any]
# Returns (peak_mb, result)
```

## Evaluation Protocol
- Warmup: 1 run discarded before timing
- Timing: 50 runs per configuration, report mean +/- std
- Quality metrics: single deterministic run (no randomness in encode/decode)
- Default target bitrate: 128 kbps
- Default sample rate: 44100 Hz
- Platform: Apple Silicon (M-series), report specific chip model

## Significance Testing
- For throughput: paired t-test CPU vs GPU, p < 0.05
- For quality: not applicable (deterministic codec, no randomness)

---

# Part B — Results (v0.12.0, M3 Pro, best of 5 runs)

## Cross-Encoder/Decoder Comparison (128 kbps mono, M3 Pro, best of 5)

### Encode

| Encoder | 10s | 60s | 300s | Technology |
|---------|-----|-----|------|------------|
| **MetalAAC** | **28 ms** | **48 ms** | **167 ms** | Metal GPU compute shaders |
| Apple afconvert | 45 ms | 162 ms | 699 ms | AudioToolbox (Apple native CPU) |
| ffmpeg aac_at | 135 ms | 315 ms | 1161 ms | AudioToolbox via ffmpeg |
| ffmpeg aac | 136 ms | 390 ms | 1755 ms | ffmpeg native (CPU software) |

MetalAAC vs Apple afconvert: **1.6x** (10s), **3.3x** (60s), **4.2x** (300s) faster.
MetalAAC vs ffmpeg aac_at (AudioToolbox): **4.8x** (10s), **6.5x** (60s), **7.0x** (300s) faster.

Note: AudioToolbox is NOT hardware-accelerated — AAC has no dedicated hardware
decoder on Mac (unlike H.264/HEVC which use VideoToolbox silicon). AudioToolbox
is Apple's highly optimized C/NEON SIMD software implementation running on CPU.
MetalAAC uses GPU parallel binary search (1024 threads per frame) to outperform it.

### Decode

| Decoder | 10s | 60s | 300s | Technology |
|---------|-----|-----|------|------------|
| **MetalAAC** | **22 ms** | **42 ms** | **131 ms** | Native C (GCD parallel) + MLX GPU |
| ffmpeg aac | 82 ms | 120 ms | 277 ms | ffmpeg native (CPU software) |
| ffmpeg aac_at | 97 ms | 137 ms | 321 ms | AudioToolbox via ffmpeg |

MetalAAC vs ffmpeg aac: **3.7x** (10s), **2.8x** (60s), **2.1x** (300s) faster.
MetalAAC vs ffmpeg aac_at (AudioToolbox): **4.4x** (10s), **3.2x** (60s), **2.5x** (300s) faster.

Note: ffmpeg aac_at (AudioToolbox) is slower than ffmpeg native for decoding,
likely due to ffmpeg↔AudioToolbox bridging overhead.

## Quality (ADTS, ffmpeg decode, 128 kbps target)

| Signal | SNR (dB) | Bitrate (kbps) |
|--------|----------|----------------|
| sine_440 | 53.4 | 107 |
| sine_1k | 52.6 | 106 |
| sine_4k | 51.0 | 115 |
| multitone | 51.4 | 112 |
| chirp | 51.8 | 109 |
| white_noise | 10.2 | 134 |
| pink_noise | 16.2 | 135 |
| music_1min | 46.5 | 111 |

## Stereo (M/S coding, 128 kbps target)

| Signal | SNR L (dB) | SNR R (dB) | Bitrate (kbps) |
|--------|-----------|-----------|----------------|
| Correlated (L=R) | 51.5 | 51.5 | 52 |
| Uncorrelated (440+1kHz) | 51.4 | 50.5 | 100 |

## Encoder Stage Breakdown (60s mono ADTS)

| Stage | Time (ms) | Accelerator |
|-------|-----------|-------------|
| Framing | 2 | MLX GPU |
| Transient detect | 2 | MLX GPU |
| MDCT | 17 | MLX GPU (matmul) |
| Psychoacoustic | 3 | MLX GPU (FFT) |
| Quantization | 8 | Metal GPU (binary search) |
| Huffman + ADTS | 17 | Metal GPU |
| **Total** | **48** | |

## Decoder Stage Breakdown (60s mono ADTS)

| Stage | Time (ms) | Accelerator |
|-------|-----------|-------------|
| ADTS header parse | 21 | Python (sync word scan) |
| Huffman decode | 4 | Native C + GCD (LUT) |
| Dequantize | 7 | MLX GPU |
| IMDCT | 24 | MLX GPU (matmul) |
| Overlap-add | 5 | NumPy |
| **Total** | **56** | |

Note: previous reports listed Huffman decode as "~0ms" because the ADTS
header parsing (21ms, Python) was measured separately. The actual C+GCD
Huffman decode is 4ms; the 21ms Python ADTS parse is the decode bottleneck.

## CPU Baseline vs Metal — Per-Stage Speedup (60s mono)

### Encode

| Stage | CPU Baseline | Metal v0.13.1 | Speedup |
|-------|-------------|---------------|---------|
| Framing | 3 ms | 2 ms | — |
| MDCT | 21 ms | 17 ms | 1.2x |
| Psychoacoustic | 51 ms | 3 ms | **17x** |
| Quantization | 8050 ms | 8 ms | **1000x** |
| Huffman + ADTS | 236 ms | 17 ms | **14x** |
| **Total** | **8362 ms** | **48 ms** | **174x** |

### Decode

| Stage | CPU Baseline | Metal v0.13.1 | Speedup |
|-------|-------------|---------------|---------|
| ADTS + Huffman parse | 71 ms | 25 ms (21 Python + 4 C) | **3x** |
| Dequantize | 26 ms | 7 ms | **4x** |
| IMDCT | 23 ms | 24 ms | — |
| Overlap-add | 4 ms | 5 ms | — |
| **Total** | **125 ms** | **56 ms** | **2x** |

## Encode Optimization History

| Version | 60s encode | vs Apple | Key change |
|---------|-----------|----------|------------|
| v0.1.0 | ~9000 ms | 57x slower | Python baseline |
| v0.2.0 | 33 ms | 5x faster | Metal GPU pipeline (legacy format) |
| v0.7.0 | 314 ms | 2x slower | ISO ADTS + clipping-aware rate allocation |
| v0.11.0 | 320 ms | 2x slower | Single-pass quantization |
| **v0.12.0** | **46 ms** | **3.4x faster** | Metal ISO quantizer |

## Decode Optimization History

| Version | 60s decode | Method |
|---------|-----------|--------|
| v0.10.1 | 2386 ms | Python bit-by-bit tree |
| v0.10.2 | 1391 ms | Python LUT (1.7x) |
| v0.10.3 | 102 ms | Native C + GCD (23x) |
| **v0.10.4** | **56 ms** | + MLX dequant/IMDCT (42x) |

## Abandoned Directions

### GPU precomputed codewords + multiprocess packing
- Precomputing codewords on GPU then sending to ProcessPoolExecutor workers was **slower** than
  having each worker compute its own codewords, due to IPC serialization overhead of the large
  precomputed arrays (2 × B × N × 8 bytes).

### FFT-based MDCT
- O(N log N) vs O(N²) theoretically better, but MLX matmul is faster at N=1024 because
  Apple's Metal matmul kernel is highly optimized (1 dispatch) while FFT path needs 5+
  kernel launches (gather, complex pack, FFT, twiddle, real extract).

### Reducing mx.eval() calls in quantization loop
- Deferring all evals to the end of 8 iterations gave identical performance (271ms vs 265ms)
  because the computation has loop-carried dependencies that MLX cannot parallelize regardless
  of eval placement. The real fix was moving to Metal (215ms → 22ms).
