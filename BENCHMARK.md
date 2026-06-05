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

## Cross-Encoder Comparison (encode-only, 128 kbps mono)

| Duration | MetalAAC | Apple afconvert | ffmpeg aac | vs Apple |
|----------|----------|-----------------|------------|----------|
| 10s | **36 ms** | 44 ms | 138 ms | 1.2x |
| 60s | **46 ms** | 157 ms | 390 ms | **3.4x** |
| 300s | **169 ms** | 707 ms | 1764 ms | **4.2x** |

## Cross-Decoder Comparison (60s mono ADTS)

| Decoder | Time | vs ffmpeg |
|---------|------|----------|
| MetalAAC GPU | **56 ms** | **2.1x faster** |
| MetalAAC CPU | 59 ms | 2.0x faster |
| ffmpeg | 117 ms | baseline |

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
| Framing | 3 | MLX GPU |
| Transient detect | 4 | MLX GPU |
| MDCT | 27 | MLX GPU (matmul) |
| Psychoacoustic | 3 | MLX GPU (FFT) |
| Quantization | 9 | Metal GPU (binary search) |
| Huffman + ADTS | 15 | Metal GPU |
| **Total** | **61** | |

## Decoder Stage Breakdown (60s mono ADTS)

| Stage | Time (ms) | Accelerator |
|-------|-----------|-------------|
| Huffman parse + overhead | 24 | Native C + GCD |
| Dequantize | 9 | MLX GPU |
| IMDCT | 19 | MLX GPU (matmul) |
| Overlap-add | 7 | NumPy |
| **Total** | **56** (GPU) | |

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
