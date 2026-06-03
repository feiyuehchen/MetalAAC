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

# Part B -- Results (accumulating; updated every run)

## v0.1.x Results (current metric definitions)

### Per-Stage GPU Speedup (Apple M3 Pro, MLX 0.31.2, 50 runs, mean ± std)

| Duration | Frames | MDCT CPU (ms) | MDCT GPU (ms) | MDCT Speedup | Psycho CPU (ms) | Psycho GPU (ms) | Psycho Speedup | IMDCT CPU (ms) | IMDCT GPU (ms) | IMDCT Speedup |
|----------|--------|---------------|---------------|--------------|-----------------|-----------------|----------------|----------------|----------------|---------------|
| 10s | 431 | 1.24 ± 0.09 | 1.04 ± 0.52 | 1.19x | 3.42 ± 0.36 | 0.73 ± 0.28 | 4.71x | 1.47 ± 0.09 | 0.81 ± 0.44 | 1.82x |
| 60s | 2,584 | 8.21 ± 0.21 | 2.57 ± 0.72 | 3.20x | 22.38 ± 1.22 | 1.92 ± 0.47 | 11.68x | 8.68 ± 1.90 | 2.55 ± 0.52 | 3.40x |
| 300s | 12,920 | 38.52 ± 1.60 | 10.66 ± 0.20 | 3.61x | 121.09 ± 4.41 | 6.88 ± 0.53 | 17.60x | 40.89 ± 2.66 | 10.78 ± 0.62 | 3.79x |
| **3600s** | **155,040** | **491.16 ± 87.45** | **129.15 ± 22.04** | **3.80x** | **1899.46 ± 444.97** | **82.49 ± 2.60** | **23.03x** | **508.44 ± 72.12** | **123.87 ± 0.40** | **4.10x** |

### Apple AAC (afconvert) Throughput (50 runs, mean ± std)

| Duration | Apple Encode (ms) | Apple RTF |
|----------|-------------------|-----------|
| 10s | 38.6 ± 2.2 | 0.003865 |
| 60s | 123.8 ± 2.9 | 0.002064 |
| 300s | 527.7 ± 5.4 | 0.001759 |
| 3600s | 6048.2 ± 61.3 | 0.001680 |

### End-to-End Quality (128 kbps target, CPU path)

| Signal | SNR (dB) | SC | Actual Bitrate (kbps) | Encoder RTF | Decoder RTF |
|--------|----------|----|----------------------|-------------|-------------|
| sine_440 (1s) | 19.9 | 0.0178 | 77.3 | 0.206 | 0.035 |
| chirp (2s) | 24.0 | 0.0057 | 79.0 | 0.195 | 0.035 |
| long_music (10s) | 25.2 | 0.0081 | 87.5 | 0.182 | 0.035 |
| music_1min (60s) | 38.5 | 0.0037 | 91.5 | 0.150 | 0.035 |
| music_5min (300s) | 44.0 | 0.0033 | 91.5 | 0.152 | 0.035 |

### End-to-End Encoder: Full Optimization History

| Duration | v1 CPU | v2 MLX GPU | v3 +Metal Huffman | v4 +Metal Quant | v5 +MLX Framing | Apple AAC | v5/Apple |
|----------|--------|-----------|-------------------|-----------------|-----------------|-----------|----------|
| 10s | 1.502s | 0.268s | 0.069s | 0.023s | **0.023s** | 0.042s | **1.9x faster** |
| 60s | 8.869s | 0.512s | 0.176s | 0.038s | **0.033s** | 0.128s | **3.9x faster** |
| 300s | 44.789s | 1.754s | 0.699s | 0.121s | **0.102s** | 0.529s | **5.2x faster** |

### Pipeline Breakdown (v5 final, 300s, 12920 frames)

| Stage | Time (ms) | % | Accelerator |
|-------|-----------|---|-------------|
| Framing | 7.1 | 7.0% | MLX GPU (gather indexing) |
| MDCT | 29.1 | 28.5% | MLX GPU (batch matmul) |
| Psychoacoustic | 8.0 | 7.8% | MLX GPU (batch FFT + matmul) |
| Quantization | 21.4 | 21.0% | Metal GPU (binary search kernel) |
| Huffman + bitstream | 30.1 | 29.5% | Metal GPU (4 compute kernels) |
| **Total** | **102.0** | | |

### Optimization Progression on Quantization Stage (300s)

| Version | Method | Time (ms) | Bottleneck? |
|---------|--------|-----------|-------------|
| v2 | CPU Python per-frame loop | ~44,000 | Yes (99%) |
| v3 | MLX batch GPU, 20 iterations | 518 | Yes (79%) |
| v4a | MLX batch GPU, 8 iterations | 215 | Yes (63%) — 20 iter was overkill for [0,255] range |
| v4b | Metal kernel, 8 iterations | **22** | No (21%) — 0 host round-trips, 1 GPU dispatch |

### Optimization Progression on Huffman Stage (300s)

| Version | Method | Time (ms) | Bottleneck? |
|---------|--------|-----------|-------------|
| v1 | CPU Python per-frame sequential | ~44,000 | Yes (shared w/ quant) |
| v2 | CPU multiprocess (12 cores) | 1,100 | Yes (86%) |
| v3 | Metal GPU (codeword + prefix sum + atomic scatter) | **30** | No (29%) |

### Key Observations

1. **5.2x faster than Apple AAC at 300s** — Python + MLX + Metal beats Apple's optimized C encoder
2. **Pipeline is balanced** — no single stage exceeds 30% of total time
3. **Total speedup from v1 to v5: 439x** (44.8s → 0.102s for 300s)
4. **Metal eliminated two bottlenecks**: quantization (MLX eval overhead) and Huffman (Python bit-packing)
5. **MLX matmul MDCT outperforms FFT MDCT** at this batch size — Apple's matmul kernel is highly optimized, and FFT path has more kernel launches
6. **MLX framing (gather) 3x faster than numpy loop** — GPU-native indexing avoids CPU memory allocation

### Per-Stage GPU Speedup (50 runs, mean ± std)

| Duration | Frames | MDCT CPU (ms) | MDCT GPU (ms) | MDCT Speedup | Psycho CPU (ms) | Psycho GPU (ms) | Psycho Speedup |
|----------|--------|---------------|---------------|--------------|-----------------|-----------------|----------------|
| 10s | 431 | 1.24 ± 0.09 | 1.04 ± 0.52 | 1.19x | 3.42 ± 0.36 | 0.73 ± 0.28 | 4.71x |
| 60s | 2,584 | 8.21 ± 0.21 | 2.57 ± 0.72 | 3.20x | 22.38 ± 1.22 | 1.92 ± 0.47 | 11.68x |
| 300s | 12,920 | 38.52 ± 1.60 | 10.66 ± 0.20 | 3.61x | 121.09 ± 4.41 | 6.88 ± 0.53 | 17.60x |
| 3600s | 155,040 | 491.16 ± 87.45 | 129.15 ± 22.04 | 3.80x | 1899.46 ± 444.97 | 82.49 ± 2.60 | 23.03x |

### Cross-Encoder Comparison (encode-only, 128 kbps, 10 runs)

All times are encode-only. MetalAAC encode-only times from standalone benchmark.

| Encoder | 10s (ms) | 60s (ms) | 300s (ms) | 300s RTF | SNR (dB) | SC |
|---------|----------|----------|-----------|----------|----------|-----|
| **MetalAAC (ours)** | **22.5** | **32.7** | **102.0** | **0.000340** | 42.1 | 0.0034 |
| Apple afconvert | 38.4 | 123.5 | 525.9 | 0.001753 | 30.0 | 0.0232 |
| ffmpeg (native aac) | 143.6 | 438.4 | 1862.0 | 0.006207 | 49.0 | 0.0029 |
| ffmpeg (aac_at) | 151.7 | 410.4 | 1642.2 | 0.005474 | 57.9 | 0.0008 |

Notes:
- **MetalAAC is 5.2x faster than Apple afconvert, 18x faster than ffmpeg** at 300s
- ffmpeg aac_at uses Apple AudioToolbox under the hood (same engine as afconvert) but has more subprocess overhead
- ffmpeg native aac has higher SNR than afconvert because it's a different encoder implementation
- Quality comparison is approximate — different encoders use different psychoacoustic models
- PyAV omitted from table: its AAC decode produced corrupted output (SNR=-0.0), likely a framing bug in the benchmark wrapper

### Apple AAC (afconvert) Throughput (50 runs, mean ± std)

| Duration | Apple Encode (ms) | Apple RTF |
|----------|-------------------|-----------|
| 10s | 38.6 ± 2.2 | 0.003865 |
| 60s | 123.8 ± 2.9 | 0.002064 |
| 300s | 527.7 ± 5.4 | 0.001759 |
| 3600s | 6048.2 ± 61.3 | 0.001680 |

---

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
