# MetalAAC

GPU-accelerated AAC-LC encoder/decoder for Apple Silicon.
**3-4x faster than Apple's `afconvert`**, produces standard ADTS decodable by ffmpeg/VLC.

## Performance (M3 Pro, best of 5 runs)

### Encode Speed (mono, 128 kbps)

| Duration | MetalAAC | Apple afconvert | ffmpeg aac | vs Apple |
|----------|----------|-----------------|------------|----------|
| 10s | **36 ms** | 44 ms | 138 ms | 1.2x |
| 60s | **48 ms** | 158 ms | 390 ms | **3.3x** |
| 300s | **169 ms** | 707 ms | 1764 ms | **4.2x** |

### Decode Speed (60s mono)

| Decoder | Time | vs ffmpeg |
|---------|------|----------|
| MetalAAC GPU | **56 ms** | **2.1x faster** |
| ffmpeg | 117 ms | baseline |

### Quality (ADTS, ffmpeg decode)

| Signal | SNR | Bitrate |
|--------|-----|---------|
| 440 Hz sine | 53.4 dB | 107 kbps |
| 1 kHz sine | 52.6 dB | 106 kbps |
| 4 kHz sine | 51.0 dB | 115 kbps |
| Multi-tone | 51.4 dB | 112 kbps |
| Chirp | 51.8 dB | 109 kbps |
| White noise | 10.2 dB | 134 kbps |
| Pink noise | 16.2 dB | 135 kbps |
| Music (1 min) | 46.5 dB | 111 kbps |

### Stereo (M/S coding, 128 kbps)

| Signal | SNR L | SNR R | Bitrate |
|--------|-------|-------|---------|
| Correlated (L=R) | 51.5 dB | 51.5 dB | 52 kbps |
| Uncorrelated (440+1kHz) | 51.4 dB | 50.5 dB | 100 kbps |

## Quick Start

```bash
# Requires macOS + Apple Silicon + Python 3.11+
git clone git@github.com:feiyuehchen/MetalAAC.git && cd MetalAAC
cd metal && make && cd ..
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/  # 82 tests
```

## Usage

### Encode

```python
from metal_aac import encode, EncoderConfig

# Mono
result = encode(pcm_float32, EncoderConfig(output_format="adts"))
with open("output.aac", "wb") as f:
    f.write(result.bitstream)

# Stereo (M/S coding automatic)
result = encode(stereo_float32, EncoderConfig(output_format="adts"))

# Custom settings
result = encode(pcm, EncoderConfig(
    output_format="adts",
    target_bitrate_kbps=64.0,
    sample_rate=48000,
))
```

### Decode

```python
from metal_aac import decode

result = decode(open("output.aac", "rb").read())
pcm = result.pcm          # (N,) mono or (N, 2) stereo
sr = result.sample_rate    # auto-detected
```

### Quality Metrics

```python
from metal_aac.metrics.quality import compute_snr, compute_spectral_convergence

snr = compute_snr(original, decoded)                    # dB, higher=better
sc = compute_spectral_convergence(original, decoded)    # 0-1, lower=better
```

### Benchmark

```bash
python scripts/benchmark.py
python scripts/compare_all_encoders.py --durations 60
```

---

## API Reference

### `encode(pcm, config=None) → EncoderResult`

| Parameter | Type | Description |
|-----------|------|-------------|
| `pcm` | `np.ndarray` | `(N,)` float32 mono or `(N, 2)` stereo |
| `config` | `EncoderConfig` | Optional, defaults below |

### `EncoderConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `output_format` | `"legacy"` | `"adts"` for standard AAC output |
| `target_bitrate_kbps` | `128.0` | Target bitrate |
| `sample_rate` | `44100` | 8000–96000 Hz |
| `use_gpu` | `True` | Metal + MLX acceleration |
| `enable_window_switching` | `True` | Short-window MDCT for transients |

### `EncoderResult`

| Field | Type | Description |
|-------|------|-------------|
| `bitstream` | `bytes` | Encoded AAC data |
| `actual_bitrate_kbps` | `float` | Achieved bitrate |
| `num_frames` | `int` | Frame count |
| `audio_duration` | `float` | Seconds |
| `timings` | `dict` | Per-stage timing breakdown |

### `decode(bitstream, config=None) → DecoderResult`

| Parameter | Type | Description |
|-----------|------|-------------|
| `bitstream` | `bytes` | ADTS or legacy AAC (auto-detected) |
| `config` | `DecoderConfig` | Optional |

### `DecoderConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `use_gpu` | `True` | MLX for dequant + IMDCT |
| `output_length` | `None` | Truncate to N samples |

### `DecoderResult`

| Field | Type | Description |
|-------|------|-------------|
| `pcm` | `np.ndarray` | `(N,)` mono or `(N, 2)` stereo |
| `sample_rate` | `int` | From bitstream |
| `timings` | `dict` | Per-stage timing |

---

## Architecture

Backend is auto-detected at startup (`has_metal()`). Each encode/decode uses
one consistent path — no mixed Metal/MLX within a single call.

```
Metal path (primary, Apple Silicon + dylib):
  Encode: PCM → Frame(MLX) → MDCT(MLX) → Quantize(Metal) → Huffman(Metal) → ADTS
  Decode: ADTS → Huffman(C+GCD) → Dequant(MLX) → IMDCT(MLX) → PCM

MLX fallback (no dylib built):
  Encode: PCM → Frame(MLX) → MDCT(MLX) → Quantize(MLX) → Huffman(Python) → ADTS
  Decode: ADTS → Huffman(Python LUT) → Dequant(NumPy) → IMDCT(CPU) → PCM
```

### Pipeline Stages (60s mono, Metal path)

| Encode Stage | Time | Accelerator |
|-------------|------|-------------|
| Framing | 8 ms | MLX GPU |
| Transient detect | 6 ms | MLX GPU |
| MDCT | 33 ms | MLX GPU (matmul) |
| Psychoacoustic | 6 ms | MLX GPU (FFT) |
| Quantization | 21 ms | Metal GPU (binary search) |
| Huffman + ADTS | 18 ms | Metal GPU |

| Decode Stage | Time | Accelerator |
|-------------|------|-------------|
| Huffman parse | ~0 ms | Native C + GCD |
| Dequantize | 9 ms | MLX GPU |
| IMDCT | 19 ms | MLX GPU (matmul) |
| Overlap-add | 7 ms | NumPy |

## Project Structure

```
MetalAAC/
├── metal/                     # Metal shaders + native C
│   ├── huffman_kernels.metal  # GPU compute kernels
│   ├── metal_huffman.h/.m     # ObjC host + C decoder
│   └── Makefile
├── src/metal_aac/
│   ├── encoder.py             # encode() — mono + stereo
│   ├── decoder.py             # decode() — auto-detect
│   ├── core/
│   │   ├── quantization.py    # ISO quantizer (MLX fallback)
│   │   ├── raw_data_block.py  # ISO bitstream read/write
│   │   ├── adts.py            # ADTS container
│   │   ├── mdct.py            # MDCT/IMDCT
│   │   ├── psychoacoustic.py  # Bark-scale masking
│   │   └── metal_bridge.py    # ctypes bridge
│   ├── tables/                # Huffman codebooks, SFB tables
│   ├── metrics/               # SNR, spectral convergence
│   └── data/                  # Test signal generation
├── tests/                     # 82 tests
├── BENCHMARK.md               # Full results
└── CHANGELOG.md               # v0.1.0 → v0.12.0
```

## Version History

| Version | Encode 60s | vs Apple | Key Change |
|---------|-----------|----------|------------|
| v0.1.0 | ~9000 ms | 57x slower | Python baseline |
| v0.2.0 | 33 ms | 5x faster | Metal GPU (legacy format) |
| v0.7.0 | 314 ms | 2x slower | ISO ADTS + rate allocation |
| v0.11.0 | 320 ms | 2x slower | Single-pass quantization |
| v0.12.0 | 46 ms | 3.4x faster | Metal ISO quantizer |
| **v0.13.1** | **48 ms** | **3.3x faster** | **Short windows + unified backend** |

## License

MIT
