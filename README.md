# MetalAAC

GPU-accelerated AAC-LC encoder/decoder for Apple Silicon.
**3-4x faster than Apple's `afconvert`**, produces standard ADTS decodable by ffmpeg/VLC.

## Performance (M3 Pro)

| | MetalAAC | Apple afconvert | ffmpeg |
|---|---------|-----------------|--------|
| **Encode 60s** | **53 ms** | 161 ms | 399 ms |
| **Encode 300s** | **176 ms** | 709 ms | 1817 ms |
| **Decode 60s** | **69 ms** | — | 123 ms |
| **SNR** | 52.9 dB | — | — |

## Quick Start

```bash
# Setup (requires macOS + Apple Silicon + Python 3.11+)
git clone git@github.com:feiyuehchen/MetalAAC.git && cd MetalAAC
cd metal && make && cd ..
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/  # 82 tests
```

## Usage

### Encode

```python
import numpy as np
from metal_aac.encoder import encode, EncoderConfig

# Mono — just pass a 1D float32 array
pcm = np.random.randn(44100 * 5).astype(np.float32) * 0.3
result = encode(pcm, EncoderConfig(output_format="adts"))

with open("output.aac", "wb") as f:
    f.write(result.bitstream)
# result.actual_bitrate_kbps → ~128
# result.timings → per-stage breakdown
```

```python
# Stereo — pass (N, 2) array, M/S coding is automatic
stereo = np.column_stack([left, right])
result = encode(stereo, EncoderConfig(output_format="adts"))
```

```python
# Custom bitrate
result = encode(pcm, EncoderConfig(
    output_format="adts",
    target_bitrate_kbps=64.0,
    sample_rate=48000,
))
```

### Decode

```python
from metal_aac.decoder import decode

result = decode(open("output.aac", "rb").read())
pcm = result.pcm          # (N,) mono or (N, 2) stereo
sr = result.sample_rate    # auto-detected from bitstream
```

### Measure Quality

```python
from metal_aac.metrics.quality import compute_snr, compute_spectral_convergence

snr = compute_snr(original, decoded)                    # dB, higher=better
sc = compute_spectral_convergence(original, decoded)    # 0-1, lower=better
```

### Benchmark

```bash
python scripts/benchmark.py                              # full benchmark suite
python scripts/compare_all_encoders.py --durations 60    # vs Apple, ffmpeg
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
| `output_format` | `"legacy"` | `"adts"` for standard AAC, `"legacy"` for internal format |
| `target_bitrate_kbps` | `128.0` | Target bitrate in kbps |
| `sample_rate` | `44100` | Input sample rate (8000-96000 Hz) |
| `use_gpu` | `True` | Use Metal + MLX acceleration |
| `enable_window_switching` | `True` | Short-window MDCT for transients |
| `frame_size` | `2048` | MDCT frame size |
| `hop_size` | `1024` | Frame hop (50% overlap) |
| `window_type` | `"kbd"` | Kaiser-Bessel-Derived window |

### `EncoderResult`

| Field | Type | Description |
|-------|------|-------------|
| `bitstream` | `bytes` | Encoded AAC data |
| `actual_bitrate_kbps` | `float` | Achieved bitrate |
| `num_frames` | `int` | Frame count |
| `audio_duration` | `float` | Input duration in seconds |
| `timings` | `dict` | Per-stage timing: framing, mdct, quantization, huffman_bitstream, ... |

### `decode(bitstream, config=None) → DecoderResult`

| Parameter | Type | Description |
|-----------|------|-------------|
| `bitstream` | `bytes` | ADTS or legacy AAC data (auto-detected) |
| `config` | `DecoderConfig` | Optional |

### `DecoderConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `use_gpu` | `True` | Use MLX for dequant + IMDCT |
| `output_length` | `None` | Truncate output to this many samples |

### `DecoderResult`

| Field | Type | Description |
|-------|------|-------------|
| `pcm` | `np.ndarray` | `(N,)` mono or `(N, 2)` stereo float32 |
| `sample_rate` | `int` | From bitstream header |
| `num_frames` | `int` | Decoded frame count |
| `timings` | `dict` | Per-stage timing: huffman_bitstream, dequantization, imdct, overlap_add |

### Quality Metrics

```python
compute_snr(original, reconstructed) → float       # dB
compute_spectral_convergence(original, reconstructed,
                              n_fft=2048, hop_length=512) → float  # 0-1
```

### Test Signals

```python
from metal_aac.data.synthetic import generate_test_signals

signals = generate_test_signals()  # dict of 12 deterministic signals
# Keys: sine_440, sine_1k, sine_4k, multitone, chirp,
#        white_noise, pink_noise, silence, impulse,
#        music_1min, music_5min, long_music
# Each: {"pcm": ndarray, "sample_rate": 44100, "duration": float}
```

---

## Architecture

```
Encoder pipeline (53ms for 60s mono):

  PCM → Frame → MDCT → Quantize → Huffman → ADTS
        MLX     MLX     Metal      Metal
        2ms     17ms    8ms        15ms

Decoder pipeline (69ms for 60s mono):

  ADTS → Huffman → Dequant → IMDCT → PCM
         C+GCD     MLX       MLX
         ~0ms      7ms       21ms
```

### Metal kernels

| Kernel | Threads | Purpose |
|--------|---------|---------|
| `kernel_quantize_iso` | 1024/frame | ISO binary search with parallel bit-count reduction |
| `kernel_encode_raw_data_block` | 1024/frame | ISO Huffman spectral + SF encoding |
| `kernel_quantize` | 1024/frame | Legacy quantization (exp-Golomb) |
| `kernel_compute_codewords` | N×B | Exp-Golomb codeword computation |
| `kernel_prefix_sum` | 1024/frame | Blelloch exclusive scan |
| `kernel_scatter_write` | 1024/frame | Atomic bit packing |
| `kernel_decode_frames` | 1/frame | Legacy exp-Golomb decode |

### Native C functions (GCD parallel)

| Function | Purpose |
|----------|---------|
| `metal_decode_iso_frames` | ISO Huffman decode with LUT, `dispatch_apply` |
| `metal_quantize_iso` | Metal GPU ISO quantization dispatch |

## Project Structure

```
MetalAAC/
├── metal/                     # Metal shaders + native C
│   ├── huffman_kernels.metal  # GPU compute kernels
│   ├── metal_huffman.h/.m     # ObjC host + C decoder
│   └── Makefile
├── src/metal_aac/
│   ├── encoder.py             # encode() — mono + stereo
│   ├── decoder.py             # decode() — auto-detect format
│   ├── core/
│   │   ├── quantization.py    # ISO quantizer (MLX fallback)
│   │   ├── raw_data_block.py  # ISO bitstream read/write
│   │   ├── adts.py            # ADTS container
│   │   ├── mdct.py            # MDCT/IMDCT (CPU + MLX)
│   │   ├── psychoacoustic.py  # Bark-scale masking
│   │   └── metal_bridge.py    # ctypes bridge to native lib
│   ├── tables/                # Huffman codebooks, SFB tables
│   ├── metrics/               # SNR, spectral convergence
│   └── data/                  # Test signal generation
├── tests/                     # 82 tests
├── DATASET.md                 # Test signal spec
├── BENCHMARK.md               # Full benchmark results
└── CHANGELOG.md               # v0.1.0 → v0.12.0
```

## Version History

| Version | Encode 60s | vs Apple | Key Change |
|---------|-----------|----------|------------|
| v0.1.0 | ~9000 ms | 57x slower | Python baseline |
| v0.2.0 | 33 ms | 5.2x faster | Metal GPU pipeline (legacy) |
| v0.7.0 | 314 ms | 2x slower | ISO ADTS + rate allocation |
| v0.11.0 | 320 ms | 2x slower | Single-pass + merged framing |
| **v0.12.0** | **53 ms** | **3x faster** | **Metal ISO quantizer** |

## License

MIT
