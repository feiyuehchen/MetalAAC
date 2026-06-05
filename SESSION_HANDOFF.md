# MetalAAC Session Handoff

> Last updated: 2026-06-05, v0.13.1
> Repo: git@github.com:feiyuehchen/MetalAAC.git
> Branch: main (pushed)
> 82 tests passing

## Current State

GPU-accelerated AAC-LC encoder/decoder for Apple Silicon.
**3.3x faster than Apple afconvert**, ISO ADTS compliant, mono + stereo M/S.

### Performance (60s mono, M3 Pro)

| | MetalAAC | Apple afconvert | ffmpeg |
|---|---------|-----------------|--------|
| Encode | **48 ms** | 158 ms | 390 ms |
| Decode | **56 ms** | — | 117 ms |
| SNR | 52.9 dB | — | — |

### Backend Architecture

One-time detection at startup (`has_metal()`), then consistent path:
```
Metal path (primary):
  Encode: Metal quantize → Metal Huffman → ADTS
  Decode: C+GCD Huffman → MLX dequant → MLX IMDCT → PCM

MLX fallback (no dylib):
  Encode: MLX quantize → Python Huffman → ADTS
  Decode: Python Huffman → NumPy dequant → CPU IMDCT → PCM
```

### What Works

- ISO ADTS output decodable by ffmpeg/VLC (0 frame errors on real audio)
- Mono + stereo (CPE with M/S coding)
- Short-window MDCT (EIGHT_SHORT_SEQUENCE with ISO SFB reordering)
- Bit reservoir tracking + ADTS buffer_fullness
- Window switching (transient detection)
- 11 Huffman codebooks + SF codebook (bit-exact with ffmpeg)

### Known Limitations

- Bitrate 105-138 kbps at 128 target (no VBR smoothing)
- `masking_thresholds` unused in Metal ISO quantizer
- No TNS, PNS
- Metal kernel `kernel_encode_raw_data_block` doesn't handle short windows
  (encoder routes short-window frames to Python path)
- C decoder doesn't handle CPE (Python decoder handles stereo)

### Key Files

| File | Purpose |
|------|---------|
| `metal/huffman_kernels.metal` | `kernel_quantize_iso` + `kernel_encode_raw_data_block` |
| `metal/metal_huffman.m` | ObjC dispatch + C ISO decoder (GCD) |
| `src/metal_aac/core/metal_bridge.py` | `has_metal()`, `metal()`, ctypes bridge |
| `src/metal_aac/encoder.py` | `encode()` — mono + stereo |
| `src/metal_aac/decoder.py` | `decode()` — auto-detect format + channels |
| `src/metal_aac/core/quantization.py` | MLX fallback + `reorder_short_to_iso()` |
| `src/metal_aac/core/raw_data_block.py` | ISO bitstream read/write (SCE + CPE) |

### Development

```bash
cd metal && make                        # Build native lib
.venv/bin/python3 -m pytest tests/      # 82 tests
```
