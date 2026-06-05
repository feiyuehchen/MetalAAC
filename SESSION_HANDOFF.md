# MetalAAC Session Handoff

> Last updated: 2026-06-04, v0.10.3
> Repo: git@github.com:feiyuehchen/MetalAAC.git
> Branch: main (clean, all committed)
> 82 tests passing

## What This Is

A GPU-accelerated AAC-LC encoder/decoder for Apple Silicon using MLX + Metal compute
shaders + native C. Produces ISO/IEC 14496-3 compliant ADTS output decodable by
ffmpeg/VLC. Supports mono and stereo with M/S coding.

## Current State (v0.10.3)

### What Works

| Component | Implementation | Status |
|-----------|---------------|--------|
| MDCT (long + short window) | MLX batch matmul, 2/N normalized | ISO standard |
| Psychoacoustic model | MLX GPU (FFT + Bark-scale spreading) | Functional |
| Quantization | MLX ISO formula, clipping-aware per-band SF | ISO standard |
| Huffman encoding (ADTS) | Metal GPU kernel | ISO standard |
| Huffman decoding (ADTS) | Native C + GCD parallel | ISO standard, 23x faster |
| ADTS header | 0xFFF sync, MPEG-2, LC profile | Matches afconvert |
| Window switching | Transient detect + state machine + short MDCT | Wired in |
| 11 Huffman codebooks | Exact from ffmpeg aactab.c | Bit-exact |
| SF Huffman codebook | 121 entries, exact from ffmpeg | Bit-exact |
| Stereo (CPE) | M/S coding, per-SFB decision | ISO standard |
| Bit reservoir | Tracking + ADTS buffer_fullness | Functional |
| ffmpeg decode | Zero errors on all test signals | Verified |
| Internal decoder | ISO Huffman + dequant + IMDCT | 74ms/60s GPU |

### Quality (ADTS path, ffmpeg decode)

| Signal | SNR | Bitrate |
|--------|-----|---------|
| 440 Hz sine | 51.6 dB | 135 kbps |
| 1 kHz sine | 50.6 dB | 127 kbps |
| Multi-tone | 50.3 dB | 128 kbps |
| Music | 48.1 dB | 130 kbps |
| Stereo correlated | 51.7 dB | 67 kbps |
| Stereo uncorrelated | 51.3/50.5 dB | 131 kbps |

### Performance (60s mono)

| Operation | Time |
|-----------|------|
| Encode (ADTS) | 505 ms |
| Decode (GPU) | 74 ms |
| Decode (CPU) | 102 ms |

## Known Limitations

### Encoder bottlenecks
- Transient detection: 166 ms (28%) — double framing pass
- Quantization: 309 ms (53%) — 2-pass adaptive calibration in MLX
- Total encode ~12x slower than Apple afconvert

### Missing features
- **TNS** (Temporal Noise Shaping) — needs LPC analysis
- **PNS** (Perceptual Noise Substitution) — nice-to-have
- **Bit reservoir re-quantization** — currently tracks only, no re-encoding
- **masking_thresholds** parameter unused in quantizer (reserved)

### Quality gaps
- Bitrate varies [125, 136] kbps at 128 target (no VBR smoothing)
- Correlated stereo under-utilizes bits (67 kbps, side channel is zero)

## Architecture

```
MetalAAC/
├── metal/                          # Native code
│   ├── huffman_kernels.metal       # 8 Metal compute kernels
│   ├── metal_huffman.h/.m          # ObjC host + C ISO decoder (GCD)
│   └── Makefile                    # xcrun clang -arch arm64
│
├── src/metal_aac/
│   ├── core/
│   │   ├── mdct.py                 # MDCT/IMDCT (2/N normalized, CPU+GPU)
│   │   ├── psychoacoustic.py       # Bark-scale masking (CPU+GPU)
│   │   ├── quantization.py         # ISO quantizer + clipping-aware SF
│   │   ├── raw_data_block.py       # ISO bitstream encoder/decoder (SCE+CPE)
│   │   ├── adts.py                 # ADTS header writer/reader
│   │   ├── window_switching.py     # Transient detection + state machine
│   │   ├── metal_bridge.py         # ctypes wrapper for native dylib
│   │   ├── huffman.py              # Legacy exp-Golomb coding
│   │   └── bitstream.py            # Legacy frame format
│   ├── encoder.py                  # Pipeline: PCM -> ADTS (mono + stereo)
│   ├── decoder.py                  # Pipeline: ADTS -> PCM (mono + stereo)
│   └── tables/
│       ├── huffman_tables.py       # 11 codebooks + SF codebook
│       ├── scalefactor_bands.py    # SFB tables (long + short)
│       └── windows.py             # KBD, sine, transition windows
│
├── DATASET.md                      # Research contract: test signals
├── BENCHMARK.md                    # Research contract: metrics + results
├── CHANGELOG.md                    # v0.1.0 -- v0.10.3
└── tests/                          # 82 tests
```

## Key Formulas

### ISO Quantization
```
Encode: q = nint((|x| * 2^((200-sf)/4))^0.75)
Decode: x_hat = |q|^(4/3) * 2^((sf-200)/4)
```

### M/S Stereo (ISO convention)
```
Encode: M = (L+R)/2,  S = (L-R)/2
Decode: L = M+S,      R = M-S
```

### Clipping-aware SF allocation
```
safe_sf[band] = ceil(200 - 4 * log2(1516 / |x_max|))
per_band_sf = max(sf_base, safe_sf)
Binary search on sf_base [100, 255] to meet bit target
```

## Development Workflow

- Follow `/Users/fychen/research/ai-research-code-workflow.md`
- Changes to DATASET.md or BENCHMARK.md = MAJOR bump
- New features = MINOR bump, must have tests
- Build Metal: `cd metal && make`
- Run tests: `.venv/bin/python3 -m pytest tests/`
- Test ffmpeg: encode -> `ffmpeg -i out.aac -f f32le decoded.raw`

## Version History

| Version | Key Change | SNR | Bitrate |
|---------|-----------|-----|---------|
| v0.1.0 | Python baseline | -- | -- |
| v0.2.0 | Metal GPU pipeline | -- | legacy |
| v0.3.0 | ISO Huffman + ADTS | -- | legacy |
| v0.4.0 | ISO SF calibration | 13.3 dB | 10 kbps |
| v0.5.0 | MDCT 2/N normalization | 25.6 dB | 10 kbps |
| v0.6.0 | ISO-native quantizer | 46.6 dB | 10 kbps |
| v0.6.1 | Metal ADTS ISO SF | 46.6 dB | 10 kbps |
| v0.7.0 | Clipping-aware rate alloc | 51.5 dB | 107 kbps |
| v0.8.0 | ADTS decoder | 51.5 dB | 107 kbps |
| v0.8.1 | Bitrate calibration | 51.5 dB | 130 kbps |
| v0.9.0 | Stereo CPE | 51.7 dB | 132 kbps |
| v0.10.0 | M/S stereo + reservoir | 51.7 dB | 132 kbps |
| v0.10.1 | 9 code review fixes | -- | -- |
| v0.10.2 | LUT Huffman decode | -- | -- |
| v0.10.3 | Native C+GCD decoder | -- | 74ms decode |
