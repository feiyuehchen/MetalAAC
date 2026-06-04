# MetalAAC Session Handoff

> Last updated: 2026-06-04, v0.6.0
> Repo: git@github.com:feiyuehchen/MetalAAC.git
> Branch: main (clean, all committed)
> 82 tests passing

## What This Is

A GPU-accelerated AAC-LC encoder for Apple Silicon using MLX + Metal compute
shaders. Produces ISO/IEC 14496-3 compliant ADTS output decodable by ffmpeg/VLC.

## Current State (v0.6.0)

### What Works ✅

| Component | Implementation | Status |
|-----------|---------------|--------|
| MDCT (long + short window) | MLX batch matmul, 2/N normalized | ✅ ISO standard |
| Psychoacoustic model | MLX GPU (FFT + bark-scale spreading) | ✅ functional |
| Quantization (ADTS path) | MLX ISO formula: `q = nint((|x| * 2^((200-sf)/4))^0.75)` | ✅ ISO standard |
| Quantization (legacy path) | Metal GPU binary search kernel | ✅ fast (22ms/300s) |
| Huffman encoding (ADTS) | Python `encode_raw_data_block_iso()` | ✅ ISO standard, slow (~300ms) |
| Huffman encoding (legacy) | Metal 4-kernel pipeline | ✅ fast (30ms/300s) |
| Metal ISO encoding kernel | `kernel_encode_raw_data_block` | ✅ exists but uses OLD SF mapping |
| ADTS header | 0xFFF sync, MPEG-2, LC profile, buffer_fullness=0 | ✅ matches afconvert |
| Window switching | Transient detect + state machine + short MDCT | ✅ wired in |
| 11 Huffman codebooks | Exact from ffmpeg aactab.c | ✅ bit-exact |
| SF Huffman codebook | 121 entries, exact from ffmpeg | ✅ bit-exact |
| ffmpeg decode | Zero errors on all test signals | ✅ |

### Quality

| Signal | SNR | Amplitude ratio | Bitrate |
|--------|-----|-----------------|---------|
| 440 Hz sine | 46.6 dB | 1.02x | 10.0 kbps |
| 1 kHz sine | 46.9 dB | 1.05x | 10.3 kbps |
| Multi-tone | 34.8 dB | 1.02x | 17.3 kbps |

### Performance (legacy path, 60s encode)

40 ms on M3 Pro (5.2x faster than Apple afconvert at 300s)

## Known Problems ❌

### P1: ADTS path is slow (~300ms/60s vs legacy 40ms)

The ADTS path uses Python `encode_raw_data_block_iso()` for Huffman encoding
because the Metal `kernel_encode_raw_data_block` still uses the OLD SF mapping
formula (`200 - (gg - sf_int*4)/3`). Now that the quantizer outputs ISO SFs
directly, the Metal kernel needs updating to use them without conversion.

**Fix:** Update `kernel_encode_raw_data_block` in `metal/huffman_kernels.metal`
to accept ISO SFs directly (remove the `isf = 200 - ...` computation, use
the sf values as-is). Then update `metal_bridge.py::encode_adts_frames()` to
pass the ISO SFs. This is a small change — the kernel structure is correct,
only the SF handling needs updating.

### P2: Bitrate under-utilization (10 kbps vs 128 kbps target)

The ISO quantizer's binary search on sf_base converges to sf≈195 where only
~20 MDCT bins have non-zero q values. The remaining 1004 bins have q=0 (1 bit
each = 1004 bits), totaling ~1200 bits vs the 2972-bit target.

**Root cause:** Normalized MDCT coefficients are ~1.0 for signal and ~0.001 for
noise floor. At sf=195, the quantization step is ~1.0, so only coefficients
with |x| > 0.5 survive. The noise floor rounds to zero.

**Attempted fixes (all degraded SNR):**
- Lower sf_cap uniformly → amplifies noise floor → SNR drops to 0 dB
- Noise-floor SF allocation → inconsistent SF mapping → SNR drops to 0 dB
- Nested sf_cap binary search → same issue

**Proper fix requires:** Psychoacoustic-driven per-band rate allocation where
the masking model outputs meaningful thresholds that tell the quantizer "this
much noise is acceptable in this band." Currently the masking thresholds are
~1e-7 (too low to differentiate bands). Key files:
- `src/metal_aac/core/psychoacoustic.py` — needs tonal/noise masking separation
- `src/metal_aac/core/quantization.py::quantize_batch_gpu()` — needs outer SF
  loop driven by masking, not just rate

### P3: No ADTS decoder

The decoder (`src/metal_aac/decoder.py`) only decodes the legacy exp-Golomb
format. For ADTS, we rely on ffmpeg. Adding an ADTS decoder requires parsing
standard Huffman codebook spectral data (the reverse of `encode_raw_data_block_iso`).

### P4: Features not yet implemented

- **Stereo (M/S + intensity)** — Phase 3 in the plan
- **TNS (Temporal Noise Shaping)** — Phase 4
- **Bit reservoir** — Phase 4
- **PNS (Perceptual Noise Substitution)** — nice-to-have

## Architecture

```
MetalAAC/
├── metal/                          # GPU compute shaders
│   ├── huffman_kernels.metal       # 8 Metal kernels (quant, Huffman, ISO encode)
│   ├── metal_huffman.h/.m          # ObjC host + C API
│   └── Makefile                    # xcrun clang -arch arm64
│
├── src/metal_aac/
│   ├── core/
│   │   ├── mdct.py                 # MDCT/IMDCT (2/N normalized, CPU+GPU)
│   │   ├── psychoacoustic.py       # Bark-scale masking (CPU+GPU)
│   │   ├── quantization.py         # ISO quantizer (quantize_batch_gpu) + legacy
│   │   ├── raw_data_block.py       # ISO bitstream writer (encode_raw_data_block_iso)
│   │   ├── huffman.py              # Legacy exp-Golomb coding
│   │   ├── adts.py                 # ADTS header writer/reader
│   │   ├── window_switching.py     # Transient detection + state machine
│   │   ├── bitstream.py            # Legacy frame format
│   │   └── metal_bridge.py         # ctypes wrapper for Metal dylib
│   ├── encoder.py                  # Pipeline: PCM → ADTS or legacy bitstream
│   ├── decoder.py                  # Decoder (legacy only)
│   └── tables/
│       ├── huffman_tables.py       # 11 codebooks + SF codebook (from ffmpeg)
│       ├── scalefactor_bands.py    # SFB tables (long + short)
│       └── windows.py             # KBD, sine, transition windows
│
├── DATASET.md                      # Research contract: test signals
├── BENCHMARK.md                    # Research contract: metrics + results
├── CHANGELOG.md                    # Version history (v0.1.0 → v0.6.0)
└── tests/                          # 82 tests (correctness + performance)
```

## Key Formulas

### ISO Quantization (v0.6.0, in quantize_batch_gpu)
```
Encode: q = nint((|x| * 2^((200-sf)/4))^0.75)
Decode: x_hat = |q|^(4/3) * 2^((sf-200)/4)
```
Where sf ∈ [100, 255] is the ONLY parameter per band.
sf=200 → unity scale (q ≈ |x|^0.75).
Lower sf → finer quantization → more bits.

### MDCT Normalization
Forward basis includes 2/N factor so coefficients are in PCM scale.
Inverse basis has NO additional factor (raw cosine sum).

### Escape Coding
```python
count = 0
while n >= (1 << (count + 5)): count += 1
write(count ones + 0 + n in (count+4) bits)
```
Decoder: `value = (1 << (count+4)) | read_bits(count+4)`

## Development Workflow

- Follow `/Users/fychen/research/ai-research-code-workflow.md`
- Changes to DATASET.md or BENCHMARK.md = MAJOR bump
- New features = MINOR bump, must have tests
- All changes via PR to main
- Code review before merge (use /code-review skill)
- Build Metal: `cd metal && make`
- Run tests: `.venv/bin/python3 -m pytest tests/`
- Test ffmpeg: encode → `ffmpeg -i out.aac -f f32le decoded.raw`

## Version History

| Version | Key Change | SNR |
|---------|-----------|-----|
| v0.1.0 | Python baseline | — |
| v0.2.0 | Metal GPU pipeline, 5.2x faster than Apple | — |
| v0.3.0 | ISO Huffman + ADTS, ffmpeg decodable | — |
| v0.4.0 | ISO SF calibration | 13.3 dB |
| v0.5.0 | MDCT 2/N normalization | 25.6 dB |
| v0.6.0 | ISO-native quantizer | 46.6 dB |
