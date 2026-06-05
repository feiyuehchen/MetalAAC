# Changelog

This project follows a research-adapted SemVer (see ai-research-code-workflow.md).
DATASET.md and BENCHMARK.md changes trigger MAJOR bumps.

## [Unreleased]

## [0.10.4] - 2026-06-04

### Changed
- **MLX GPU dequantization** in all decode paths (CPU+GPU mono, stereo).
  Replaced numpy per-coefficient power with MLX vectorized computation.
- **MLX GPU IMDCT** in CPU decode path (was using numpy matmul fallback).
- **Vectorized inverse M/S**: uses `np.where` with sfb_map mask instead of
  per-SFB × per-frame Python loop.
- GPU decode path: sfb_map indexing moved from numpy to MLX.

### Performance
- Decode 60s: **102 → 57 ms** GPU, **102 → 61 ms** CPU
- Dequantization: **45 → 7 ms** (MLX GPU)
- Total decode speedup from v0.10.1: **42x** (2386 → 57 ms)

## [0.10.3] - 2026-06-04

### Added
- **Native C ISO Huffman decoder**: `metal_decode_iso_frames()` in ObjC with
  GCD `dispatch_apply` for multi-frame parallelism. LUT-based O(1) decode
  per codeword. Cached decode LUTs (built once, reused across calls).

### Performance
- Decode 60s: **2386 → 102 ms** (23x speedup from original Python)
- Huffman parsing: **2132 → ~0 ms** (native C eliminates Python loop)
- Remaining bottleneck: dequantization 46ms + IMDCT 22ms (numpy/MLX)

## [0.10.2] - 2026-06-04

### Changed
- **LUT Huffman decoder**: replaced bit-by-bit tree traversal with flat lookup
  tables (O(1) per codeword). Precomputed value tables for index→tuple mapping.
- **Fast BitReader**: big-integer accumulator with O(1) peek/read/skip instead
  of per-bit byte indexing.
- **Vectorized dequantization**: `dequantize_iso_cpu` uses sfb_map broadcasting
  instead of per-SFB Python loop.

### Performance
- Decode 60s: **2386 → 1391 ms** (1.7x speedup)
- Dequantization: **269 → 46 ms** (5.8x)
- Huffman parsing: **2132 → 1354 ms** (1.6x, still 95% — limited by Python loop)

## [0.10.1] - 2026-06-04

### Fixed
- **`_write_ics` wrong window detection**: was checking codebook index instead of
  window_sequence for short/long section encoding. Added `window_sequence` parameter.
- **`decode_cpe_iso` crash on common_window=0**: unbound variables when ics_info
  is per-channel. Now reads per-channel ics_info when common_window=0.
- **Stereo encode crash without MLX**: raises clear error instead of NameError.
- **`global_gain` DPCM anchor bias**: was averaging ALL SFBs including zero-codebook
  bands. Now averages only non-zero bands, reducing wasted DPCM Huffman bits.
- **Stereo calibration missing ms_used**: calibration sample frames now include M/S
  flag bits, matching final encoding.
- **M/S docstring + dead code**: removed unused `sqrt2`, corrected docstring to
  match ISO convention `M=(L+R)/2, S=(L-R)/2`.
- **M/S GPU roundtrip**: replaced NumPy loop with MLX vectorized `mx.where` —
  stays on GPU, no CPU transfer.
- Removed dead `_encode_adts_frames` function (legacy format, never called).
- Documented `masking_thresholds` as reserved/unused in `quantize_batch_gpu`.

## [0.10.0] - 2026-06-04

### Added
- **M/S stereo coding**: per-SFB mid/side transform in MDCT domain. Uses M/S
  when side energy < both L and R energy (correlated channels). Writes
  `ms_mask_present` (0/1/2) and per-SFB `ms_used` flags in CPE bitstream.
  ISO-compatible: `M=(L+R)/2, S=(L-R)/2`, decoder `L=M+S, R=M-S`.
- **Bit reservoir**: tracks running surplus/deficit across frames. Writes correct
  `buffer_fullness` in ADTS headers. Maximum 6144×nch bits (AAC-LC standard).

### Quality (stereo, 128 kbps target, ffmpeg decode)
- Correlated (L=R): **51.7 dB** SNR both channels (M/S: 100% SFBs, S≈0)
- Uncorrelated (L=440Hz, R=1kHz): **51.6 / 50.7 dB** (M/S: 8% SFBs)
- Both cases decodable by ffmpeg without errors

## [0.9.0] - 2026-06-04

### Added
- **Stereo encoding**: accepts `(N, 2)` float32 stereo input. Each channel is
  independently MDCT'd, quantized, and encoded into a CPE (Channel Pair Element)
  with common_window and shared ics_info. No M/S coding yet (dual-independent).
- **Stereo decoding**: auto-detects CPE (channel_config=2) in ADTS, decodes two
  channels to `(N, 2)` stereo output.
- `encode_cpe_iso()` / `decode_cpe_iso()` for CPE bitstream format.
- `_write_ics()` / `_read_ics_body()` refactored as shared helpers for SCE and CPE.
- Adaptive bitrate calibration applied to stereo path.

### Quality (stereo, 128 kbps target)
- 440 Hz L + 880 Hz R: **132 kbps**, ffmpeg SNR **51.7 / 51.2 dB**
- ffmpeg correctly decodes stereo output

## [0.8.1] - 2026-06-04

### Fixed
- **Adaptive bitrate calibration**: two-pass quantization samples 4 frames to
  measure exp-Golomb vs Huffman bit ratio, then re-quantizes with corrected
  target (capped at 1.15x). Bitrate range narrowed from [107, 157] to [124, 136] kbps.

### Quality
- Average bitrate: **130.6 kbps** (was ~120 kbps, target 128)
- Music: **124.6 kbps** (was 111.6 kbps)
- 60s encode: 112ms (vs 82ms without calibration — two-pass overhead)

## [0.8.0] - 2026-06-04

### Added
- **ADTS decoder**: `decode_raw_data_block_iso()` parses ISO Huffman-coded
  raw_data_blocks (section_data, SF DPCM, 11 spectral codebooks, escape coding).
  Decoder auto-detects ADTS vs legacy format.
- **ISO dequantizer**: `dequantize_iso_cpu()` uses `x_hat = sign(q) * |q|^(4/3) * 2^((sf-200)/4)`.
- BitReader class for bit-level parsing.
- Huffman decode trees built lazily from codebook tables.
- Full encode → decode roundtrip without ffmpeg dependency.

### Quality
- Our decoder vs ffmpeg decoder: **123.7 dB agreement** (floating-point identical)
- Roundtrip SNR (encode → our decode): 26.9 dB (440 Hz, 5s)

## [0.7.0] - 2026-06-04

### Changed
- **Clipping-aware per-band SF allocation**: quantizer now computes per-band safe SF
  from peak amplitude (avoids q > 200 clipping), then binary-searches sf_base over
  [100, 255] to fill bit budget. Signal bands are pinned at their safe SF; noise-floor
  bands follow sf_base downward, becoming non-zero to contribute bits.
- Binary search widened from [170, 210] to [100, 255] with 12 iterations.

### Quality (ADTS path, ffmpeg decode)
- 440 Hz sine: **SNR = 51.5 dB** (was 46.6 dB), **bitrate = 107 kbps** (was 10 kbps)
- 1 kHz sine: **SNR = 50.6 dB** (was 46.9 dB), **bitrate = 106 kbps** (was 10 kbps)
- Multi-tone: **SNR = 50.2 dB** (was 34.8 dB), **bitrate = 112 kbps** (was 17 kbps)
- Music: **SNR = 46.1 dB**, **bitrate = 112 kbps**
- Average nonzero coefficients per frame: **~500** (was ~10-30)

## [0.6.1] - 2026-06-04

### Changed
- **Metal ADTS kernel uses ISO-native scalefactors directly**: removed legacy SF
  mapping formula (`200 - (gg - sf_int*4)/3`), which was redundant since v0.6.0's
  ISO-native quantizer already outputs direct ISO SF values.
- **ADTS encoder path now uses Metal GPU** instead of Python `encode_raw_data_block_iso`
  loop, with automatic CPU fallback.

### Performance
- ADTS Huffman+bitstream: **238ms → 10ms** (24x speedup)
- Total ADTS 60s encode: **314ms → 82ms** (3.8x speedup)
- Output is **byte-identical** to Python path (verified all frames)

### Fixed
- `pyproject.toml` version synced with git tags (was 0.4.0, now 0.6.1)

## [0.6.0] - 2026-06-04

### Changed
- **ISO-native quantizer**: `quantize_batch_gpu()` now outputs direct ISO SF values
  (100-255) instead of internal format. No mapping needed for ADTS bitstream.
- ISO formula: `q = nint((|x| * 2^((200-sf)/4))^0.75)`

### Quality
- 440 Hz sine: **SNR = 46.6 dB** (was 25.6 dB in v0.5.0)
- 1 kHz sine: **SNR = 46.9 dB**
- Multi-tone: **SNR = 34.8 dB**

## [0.5.0] - 2026-06-04

### Changed
- **MDCT 2/N normalization**: forward basis includes 2/N factor so coefficients are
  in PCM scale. Inverse basis has no additional factor.
- ISO standard quantization formulas now work correctly with normalized MDCT.

### Fixed
- 4 compliance fixes: buffer_fullness, bitrate calculation, short window section
  escape, SF tuning for normalized MDCT.

### Quality
- 440 Hz sine: **SNR = 25.6 dB** (was 13.3 dB in v0.4.0)

## [0.4.0] - 2026-06-03

### Fixed
- **ISO scalefactor calibration**: encode → ffmpeg decode now reconstructs correct
  amplitude (ratio 0.95-0.99x). Formula: `iso_sf = 157 - (gg - sf_int*4) / 3`,
  accounting for ffmpeg's POW_SF2_ZERO=200 and MDCT normalization mismatch.
- DPCM prev_sf tracking: was tracking intended value instead of decoder's clamped
  state, causing cascading SF errors when diff exceeded ±60.
- Metal/Python integer division mismatch: added rounding in Metal to match Python.
- Escape coding off-by-one: `(count+5)` → `(count+4)` per ISO 14496-3 4.6.3.3.
- CPU ADTS path now uses `encode_raw_data_block` (was using legacy exp-Golomb).
- Short window `sect_esc_val` corrected to 7 with 3-bit increments (was 31/5-bit).

### Added
- max |q| ≤ 255 constraint in Metal quantization binary search (parallel max-abs
  reduction alongside bit sum).

### Quality
- 440 Hz sine: **SNR = 13.3 dB**, amplitude ratio = 0.99x
- 1 kHz sine: **SNR = 13.1 dB**, amplitude ratio = 0.95x
- Multi-tone: **SNR = 7.0 dB**, amplitude ratio = 0.90x

## [0.3.0] - 2026-06-03

### Added
- **ISO/IEC 14496-3 Huffman codebooks**: all 11 spectral codebooks + scalefactor
  codebook, exact values from ffmpeg libavcodec/aactab.c.
- **Metal `kernel_encode_raw_data_block`**: writes complete ISO-compliant SCE
  (ics_info + section_data + SF_data + spectral_data + ID_END) on GPU. Codebook
  LUTs (~10KB) in Metal constant memory.
- **`raw_data_block.py`**: Python reference implementation of ISO raw_data_block
  encoder with BitWriter (bigint accumulation).
- **ADTS bitstream format**: 7-byte fixed header (0xFFF sync, MPEG-2, LC profile).
  Decoder auto-detects ADTS vs legacy format.
- **Window switching**: transient detection (energy ratio, CPU + MLX GPU) + 4-state
  window sequence machine (ONLY_LONG / LONG_START / EIGHT_SHORT / LONG_STOP).
- **Short-window MDCT**: 8×256-pt via MLX batch matmul reshape `(B,2048)→(B*8,256)`.
- **Transition window shapes**: long-start and long-stop per ISO 11.2.1.
- **Short-window SFB tables**: 44100/48000/32000 Hz (128 MDCT lines).
- **Performance regression tests**: 12 tests with per-stage timing thresholds.
- **Pipeline restructure**: psychoacoustic runs parallel from raw PCM (transient
  detection pre-MDCT, masking post-MDCT), matching ISO encoder block diagram.

### Performance
- ADTS encode 60s: **39ms** (all GPU) — parity with legacy Metal path (40ms)
- Metal ISO Huffman: 400ms Python → 9ms Metal (**44x speedup**)

### Changed
- Default `output_format` remains `"legacy"` (ADTS opt-in via config) until
  two-loop quantizer improves decoded quality.
- ADTS header `id` bit set to 1 (MPEG-2) to match afconvert convention.

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
