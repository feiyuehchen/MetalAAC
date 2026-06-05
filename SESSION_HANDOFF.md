# MetalAAC Session Handoff

> Last updated: 2026-06-05, v0.12.0
> Repo: git@github.com:feiyuehchen/MetalAAC.git
> Branch: main
> 82 tests passing

## Current State

GPU-accelerated AAC-LC encoder/decoder for Apple Silicon. ISO/IEC 14496-3 compliant
ADTS output, mono + stereo M/S, decodable by ffmpeg/VLC.

### Performance (60s mono, M3 Pro)

| | MetalAAC | Apple afconvert | ffmpeg |
|---|---------|-----------------|--------|
| Encode | **46 ms** | 157 ms | 390 ms |
| Decode | **56 ms** | — | 117 ms |

### Quality

| Signal | SNR | Bitrate |
|--------|-----|---------|
| 440 Hz sine | 53.4 dB | 107 kbps |
| Music (1 min) | 46.5 dB | 111 kbps |
| Stereo correlated | 51.5 dB | 52 kbps |
| Stereo uncorrelated | 51.4/50.5 dB | 100 kbps |

### Pipeline

```
Encode: PCM → Frame(MLX) → MDCT(MLX) → Quantize(Metal) → Huffman(Metal) → ADTS
        3ms    27ms         9ms          15ms

Decode: ADTS → Huffman(C+GCD) → Dequant(MLX) → IMDCT(MLX) → PCM
        ~0ms     9ms            19ms
```

## Known Limitations

- Bitrate varies 105-135 kbps at 128 target (no VBR smoothing)
- Correlated stereo under-utilizes bits (52 kbps)
- `masking_thresholds` unused in ISO quantizer
- No TNS, PNS, or bit reservoir re-quantization

## Key Files

| File | Purpose |
|------|---------|
| `metal/huffman_kernels.metal` | GPU kernels (quantize, Huffman encode) |
| `metal/metal_huffman.m` | ObjC host + C ISO decoder (GCD) |
| `src/metal_aac/encoder.py` | Encode pipeline (mono + stereo) |
| `src/metal_aac/decoder.py` | Decode pipeline (mono + stereo) |
| `src/metal_aac/core/quantization.py` | ISO quantizer (MLX fallback) |
| `src/metal_aac/core/raw_data_block.py` | ISO bitstream reader/writer |
| `src/metal_aac/core/metal_bridge.py` | ctypes bridge to native lib |

## Development

```bash
cd metal && make                        # Build native lib
.venv/bin/python3 -m pytest tests/      # Run tests
```

Follows [ai-research-code-workflow.md](../ai-research-code-workflow.md).
DATASET.md/BENCHMARK.md changes = MAJOR bump.
