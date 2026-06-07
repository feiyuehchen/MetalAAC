#ifndef METAL_HUFFMAN_H
#define METAL_HUFFMAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MetalHuffmanCtx MetalHuffmanCtx;

MetalHuffmanCtx* metal_huffman_create(void);
void metal_huffman_destroy(MetalHuffmanCtx* ctx);

int metal_huffman_encode(
    MetalHuffmanCtx* ctx,
    const int32_t* quantized,
    const int32_t* scalefactors,
    const int32_t* global_gains,
    int32_t B,
    int32_t N,
    int32_t num_sfb,
    uint8_t* output_buf,
    int32_t max_frame_bytes,
    int32_t* frame_sizes
);

int metal_huffman_decode(
    MetalHuffmanCtx* ctx,
    const uint8_t* compressed,
    const int32_t* frame_offsets,
    int32_t B,
    int32_t N,
    int32_t num_sfb,
    int32_t* quantized_out,
    int32_t* scalefactors_out,
    int32_t* global_gains_out
);

int metal_huffman_max_frame_bytes(int32_t N, int32_t num_sfb);

int metal_encode_adts_frames(
    MetalHuffmanCtx* ctx,
    const int32_t* quantized,
    const int32_t* scalefactors,
    const int32_t* global_gains,
    const int32_t* window_seqs,
    const int32_t* sfb_offsets_arr,
    const void* cb_lut_data,
    int32_t cb_lut_bytes,
    const int32_t* cb_offsets,
    const int32_t* cb_dims,
    const int32_t* cb_signed,
    const int32_t* cb_max_abs,
    const void* sf_lut_data,
    int32_t sf_lut_bytes,
    int32_t B,
    int32_t N,
    int32_t num_sfb,
    int32_t max_frame_bytes,
    uint8_t* output_buf,
    int32_t* frame_sizes
);

// Decode LUT entry: value index + code length
typedef struct {
    int16_t value;
    int16_t length;
} DecodeLUTEntry;

int metal_decode_iso_frames(
    const uint8_t* payloads,
    const int32_t* payload_offsets,
    int32_t B,
    int32_t N,
    int32_t num_sfb,
    const int32_t* sfb_offsets,
    const DecodeLUTEntry* spec_luts,
    const int32_t* lut_offsets,
    const int32_t* lut_max_bits,
    const int32_t* cb_dims,
    const int32_t* cb_signed,
    const int32_t* cb_max_abs,
    const DecodeLUTEntry* sf_lut,
    int32_t sf_max_bits,
    int32_t* quantized_out,
    int32_t* scalefactors_out,
    int32_t* global_gains_out
);

int metal_quantize_iso(
    MetalHuffmanCtx* ctx,
    const float* mdct_coeffs,
    const int32_t* sfb_offsets,
    const int32_t* sfb_map,
    int32_t B, int32_t N, int32_t num_sfb,
    int32_t target_bits, int32_t max_iterations,
    int32_t* quantized_out,
    int32_t* sf_out,
    int32_t* gg_out,
    int32_t* bits_out
);

int metal_parse_adts(
    const uint8_t* data,
    int32_t data_len,
    int32_t* payload_offsets_out,
    int32_t* payload_sizes_out,
    int32_t* sample_rates_out,
    int32_t* channel_configs_out,
    int32_t max_frames,
    int32_t* num_frames_out
);

int metal_huffman_quantize(
    MetalHuffmanCtx* ctx,
    const float* mdct_coeffs,
    const float* masking_thresholds,
    const int32_t* sfb_starts,
    const int32_t* sfb_ends,
    const int32_t* sfb_map,
    int32_t B,
    int32_t N,
    int32_t num_sfb,
    int32_t target_bits,
    int32_t max_iterations,
    int32_t* quantized_out,
    int32_t* scalefactors_out,
    int32_t* global_gains_out,
    int32_t* total_bits_out
);

#ifdef __cplusplus
}
#endif

#endif
