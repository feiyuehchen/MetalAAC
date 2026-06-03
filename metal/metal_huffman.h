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
