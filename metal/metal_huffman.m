#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <string.h>
#include "metal_huffman.h"

#define TILE_SIZE 16384

// Store ObjC objects as void* with CFBridgingRetain/Release for C struct compat
struct MetalHuffmanCtx {
    void* device;       // id<MTLDevice>
    void* queue;        // id<MTLCommandQueue>
    void* pso_zero;     // id<MTLComputePipelineState>
    void* pso_codewords;
    void* pso_prefix_sum;
    void* pso_scatter;
    void* pso_decode;
    void* pso_quantize;
    void* pso_compute_sf;
    void* pso_encode_rdb;
};

#define DEV(ctx) ((__bridge id<MTLDevice>)(ctx)->device)
#define QUEUE(ctx) ((__bridge id<MTLCommandQueue>)(ctx)->queue)
#define PSO(ptr) ((__bridge id<MTLComputePipelineState>)(ptr))

static void* _retain(id obj) {
    return (void*)CFBridgingRetain(obj);
}

static id<MTLComputePipelineState> _make_pso(id<MTLDevice> device,
                                              id<MTLLibrary> library,
                                              const char* name) {
    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:
                          [NSString stringWithUTF8String:name]];
    if (!fn) {
        NSLog(@"Metal: function '%s' not found", name);
        return nil;
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
        NSLog(@"Metal: PSO creation failed for '%s': %@", name, error);
    }
    return pso;
}

MetalHuffmanCtx* metal_huffman_create(void) {
    @autoreleasepool {
        MetalHuffmanCtx* ctx = (MetalHuffmanCtx*)calloc(1, sizeof(MetalHuffmanCtx));
        if (!ctx) return NULL;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { free(ctx); return NULL; }
        ctx->device = _retain(device);
        ctx->queue = _retain([device newCommandQueue]);

        // Find the .metal source file relative to the dylib
        NSString* metalPath = nil;
        Dl_info dl_info;
        if (dladdr((const void*)metal_huffman_create, &dl_info)) {
            NSString* dylibPath = [NSString stringWithUTF8String:dl_info.dli_fname];
            NSString* dir = [dylibPath stringByDeletingLastPathComponent];
            metalPath = [dir stringByAppendingPathComponent:@"huffman_kernels.metal"];
        }

        NSError* error = nil;
        id<MTLLibrary> library = nil;

        // Try .metallib first
        if (metalPath) {
            NSString* metallibPath = [[metalPath stringByDeletingPathExtension]
                                      stringByAppendingPathExtension:@"metallib"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:metallibPath]) {
                NSURL* url = [NSURL fileURLWithPath:metallibPath];
                library = [device newLibraryWithURL:url error:&error];
            }
        }

        // Fall back to runtime compilation
        if (!library && metalPath) {
            NSString* source = [NSString stringWithContentsOfFile:metalPath
                                encoding:NSUTF8StringEncoding error:&error];
            if (source) {
                MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
                opts.fastMathEnabled = YES;
                library = [device newLibraryWithSource:source
                                              options:opts error:&error];
                if (!library) {
                    NSLog(@"Metal: shader compile failed: %@", error);
                }
            } else {
                NSLog(@"Metal: cannot read %@: %@", metalPath, error);
            }
        }

        if (!library) {
            NSLog(@"Metal: no shader library");
            CFBridgingRelease(ctx->device);
            CFBridgingRelease(ctx->queue);
            free(ctx);
            return NULL;
        }

        id<MTLComputePipelineState> p;
        p = _make_pso(device, library, "kernel_zero_output");
        if (!p) goto fail;
        ctx->pso_zero = _retain(p);

        p = _make_pso(device, library, "kernel_compute_codewords");
        if (!p) goto fail;
        ctx->pso_codewords = _retain(p);

        p = _make_pso(device, library, "kernel_prefix_sum");
        if (!p) goto fail;
        ctx->pso_prefix_sum = _retain(p);

        p = _make_pso(device, library, "kernel_scatter_write");
        if (!p) goto fail;
        ctx->pso_scatter = _retain(p);

        p = _make_pso(device, library, "kernel_decode_frames");
        if (!p) goto fail;
        ctx->pso_decode = _retain(p);

        p = _make_pso(device, library, "kernel_quantize");
        if (!p) goto fail;
        ctx->pso_quantize = _retain(p);

        p = _make_pso(device, library, "kernel_compute_scalefactors");
        if (!p) goto fail;
        ctx->pso_compute_sf = _retain(p);

        p = _make_pso(device, library, "kernel_encode_raw_data_block");
        if (!p) goto fail;
        ctx->pso_encode_rdb = _retain(p);

        return ctx;

    fail:
        metal_huffman_destroy(ctx);
        return NULL;
    }
}

void metal_huffman_destroy(MetalHuffmanCtx* ctx) {
    if (!ctx) return;
    if (ctx->device)       CFBridgingRelease(ctx->device);
    if (ctx->queue)        CFBridgingRelease(ctx->queue);
    if (ctx->pso_zero)     CFBridgingRelease(ctx->pso_zero);
    if (ctx->pso_codewords) CFBridgingRelease(ctx->pso_codewords);
    if (ctx->pso_prefix_sum) CFBridgingRelease(ctx->pso_prefix_sum);
    if (ctx->pso_scatter)  CFBridgingRelease(ctx->pso_scatter);
    if (ctx->pso_decode)   CFBridgingRelease(ctx->pso_decode);
    if (ctx->pso_quantize) CFBridgingRelease(ctx->pso_quantize);
    if (ctx->pso_compute_sf) CFBridgingRelease(ctx->pso_compute_sf);
    if (ctx->pso_encode_rdb) CFBridgingRelease(ctx->pso_encode_rdb);
    free(ctx);
}

int metal_huffman_max_frame_bytes(int32_t N, int32_t num_sfb) {
    return (1 + num_sfb) + N * 4;
}

static int _encode_tile(MetalHuffmanCtx* ctx,
                         const int32_t* quantized,
                         const int32_t* scalefactors,
                         const int32_t* global_gains,
                         int32_t B, int32_t N, int32_t num_sfb,
                         uint8_t* output_buf,
                         int32_t max_frame_bytes,
                         int32_t* frame_sizes) {
    @autoreleasepool {
        id<MTLDevice> dev = DEV(ctx);
        size_t bn = (size_t)B * N;

        id<MTLBuffer> buf_q  = [dev newBufferWithBytes:quantized
                               length:bn * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_cw = [dev newBufferWithLength:bn * sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ln = [dev newBufferWithLength:bn * sizeof(uint16_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_off = [dev newBufferWithLength:bn * sizeof(uint32_t)
                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_tb = [dev newBufferWithLength:B * sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];

        size_t out_size = (size_t)B * max_frame_bytes;
        id<MTLBuffer> buf_out = [dev newBufferWithLength:out_size
                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sz = [dev newBufferWithLength:B * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> buf_sf = [dev newBufferWithBytes:scalefactors
                               length:(size_t)B * num_sfb * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_gg = [dev newBufferWithBytes:global_gains
                               length:B * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_N  = [dev newBufferWithBytes:&N length:4
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ns = [dev newBufferWithBytes:&num_sfb length:4
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_mf = [dev newBufferWithBytes:&max_frame_bytes length:4
                               options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [QUEUE(ctx) commandBuffer];

        // K0: zero output
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:PSO(ctx->pso_zero)];
            [e setBuffer:buf_out offset:0 atIndex:0];
            NSUInteger sz = out_size;
            [e dispatchThreads:MTLSizeMake(sz, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN(256u, sz), 1, 1)];
            [e endEncoding];
        }
        // K1: compute codewords
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:PSO(ctx->pso_codewords)];
            [e setBuffer:buf_q  offset:0 atIndex:0];
            [e setBuffer:buf_cw offset:0 atIndex:1];
            [e setBuffer:buf_ln offset:0 atIndex:2];
            [e setBuffer:buf_N  offset:0 atIndex:3];
            [e dispatchThreads:MTLSizeMake(N, B, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN(256, (int)N), 1, 1)];
            [e endEncoding];
        }
        // K2: prefix sum
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:PSO(ctx->pso_prefix_sum)];
            [e setBuffer:buf_ln  offset:0 atIndex:0];
            [e setBuffer:buf_off offset:0 atIndex:1];
            [e setBuffer:buf_tb  offset:0 atIndex:2];
            [e setBuffer:buf_N   offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
            [e endEncoding];
        }
        // K3: scatter write
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:PSO(ctx->pso_scatter)];
            [e setBuffer:buf_cw  offset:0 atIndex:0];
            [e setBuffer:buf_off offset:0 atIndex:1];
            [e setBuffer:buf_ln  offset:0 atIndex:2];
            [e setBuffer:buf_sf  offset:0 atIndex:3];
            [e setBuffer:buf_gg  offset:0 atIndex:4];
            [e setBuffer:buf_tb  offset:0 atIndex:5];
            [e setBuffer:buf_out offset:0 atIndex:6];
            [e setBuffer:buf_sz  offset:0 atIndex:7];
            [e setBuffer:buf_N   offset:0 atIndex:8];
            [e setBuffer:buf_ns  offset:0 atIndex:9];
            [e setBuffer:buf_mf  offset:0 atIndex:10];
            [e dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
            [e endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            NSLog(@"Metal encode: %@", cmd.error);
            return -1;
        }

        memcpy(output_buf, buf_out.contents, out_size);
        memcpy(frame_sizes, buf_sz.contents, B * sizeof(int32_t));
        return 0;
    }
}

int metal_huffman_encode(MetalHuffmanCtx* ctx,
                          const int32_t* quantized,
                          const int32_t* scalefactors,
                          const int32_t* global_gains,
                          int32_t B, int32_t N, int32_t num_sfb,
                          uint8_t* output_buf,
                          int32_t max_frame_bytes,
                          int32_t* frame_sizes) {
    for (int32_t s = 0; s < B; s += TILE_SIZE) {
        int32_t tB = (s + TILE_SIZE <= B) ? TILE_SIZE : (B - s);
        int rc = _encode_tile(ctx,
                              quantized + (size_t)s * N,
                              scalefactors + (size_t)s * num_sfb,
                              global_gains + s,
                              tB, N, num_sfb,
                              output_buf + (size_t)s * max_frame_bytes,
                              max_frame_bytes,
                              frame_sizes + s);
        if (rc != 0) return rc;
    }
    return 0;
}

static int _decode_tile(MetalHuffmanCtx* ctx,
                         const uint8_t* compressed,
                         const int32_t* frame_offsets,
                         int32_t B, int32_t N, int32_t num_sfb,
                         int32_t* quantized_out,
                         int32_t* scalefactors_out,
                         int32_t* global_gains_out) {
    @autoreleasepool {
        id<MTLDevice> dev = DEV(ctx);
        int32_t total_compressed = frame_offsets[B];

        id<MTLBuffer> buf_c  = [dev newBufferWithBytes:compressed
                               length:total_compressed
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_fo = [dev newBufferWithBytes:frame_offsets
                               length:(B + 1) * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];

        size_t bn = (size_t)B * N;
        id<MTLBuffer> buf_q  = [dev newBufferWithLength:bn * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sf = [dev newBufferWithLength:(size_t)B * num_sfb * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_gg = [dev newBufferWithLength:B * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_N  = [dev newBufferWithBytes:&N length:4
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ns = [dev newBufferWithBytes:&num_sfb length:4
                               options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [QUEUE(ctx) commandBuffer];
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:PSO(ctx->pso_decode)];
            [e setBuffer:buf_c  offset:0 atIndex:0];
            [e setBuffer:buf_fo offset:0 atIndex:1];
            [e setBuffer:buf_q  offset:0 atIndex:2];
            [e setBuffer:buf_sf offset:0 atIndex:3];
            [e setBuffer:buf_gg offset:0 atIndex:4];
            [e setBuffer:buf_N  offset:0 atIndex:5];
            [e setBuffer:buf_ns offset:0 atIndex:6];
            [e dispatchThreads:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN(256, B), 1, 1)];
            [e endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            NSLog(@"Metal decode: %@", cmd.error);
            return -1;
        }

        memcpy(quantized_out, buf_q.contents, bn * sizeof(int32_t));
        memcpy(scalefactors_out, buf_sf.contents, (size_t)B * num_sfb * sizeof(int32_t));
        memcpy(global_gains_out, buf_gg.contents, B * sizeof(int32_t));
        return 0;
    }
}

int metal_huffman_decode(MetalHuffmanCtx* ctx,
                          const uint8_t* compressed,
                          const int32_t* frame_offsets,
                          int32_t B, int32_t N, int32_t num_sfb,
                          int32_t* quantized_out,
                          int32_t* scalefactors_out,
                          int32_t* global_gains_out) {
    for (int32_t s = 0; s < B; s += TILE_SIZE) {
        int32_t tB = (s + TILE_SIZE <= B) ? TILE_SIZE : (B - s);
        int rc = _decode_tile(ctx,
                              compressed,
                              frame_offsets + s,
                              tB, N, num_sfb,
                              quantized_out + (size_t)s * N,
                              scalefactors_out + (size_t)s * num_sfb,
                              global_gains_out + s);
        if (rc != 0) return rc;
    }
    return 0;
}

// ---- Metal quantization ----

static int _quantize_tile(MetalHuffmanCtx* ctx,
                           const float* mdct_coeffs,
                           const float* masking,
                           const int32_t* sfb_starts,
                           const int32_t* sfb_ends,
                           const int32_t* sfb_map,
                           int32_t B, int32_t N, int32_t num_sfb,
                           int32_t target_bits, int32_t max_iterations,
                           int32_t* quantized_out,
                           int32_t* scalefactors_out,
                           int32_t* global_gains_out,
                           int32_t* total_bits_out) {
    @autoreleasepool {
        id<MTLDevice> dev = DEV(ctx);
        size_t bn = (size_t)B * N;
        size_t bs = (size_t)B * num_sfb;

        id<MTLBuffer> buf_mc = [dev newBufferWithBytes:mdct_coeffs
                               length:bn * sizeof(float)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_mk = [dev newBufferWithBytes:masking
                               length:bs * sizeof(float)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ss = [dev newBufferWithBytes:sfb_starts
                               length:num_sfb * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_se = [dev newBufferWithBytes:sfb_ends
                               length:num_sfb * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sm = [dev newBufferWithBytes:sfb_map
                               length:N * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sf = [dev newBufferWithLength:bs * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_qo = [dev newBufferWithLength:bn * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_gg = [dev newBufferWithLength:B * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_tb = [dev newBufferWithLength:B * sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_N  = [dev newBufferWithBytes:&N length:4
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ns = [dev newBufferWithBytes:&num_sfb length:4
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_tg = [dev newBufferWithBytes:&target_bits length:4
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_mi = [dev newBufferWithBytes:&max_iterations length:4
                               options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [QUEUE(ctx) commandBuffer];

        // Kernel: compute scalefactors
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:PSO(ctx->pso_compute_sf)];
            [e setBuffer:buf_mc offset:0 atIndex:0];
            [e setBuffer:buf_mk offset:0 atIndex:1];
            [e setBuffer:buf_ss offset:0 atIndex:2];
            [e setBuffer:buf_se offset:0 atIndex:3];
            [e setBuffer:buf_sf offset:0 atIndex:4];
            [e setBuffer:buf_N  offset:0 atIndex:5];
            [e setBuffer:buf_ns offset:0 atIndex:6];
            // num_sfb threads per frame, B frames
            [e dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN(64, (NSUInteger)num_sfb), 1, 1)];
            [e endEncoding];
        }

        // Kernel: quantize (binary search)
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:PSO(ctx->pso_quantize)];
            [e setBuffer:buf_mc offset:0 atIndex:0];
            [e setBuffer:buf_sf offset:0 atIndex:1];
            [e setBuffer:buf_sm offset:0 atIndex:2];
            [e setBuffer:buf_qo offset:0 atIndex:3];
            [e setBuffer:buf_gg offset:0 atIndex:4];
            [e setBuffer:buf_tb offset:0 atIndex:5];
            [e setBuffer:buf_N  offset:0 atIndex:6];
            [e setBuffer:buf_ns offset:0 atIndex:7];
            [e setBuffer:buf_tg offset:0 atIndex:8];
            [e setBuffer:buf_mi offset:0 atIndex:9];
            [e dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
            [e endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            NSLog(@"Metal quantize: %@", cmd.error);
            return -1;
        }

        memcpy(quantized_out, buf_qo.contents, bn * sizeof(int32_t));
        memcpy(scalefactors_out, buf_sf.contents, bs * sizeof(int32_t));
        memcpy(global_gains_out, buf_gg.contents, B * sizeof(int32_t));
        memcpy(total_bits_out, buf_tb.contents, B * sizeof(int32_t));
        return 0;
    }
}

int metal_huffman_quantize(MetalHuffmanCtx* ctx,
                            const float* mdct_coeffs,
                            const float* masking,
                            const int32_t* sfb_starts,
                            const int32_t* sfb_ends,
                            const int32_t* sfb_map,
                            int32_t B, int32_t N, int32_t num_sfb,
                            int32_t target_bits, int32_t max_iterations,
                            int32_t* quantized_out,
                            int32_t* scalefactors_out,
                            int32_t* global_gains_out,
                            int32_t* total_bits_out) {
    for (int32_t s = 0; s < B; s += TILE_SIZE) {
        int32_t tB = (s + TILE_SIZE <= B) ? TILE_SIZE : (B - s);
        int rc = _quantize_tile(ctx,
                                mdct_coeffs + (size_t)s * N,
                                masking + (size_t)s * num_sfb,
                                sfb_starts, sfb_ends, sfb_map,
                                tB, N, num_sfb,
                                target_bits, max_iterations,
                                quantized_out + (size_t)s * N,
                                scalefactors_out + (size_t)s * num_sfb,
                                global_gains_out + s,
                                total_bits_out + s);
        if (rc != 0) return rc;
    }
    return 0;
}

// ---- Metal ISO raw_data_block encoding ----

int metal_encode_adts_frames(MetalHuffmanCtx* ctx,
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
                              int32_t B, int32_t N, int32_t num_sfb,
                              int32_t max_frame_bytes,
                              uint8_t* output_buf,
                              int32_t* frame_sizes) {
    @autoreleasepool {
        id<MTLDevice> dev = DEV(ctx);

        // Create buffers
        id<MTLBuffer> buf_q  = [dev newBufferWithBytes:quantized length:(size_t)B*N*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sf = [dev newBufferWithBytes:scalefactors length:(size_t)B*num_sfb*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_gg = [dev newBufferWithBytes:global_gains length:B*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ws = [dev newBufferWithBytes:window_seqs length:B*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sfb = [dev newBufferWithBytes:sfb_offsets_arr length:(num_sfb+1)*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_cblut = [dev newBufferWithBytes:cb_lut_data length:cb_lut_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_cboff = [dev newBufferWithBytes:cb_offsets length:12*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_cbdim = [dev newBufferWithBytes:cb_dims length:12*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_cbsig = [dev newBufferWithBytes:cb_signed length:12*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_cbmab = [dev newBufferWithBytes:cb_max_abs length:12*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sflut = [dev newBufferWithBytes:sf_lut_data length:sf_lut_bytes options:MTLResourceStorageModeShared];

        size_t out_size = (size_t)B * max_frame_bytes;
        id<MTLBuffer> buf_out = [dev newBufferWithLength:out_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sz = [dev newBufferWithLength:B*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_N = [dev newBufferWithBytes:&N length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ns = [dev newBufferWithBytes:&num_sfb length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_mfb = [dev newBufferWithBytes:&max_frame_bytes length:4 options:MTLResourceStorageModeShared];

        // Zero output
        memset(buf_out.contents, 0, out_size);

        id<MTLCommandBuffer> cmd = [QUEUE(ctx) commandBuffer];
        {
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:PSO(ctx->pso_encode_rdb)];
            [e setBuffer:buf_q    offset:0 atIndex:0];
            [e setBuffer:buf_sf   offset:0 atIndex:1];
            [e setBuffer:buf_gg   offset:0 atIndex:2];
            [e setBuffer:buf_ws   offset:0 atIndex:3];
            [e setBuffer:buf_sfb  offset:0 atIndex:4];
            [e setBuffer:buf_cblut offset:0 atIndex:5];
            [e setBuffer:buf_cboff offset:0 atIndex:6];
            [e setBuffer:buf_cbdim offset:0 atIndex:7];
            [e setBuffer:buf_cbsig offset:0 atIndex:8];
            [e setBuffer:buf_cbmab offset:0 atIndex:9];
            [e setBuffer:buf_sflut offset:0 atIndex:10];
            [e setBuffer:buf_out  offset:0 atIndex:11];
            [e setBuffer:buf_sz   offset:0 atIndex:12];
            [e setBuffer:buf_N    offset:0 atIndex:13];
            [e setBuffer:buf_ns   offset:0 atIndex:14];
            [e setBuffer:buf_mfb  offset:0 atIndex:15];
            [e dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
            [e endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            NSLog(@"Metal encode_rdb: %@", cmd.error);
            return -1;
        }

        memcpy(output_buf, buf_out.contents, out_size);
        memcpy(frame_sizes, buf_sz.contents, B * 4);
        return 0;
    }
}

// ---- Native ISO Huffman decoder (C with GCD parallelism) ----

static inline uint32_t bits_peek(const uint8_t* data, int pos, int n, int total_bits) {
    if (n <= 0) return 0;
    int avail = total_bits - pos;
    if (avail <= 0) return 0;
    if (n > avail) n = avail;
    uint32_t result = 0;
    for (int i = 0; i < n; i++) {
        int bp = pos + i;
        result = (result << 1) | ((data[bp >> 3] >> (7 - (bp & 7))) & 1);
    }
    if (n < 32) {
        // nothing to pad
    }
    return result;
}

static inline int bits_read(const uint8_t* data, int* pos, int n, int total_bits) {
    uint32_t val = bits_peek(data, *pos, n, total_bits);
    *pos += n;
    return (int)val;
}

static inline int huff_decode(const uint8_t* data, int* pos, int total_bits,
                               const DecodeLUTEntry* lut, int max_bits) {
    int avail = total_bits - *pos;
    if (avail <= 0) return -1;
    int peek_bits = (avail < max_bits) ? avail : max_bits;
    uint32_t prefix = bits_peek(data, *pos, peek_bits, total_bits);
    if (peek_bits < max_bits) prefix <<= (max_bits - peek_bits);
    DecodeLUTEntry e = lut[prefix];
    if (e.value < 0) return -1;
    *pos += e.length;
    return e.value;
}

static void decode_one_frame(
    const uint8_t* payload, int payload_bytes,
    int N, int num_sfb, const int32_t* sfb_offsets,
    const DecodeLUTEntry* spec_luts, const int32_t* lut_offsets,
    const int32_t* lut_max_bits, const int32_t* cb_dims,
    const int32_t* cb_signed, const int32_t* cb_max_abs,
    const DecodeLUTEntry* sf_lut, int sf_max_bits,
    int32_t* q_out, int32_t* sf_out, int32_t* gg_out)
{
    int total_bits = payload_bytes * 8;
    int pos = 0;

    int elem_id = bits_read(payload, &pos, 3, total_bits);
    bits_read(payload, &pos, 4, total_bits); // tag

    int global_gain = 0;
    int window_seq = 0;
    int max_sfb = 0;
    int sect_esc = 31, sect_nbits = 5;

    if (elem_id == 1) {
        // CPE: common_window + ics_info + ms_mask
        int cw = bits_read(payload, &pos, 1, total_bits);
        if (cw) {
            bits_read(payload, &pos, 1, total_bits); // reserved
            window_seq = bits_read(payload, &pos, 2, total_bits);
            bits_read(payload, &pos, 1, total_bits); // shape
            if (window_seq == 2) {
                max_sfb = bits_read(payload, &pos, 4, total_bits);
                bits_read(payload, &pos, 7, total_bits);
                sect_esc = 7; sect_nbits = 3;
            } else {
                max_sfb = bits_read(payload, &pos, 6, total_bits);
                bits_read(payload, &pos, 1, total_bits);
            }
            int ms_mask = bits_read(payload, &pos, 2, total_bits);
            if (ms_mask == 1) pos += (max_sfb < num_sfb ? max_sfb : num_sfb);
        }
        if (max_sfb > num_sfb) max_sfb = num_sfb;
        // Decode channel 0
        global_gain = bits_read(payload, &pos, 8, total_bits);
    } else {
        // SCE
        global_gain = bits_read(payload, &pos, 8, total_bits);
        bits_read(payload, &pos, 1, total_bits);
        window_seq = bits_read(payload, &pos, 2, total_bits);
        bits_read(payload, &pos, 1, total_bits);
        if (window_seq == 2) {
            max_sfb = bits_read(payload, &pos, 4, total_bits);
            bits_read(payload, &pos, 7, total_bits);
            sect_esc = 7; sect_nbits = 3;
        } else {
            max_sfb = bits_read(payload, &pos, 6, total_bits);
            bits_read(payload, &pos, 1, total_bits);
        }
        if (max_sfb > num_sfb) max_sfb = num_sfb;
    }

    *gg_out = global_gain;

    // section_data
    int sfb_cb[64];
    memset(sfb_cb, 0, sizeof(sfb_cb));
    int sections[64][3]; // start, end, cb
    int n_sections = 0;
    int k = 0;
    while (k < max_sfb && n_sections < 64) {
        int cb = bits_read(payload, &pos, 4, total_bits);
        int slen = 0;
        while (1) {
            int inc = bits_read(payload, &pos, sect_nbits, total_bits);
            slen += inc;
            if (inc < sect_esc) break;
        }
        int end = k + slen;
        if (end > max_sfb) end = max_sfb;
        sections[n_sections][0] = k;
        sections[n_sections][1] = end;
        sections[n_sections][2] = cb;
        n_sections++;
        for (int s = k; s < end && s < 64; s++) sfb_cb[s] = cb;
        k = end;
    }

    // scale_factor_data
    int prev_sf = global_gain;
    for (int sb = 0; sb < max_sfb; sb++) {
        if (sfb_cb[sb] == 0) { sf_out[sb] = prev_sf; continue; }
        int idx = huff_decode(payload, &pos, total_bits, sf_lut, sf_max_bits);
        if (idx < 0) { sf_out[sb] = prev_sf; continue; }
        prev_sf += idx - 60;
        sf_out[sb] = prev_sf;
    }
    for (int sb = max_sfb; sb < num_sfb; sb++) sf_out[sb] = prev_sf;

    // pulse/tns/gain
    pos += 3;

    // spectral_data
    memset(q_out, 0, N * sizeof(int32_t));
    for (int si = 0; si < n_sections; si++) {
        int cb = sections[si][2];
        if (cb == 0 || cb < 1 || cb > 11) continue;
        int dim = cb_dims[cb];
        int is_signed = cb_signed[cb];
        int mabs = cb_max_abs[cb];
        int dim_size = is_signed ? (2*mabs+1) : (mabs+1);
        int offset = is_signed ? mabs : 0;
        const DecodeLUTEntry* lut = spec_luts + lut_offsets[cb];
        int mbits = lut_max_bits[cb];
        int lo = sfb_offsets[sections[si][0]];
        int hi = sfb_offsets[sections[si][1]];
        if (hi > N) hi = N;

        for (int i = lo; i < hi; i += dim) {
            int idx = huff_decode(payload, &pos, total_bits, lut, mbits);
            if (idx < 0) break;

            int vals[4];
            int rem = idx;
            for (int d = dim - 1; d >= 0; d--) {
                vals[d] = (rem % dim_size) - offset;
                rem /= dim_size;
            }

            if (!is_signed) {
                for (int d = 0; d < dim; d++) {
                    if (vals[d] > 0) {
                        if (bits_read(payload, &pos, 1, total_bits)) vals[d] = -vals[d];
                    }
                }
                if (cb == 11) {
                    for (int d = 0; d < dim; d++) {
                        int av = vals[d] < 0 ? -vals[d] : vals[d];
                        if (av >= 16) {
                            int sign = vals[d] < 0 ? -1 : 1;
                            int cnt = 0;
                            while (bits_read(payload, &pos, 1, total_bits) == 1) cnt++;
                            int esc_val = (1 << (cnt + 4)) | bits_read(payload, &pos, cnt + 4, total_bits);
                            vals[d] = sign * esc_val;
                        }
                    }
                }
            }

            for (int d = 0; d < dim && (i+d) < N; d++) q_out[i+d] = vals[d];
        }
    }
}

int metal_decode_iso_frames(
    const uint8_t* payloads,
    const int32_t* payload_offsets,
    int32_t B, int32_t N, int32_t num_sfb,
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
    int32_t* global_gains_out)
{
    dispatch_apply((size_t)B, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t b) {
            int offset = payload_offsets[b];
            int len = payload_offsets[b+1] - offset;
            decode_one_frame(
                payloads + offset, len,
                N, num_sfb, sfb_offsets,
                spec_luts, lut_offsets, lut_max_bits,
                cb_dims, cb_signed, cb_max_abs,
                sf_lut, sf_max_bits,
                quantized_out + b * N,
                scalefactors_out + b * num_sfb,
                global_gains_out + b);
        });
    return 0;
}
