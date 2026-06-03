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
