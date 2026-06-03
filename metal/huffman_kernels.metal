#include <metal_stdlib>
using namespace metal;

// ============================================================
// Kernel 5: Full quantization with binary search
// One threadgroup per frame, 1024 threads per TG.
// Each thread handles one coefficient; threads cooperate on
// parallel reduction for bit counting. The binary search loop
// runs entirely inside the kernel — zero host round-trips.
// ============================================================
kernel void kernel_quantize(
    device const float*   mdct_coeffs     [[buffer(0)]],  // (B*N)
    device const int32_t* scalefactors    [[buffer(1)]],  // (B*num_sfb)
    device const int32_t* sfb_map         [[buffer(2)]],  // (N,) coeff→sfb index
    device int32_t*       quantized_out   [[buffer(3)]],  // (B*N)
    device int32_t*       global_gains_out [[buffer(4)]],  // (B,)
    device int32_t*       total_bits_out  [[buffer(5)]],  // (B,)
    constant int32_t&     N              [[buffer(6)]],
    constant int32_t&     num_sfb        [[buffer(7)]],
    constant int32_t&     target_bits    [[buffer(8)]],
    constant int32_t&     max_iterations [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    // Shared state for binary search (only thread 0 writes)
    threadgroup int gain_lo;
    threadgroup int gain_hi;
    threadgroup int best_gain;
    threadgroup int best_bits;
    threadgroup int current_gain;

    // Shared memory for parallel reduction of bit counts
    threadgroup uint bit_sums[1024];

    uint b = gid;
    uint idx = b * (uint)N + tid;

    // Load this thread's coefficient data (persistent across iterations)
    float coeff = (tid < (uint)N) ? mdct_coeffs[idx] : 0.0f;
    float sign_val = (coeff > 0.0f) ? 1.0f : ((coeff < 0.0f) ? -1.0f : 0.0f);
    float abs_coeff = fabs(coeff);
    float powered = pow(abs_coeff + 1e-20f, 0.75f);

    // Per-coefficient scalefactor
    int32_t sf = 0;
    if (tid < (uint)N) {
        int32_t sfb_idx = sfb_map[tid];
        sf = scalefactors[b * (uint)num_sfb + sfb_idx];
    }

    // Initialize search
    if (tid == 0u) {
        gain_lo = 0;
        gain_hi = 255;
        best_gain = 0;
        best_bits = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Binary search: 8 iterations in a tight GPU loop
    for (int iter = 0; iter < max_iterations; iter++) {
        // Thread 0 computes mid-point
        if (tid == 0u) {
            current_gain = (gain_lo + gain_hi) / 2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int gain = current_gain;

        // Each thread quantizes its coefficient
        float gf = pow(2.0f, (float)(gain - sf * 4) / 16.0f);
        int32_t q = (int32_t)(sign_val * floor(powered * gf + 0.4054f));

        // Estimate bits (exp-Golomb)
        int32_t abs_q = (q >= 0) ? q : -q;
        uint code_num = (q > 0) ? (uint)(2 * abs_q - 1) :
                        (q < 0) ? (uint)(2 * abs_q) : 0u;
        uint v = code_num + 1u;
        uint m = (v > 0u) ? (31u - clz(v)) : 0u;
        uint bits = 2u * m + 1u;

        bit_sums[tid] = (tid < (uint)N) ? bits : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel reduction (tree sum, 10 steps for 1024 elements)
        for (uint s = 512u; s > 0u; s >>= 1u) {
            if (tid < s) {
                bit_sums[tid] += bit_sums[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Thread 0 updates search bounds
        if (tid == 0u) {
            int total = (int)bit_sums[0];
            if (total <= target_bits) {
                best_gain = gain;
                best_bits = total;
                gain_lo = gain + 1;
            } else {
                gain_hi = gain - 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final quantization with the optimal gain
    if (tid < (uint)N) {
        int fg = best_gain;
        float gf = pow(2.0f, (float)(fg - sf * 4) / 16.0f);
        quantized_out[idx] = (int32_t)(sign_val * floor(powered * gf + 0.4054f));
    }

    if (tid == 0u) {
        global_gains_out[b] = best_gain;
        total_bits_out[b] = best_bits;
    }
}

// ============================================================
// Kernel 6: Compute scalefactors from masking thresholds
// One threadgroup per frame, num_sfb threads (49)
// ============================================================
kernel void kernel_compute_scalefactors(
    device const float*   mdct_coeffs      [[buffer(0)]],  // (B*N)
    device const float*   masking          [[buffer(1)]],  // (B*num_sfb)
    device const int32_t* sfb_starts       [[buffer(2)]],  // (num_sfb,)
    device const int32_t* sfb_ends         [[buffer(3)]],  // (num_sfb,)
    device int32_t*       scalefactors_out [[buffer(4)]],  // (B*num_sfb)
    constant int32_t&     N               [[buffer(5)]],
    constant int32_t&     num_sfb         [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint b = gid;
    if (tid >= (uint)num_sfb) return;

    uint lo = (uint)sfb_starts[tid];
    uint hi = (uint)sfb_ends[tid];
    uint base = b * (uint)N;

    // Compute band power
    float sum_sq = 0.0f;
    for (uint i = lo; i < hi; i++) {
        float c = mdct_coeffs[base + i];
        sum_sq += c * c;
    }
    float band_power = sum_sq / (float)(hi - lo) + 1e-20f;

    float mask_power = masking[b * (uint)num_sfb + tid];
    if (mask_power < 1e-20f || isnan(mask_power)) mask_power = band_power;

    float smr_db = 10.0f * log10(band_power / mask_power);
    if (isnan(smr_db)) smr_db = 0.0f;

    int sf = (int)clamp(60.0f - smr_db * 0.25f, 0.0f, 60.0f);
    scalefactors_out[b * (uint)num_sfb + tid] = sf;
}

// ============================================================

// Kernel 0: Zero-initialize output buffer
kernel void kernel_zero_output(
    device uint8_t* output_buf [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    output_buf[gid] = 0;
}

// Kernel 1: Compute exp-Golomb codewords and lengths
// Grid: (N, B) — one thread per coefficient per frame
kernel void kernel_compute_codewords(
    device const int32_t* quantized  [[buffer(0)]],
    device uint32_t*      codewords  [[buffer(1)]],
    device uint16_t*      lengths    [[buffer(2)]],
    constant int32_t&     N          [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;  // coefficient index
    uint b = gid.y;  // frame index
    uint idx = b * (uint)N + i;

    int32_t val = quantized[idx];
    int32_t abs_val = (val >= 0) ? val : -val;

    // Signed exp-Golomb: +v->2v-1, -v->2|v|, 0->0
    uint32_t code_num = (val > 0) ? (uint32_t)(2 * abs_val - 1) :
                        (val < 0) ? (uint32_t)(2 * abs_val) : 0u;
    uint32_t cw = code_num + 1;

    // floor(log2(cw)) via count leading zeros
    uint m = (cw > 0) ? (31u - clz(cw)) : 0u;
    uint len = 2u * m + 1u;

    codewords[idx] = cw;
    lengths[idx] = (uint16_t)len;
}

// Kernel 2: Per-frame exclusive prefix sum of codeword lengths (Blelloch scan)
// Grid: B threadgroups × 1024 threads/TG
kernel void kernel_prefix_sum(
    device const uint16_t* lengths          [[buffer(0)]],
    device uint32_t*       bit_offsets      [[buffer(1)]],
    device uint32_t*       frame_total_bits [[buffer(2)]],
    constant int32_t&      N               [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup uint shared_data[1024];

    uint b = gid;
    uint idx = b * (uint)N + tid;
    shared_data[tid] = (tid < (uint)N) ? (uint)lengths[idx] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce)
    for (uint stride = 1; stride < 1024u; stride <<= 1) {
        uint ai = (tid + 1u) * (stride << 1) - 1u;
        if (ai < 1024u) {
            shared_data[ai] += shared_data[ai - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store total and clear last
    if (tid == 0u) {
        frame_total_bits[b] = shared_data[1023];
        shared_data[1023] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint stride = 512u; stride >= 1u; stride >>= 1) {
        uint ai = (tid + 1u) * (stride << 1) - 1u;
        if (ai < 1024u) {
            uint temp = shared_data[ai - stride];
            shared_data[ai - stride] = shared_data[ai];
            shared_data[ai] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < (uint)N) {
        bit_offsets[idx] = shared_data[tid];
    }
}

// Kernel 3: Scatter-write codewords into output buffer
// Uses atomic OR on uint32 words in threadgroup memory to avoid races
// when multiple threads write bits to the same byte.
// Grid: B threadgroups × 1024 threads/TG
kernel void kernel_scatter_write(
    device const uint32_t* codewords        [[buffer(0)]],
    device const uint32_t* bit_offsets      [[buffer(1)]],
    device const uint16_t* lengths_buf      [[buffer(2)]],
    device const int32_t*  scalefactors     [[buffer(3)]],
    device const int32_t*  global_gains     [[buffer(4)]],
    device const uint32_t* frame_total_bits [[buffer(5)]],
    device uint8_t*        output_buf       [[buffer(6)]],
    device int32_t*        frame_sizes      [[buffer(7)]],
    constant int32_t&      N               [[buffer(8)]],
    constant int32_t&      num_sfb         [[buffer(9)]],
    constant int32_t&      max_frame_bytes [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    // Use atomic uint32 words for race-free bit writes
    // 560 words = 2240 bytes, enough for max_frame_bytes
    threadgroup atomic_uint shared_words[560];

    uint b = gid;
    uint total_header_bits = 8u + (uint)num_sfb * 8u;
    uint total_spectral_bits = frame_total_bits[b];
    uint total_bits = total_header_bits + total_spectral_bits;
    uint total_bytes = (total_bits + 7u) / 8u;
    uint total_words = (total_bytes + 3u) / 4u;

    // Zero shared memory cooperatively
    for (uint i = tid; i < 560u; i += 1024u) {
        atomic_store_explicit(&shared_words[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 writes the header (byte-aligned, no conflicts with spectral)
    if (tid == 0u) {
        // Global gain: byte 0
        uint8_t gg = (uint8_t)(global_gains[b] & 0xFF);
        // Pack into word 0, big-endian: byte 0 is bits 31..24 of word 0
        atomic_fetch_or_explicit(&shared_words[0], (uint)gg << 24, memory_order_relaxed);

        // Scalefactors: bytes 1..49
        int32_t prev_sf = 0;
        for (int sb = 0; sb < num_sfb; sb++) {
            int32_t sf = scalefactors[b * num_sfb + sb];
            int32_t diff = sf - prev_sf;
            if (diff < -128) diff = -128;
            if (diff > 127) diff = 127;
            uint8_t byte_val = (uint8_t)(diff & 0xFF);

            uint byte_pos = 1u + (uint)sb;
            uint word_idx = byte_pos / 4u;
            uint byte_in_word = byte_pos % 4u;
            // Big-endian: byte 0 of word is bits 31..24
            uint shift = (3u - byte_in_word) * 8u;
            atomic_fetch_or_explicit(&shared_words[word_idx],
                                     (uint)byte_val << shift,
                                     memory_order_relaxed);
            prev_sf = sf;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread writes its spectral codeword using atomic OR
    if (tid < (uint)N) {
        uint idx = b * (uint)N + tid;
        uint32_t cw = codewords[idx];
        uint16_t len = lengths_buf[idx];
        uint32_t bit_pos = total_header_bits + bit_offsets[idx];

        // Write codeword into big-endian bit stream using atomic OR on uint32 words
        // Bit 0 of the stream = bit 31 of word 0 (MSB-first)
        uint32_t word_idx = bit_pos / 32u;
        uint32_t bit_in_word = bit_pos % 32u;

        if (bit_in_word + (uint)len <= 32u) {
            // Codeword fits in one word
            uint32_t shift = 32u - bit_in_word - (uint)len;
            uint32_t mask = cw << shift;
            atomic_fetch_or_explicit(&shared_words[word_idx], mask,
                                     memory_order_relaxed);
        } else {
            // Codeword spans two words
            uint32_t bits_in_first = 32u - bit_in_word;
            uint32_t mask1 = cw >> ((uint)len - bits_in_first);
            atomic_fetch_or_explicit(&shared_words[word_idx], mask1,
                                     memory_order_relaxed);

            uint32_t remaining = (uint)len - bits_in_first;
            uint32_t mask2 = (cw & ((1u << remaining) - 1u)) << (32u - remaining);
            atomic_fetch_or_explicit(&shared_words[word_idx + 1u], mask2,
                                     memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative copy: convert uint32 words (big-endian) to bytes in device memory
    device uint8_t* frame_out = output_buf + b * (uint)max_frame_bytes;
    for (uint i = tid; i < total_words; i += 1024u) {
        uint32_t word = atomic_load_explicit(&shared_words[i], memory_order_relaxed);
        uint base = i * 4u;
        if (base < total_bytes) frame_out[base]     = (uint8_t)(word >> 24);
        if (base + 1u < total_bytes) frame_out[base + 1u] = (uint8_t)(word >> 16);
        if (base + 2u < total_bytes) frame_out[base + 2u] = (uint8_t)(word >> 8);
        if (base + 3u < total_bytes) frame_out[base + 3u] = (uint8_t)(word);
    }

    if (tid == 0u) {
        frame_sizes[b] = (int32_t)total_bytes;
    }
}

// Kernel 4: Decode frames (one thread per frame)
// Grid: B threads
kernel void kernel_decode_frames(
    device const uint8_t*  compressed       [[buffer(0)]],
    device const int32_t*  frame_offsets    [[buffer(1)]],
    device int32_t*        quantized_out    [[buffer(2)]],
    device int32_t*        scalefactors_out [[buffer(3)]],
    device int32_t*        global_gains_out [[buffer(4)]],
    constant int32_t&      N               [[buffer(5)]],
    constant int32_t&      num_sfb         [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint b = gid;
    uint frame_start = (uint)frame_offsets[b];
    device const uint8_t* data = compressed + frame_start;

    uint bit_pos = 0;

    // Read global_gain (8 bits)
    uint32_t gg = 0;
    for (uint i = 0; i < 8u; i++) {
        uint byte_idx = (bit_pos + i) / 8u;
        uint bit_in_byte = 7u - ((bit_pos + i) % 8u);
        gg = (gg << 1) | ((data[byte_idx] >> bit_in_byte) & 1u);
    }
    bit_pos += 8;
    global_gains_out[b] = (int32_t)gg;

    // Read scalefactors (differential, 8 bits each)
    int32_t prev_sf = 0;
    for (int sb = 0; sb < num_sfb; sb++) {
        uint32_t byte_val = 0;
        for (uint i = 0; i < 8u; i++) {
            uint bp = bit_pos + i;
            uint byte_idx = bp / 8u;
            uint bit_in_byte = 7u - (bp % 8u);
            byte_val = (byte_val << 1) | ((data[byte_idx] >> bit_in_byte) & 1u);
        }
        bit_pos += 8;
        int32_t diff = (int32_t)byte_val;
        if (diff > 127) diff -= 256;
        prev_sf += diff;
        scalefactors_out[b * num_sfb + sb] = prev_sf;
    }

    // Decode 1024 exp-Golomb codewords
    for (int i = 0; i < N; i++) {
        // Count leading zeros
        uint m = 0;
        while (true) {
            uint byte_idx = bit_pos / 8u;
            uint bit_in_byte = 7u - (bit_pos % 8u);
            uint bit = (data[byte_idx] >> bit_in_byte) & 1u;
            bit_pos++;
            if (bit == 1u) break;
            m++;
        }

        // Read m-bit remainder
        uint32_t remainder = 0;
        for (uint k = 0; k < m; k++) {
            uint byte_idx = bit_pos / 8u;
            uint bit_in_byte = 7u - (bit_pos % 8u);
            remainder = (remainder << 1) | ((data[byte_idx] >> bit_in_byte) & 1u);
            bit_pos++;
        }

        int32_t code_num = (int32_t)((1u << m) + remainder) - 1;

        // Inverse signed mapping: 0->0, odd->+, even->-
        int32_t value;
        if (code_num == 0) {
            value = 0;
        } else if (code_num % 2 == 1) {
            value = (code_num + 1) / 2;
        } else {
            value = -(code_num / 2);
        }

        quantized_out[b * N + i] = value;
    }
}
