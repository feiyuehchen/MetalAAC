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

// ============================================================
// ISO AAC-LC raw_data_block encoding kernel
//
// Writes a complete SCE (single_channel_element) per frame:
//   ID_SCE + tag + global_gain + ics_info + section_data +
//   scale_factor_data + pulse/tns/gain + spectral_data + ID_END
//
// One threadgroup per frame, 1024 threads per TG.
// Thread 0 handles sequential header; all threads do spectral.
// ============================================================

struct HuffEntry {
    uint32_t code;
    uint8_t  bits;
    uint8_t  pad[3];
};

// Helper: write `len` bits of `cw` at `bit_pos` into atomic uint32 buffer
inline void write_bits(
    threadgroup atomic_uint* buf,
    uint32_t bit_pos,
    uint32_t cw,
    uint16_t len)
{
    if (len == 0) return;
    uint32_t word_idx = bit_pos / 32u;
    uint32_t bit_in_word = bit_pos % 32u;

    if (bit_in_word + (uint)len <= 32u) {
        uint32_t shift = 32u - bit_in_word - (uint)len;
        atomic_fetch_or_explicit(&buf[word_idx], cw << shift, memory_order_relaxed);
    } else {
        uint32_t bits_first = 32u - bit_in_word;
        atomic_fetch_or_explicit(&buf[word_idx], cw >> ((uint)len - bits_first), memory_order_relaxed);
        uint32_t remaining = (uint)len - bits_first;
        uint32_t mask2 = (cw & ((1u << remaining) - 1u)) << (32u - remaining);
        atomic_fetch_or_explicit(&buf[word_idx + 1u], mask2, memory_order_relaxed);
    }
}

// Helper: write a fixed-width field using thread 0 (non-atomic, sequential)
inline void write_bits_seq(
    threadgroup uint32_t* buf,
    uint32_t bit_pos,
    uint32_t cw,
    uint16_t len)
{
    for (uint k = 0; k < (uint)len; k++) {
        uint gbit = bit_pos + k;
        uint widx = gbit / 32u;
        uint binw = 31u - (gbit % 32u);
        uint bval = (cw >> ((uint)len - 1u - k)) & 1u;
        buf[widx] |= (bval << binw);
    }
}

kernel void kernel_encode_raw_data_block(
    device const int32_t*  quantized        [[buffer(0)]],   // (B*N)
    device const int32_t*  scalefactors     [[buffer(1)]],   // (B*num_sfb)
    device const int32_t*  global_gains     [[buffer(2)]],   // (B,)
    device const int32_t*  window_seqs      [[buffer(3)]],   // (B,)
    device const int32_t*  sfb_offsets_buf  [[buffer(4)]],   // (num_sfb+1,)
    constant HuffEntry*    cb_luts          [[buffer(5)]],   // all codebook entries, flattened
    constant int32_t*      cb_offsets       [[buffer(6)]],   // offset into cb_luts per codebook (12 entries)
    constant int32_t*      cb_dims          [[buffer(7)]],   // dimension per codebook (12 entries: 0,4,4,4,4,2,2,2,2,2,2,2)
    constant int32_t*      cb_signed        [[buffer(8)]],   // signed flag per codebook
    constant int32_t*      cb_max_abs       [[buffer(9)]],   // max abs per codebook
    constant HuffEntry*    sf_lut           [[buffer(10)]],  // SF codebook (121 entries)
    device uint8_t*        output_buf       [[buffer(11)]],  // (B*max_frame_bytes)
    device int32_t*        frame_sizes      [[buffer(12)]],  // (B,)
    constant int32_t&      N               [[buffer(13)]],
    constant int32_t&      num_sfb         [[buffer(14)]],
    constant int32_t&      max_frame_bytes [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    // 560 uint32 words = 2240 bytes staging buffer
    threadgroup atomic_uint shared_words[560];
    // Per-SFB codebook selection (computed by thread 0, read by all)
    threadgroup int32_t sfb_cb[64];  // max 64 SFBs
    threadgroup uint32_t header_bits;  // total header bit count

    uint b = gid;

    // Zero shared buffer
    for (uint i = tid; i < 560u; i += 1024u) {
        atomic_store_explicit(&shared_words[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Thread 0: compute sections + write header ----
    if (tid == 0u) {
        int32_t gg = global_gains[b];
        int32_t wseq = window_seqs[b];
        threadgroup uint32_t* raw = (threadgroup uint32_t*)shared_words;
        uint bp = 0;

        // ID_SCE(3) + tag(4) + global_gain(8)
        write_bits_seq(raw, bp, 0, 3); bp += 3;
        write_bits_seq(raw, bp, 0, 4); bp += 4;
        write_bits_seq(raw, bp, (uint)gg & 0xFF, 8); bp += 8;

        // ics_info
        write_bits_seq(raw, bp, 0, 1); bp += 1;  // reserved
        write_bits_seq(raw, bp, (uint)wseq & 3, 2); bp += 2;
        write_bits_seq(raw, bp, 1, 1); bp += 1;  // KBD
        write_bits_seq(raw, bp, (uint)num_sfb & 0x3F, 6); bp += 6;  // max_sfb
        write_bits_seq(raw, bp, 0, 1); bp += 1;  // predictor=0

        // Compute per-SFB codebook from max abs value
        for (int sb = 0; sb < num_sfb; sb++) {
            int lo = sfb_offsets_buf[sb];
            int hi = sfb_offsets_buf[sb + 1];
            int max_abs = 0;
            for (int j = lo; j < hi; j++) {
                int v = quantized[b * N + j];
                int av = (v >= 0) ? v : -v;
                if (av > max_abs) max_abs = av;
            }
            // Clamp to 255 for ESC safety
            if (max_abs > 255) max_abs = 255;

            int cb;
            if (max_abs == 0) cb = 0;
            else if (max_abs <= 1) cb = 1;
            else if (max_abs <= 2) cb = 3;
            else if (max_abs <= 4) cb = 5;
            else if (max_abs <= 7) cb = 7;
            else if (max_abs <= 12) cb = 9;
            else cb = 11;
            sfb_cb[sb] = cb;
        }

        // section_data: merge adjacent SFBs with same codebook
        int k = 0;
        while (k < num_sfb) {
            int cb = sfb_cb[k];
            int j = k + 1;
            while (j < num_sfb && sfb_cb[j] == cb) j++;
            int sect_len = j - k;

            write_bits_seq(raw, bp, (uint)cb & 0xF, 4); bp += 4;
            while (sect_len >= 31) {
                write_bits_seq(raw, bp, 31, 5); bp += 5;
                sect_len -= 31;
            }
            write_bits_seq(raw, bp, (uint)sect_len, 5); bp += 5;
            k = j;
        }

        // scale_factor_data: DPCM with SF codebook
        int prev_sf = gg;
        for (int sb = 0; sb < num_sfb; sb++) {
            if (sfb_cb[sb] == 0) continue;
            int iso_sf = gg - scalefactors[b * num_sfb + sb];
            if (iso_sf < 0) iso_sf = 0;
            if (iso_sf > 255) iso_sf = 255;
            int diff = iso_sf - prev_sf;
            if (diff < -60) diff = -60;
            if (diff > 60) diff = 60;
            int idx = diff + 60;
            HuffEntry e = sf_lut[idx];
            write_bits_seq(raw, bp, e.code, e.bits); bp += e.bits;
            prev_sf = iso_sf;
        }

        // pulse/tns/gain control
        write_bits_seq(raw, bp, 0, 1); bp += 1;
        write_bits_seq(raw, bp, 0, 1); bp += 1;
        write_bits_seq(raw, bp, 0, 1); bp += 1;

        header_bits = bp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- All threads: spectral_data (pair encoding) ----
    // Each thread encodes one coefficient pair (tid covers 0..511 pairs)
    uint hbits = header_bits;

    if (tid < (uint)N / 2u) {
        uint coeff_idx = tid * 2u;
        int lo_sfb = -1;
        int cb = 0;

        // Find which SFB this pair belongs to and its codebook
        for (int sb = 0; sb < num_sfb; sb++) {
            uint slo = (uint)sfb_offsets_buf[sb];
            uint shi = (uint)sfb_offsets_buf[sb + 1];
            if (coeff_idx >= slo && coeff_idx < shi) {
                cb = sfb_cb[sb];
                break;
            }
        }

        if (cb != 0 && cb >= 1 && cb <= 11) {
            int v0 = quantized[b * N + coeff_idx];
            int v1 = quantized[b * N + coeff_idx + 1];

            // Clamp
            if (v0 > 255) v0 = 255; if (v0 < -255) v0 = -255;
            if (v1 > 255) v1 = 255; if (v1 < -255) v1 = -255;

            int is_signed = cb_signed[cb];
            int dim = cb_dims[cb];
            int mabs = cb_max_abs[cb];
            int cb_off = cb_offsets[cb];

            // For pair codebooks (dim=2)
            if (dim == 2) {
                int a0, a1;
                if (is_signed) {
                    a0 = v0; a1 = v1;
                } else {
                    a0 = (v0 >= 0) ? v0 : -v0;
                    a1 = (v1 >= 0) ? v1 : -v1;
                    if (cb == 11) {
                        if (a0 > 16) a0 = 16;
                        if (a1 > 16) a1 = 16;
                    } else {
                        if (a0 > mabs) a0 = mabs;
                        if (a1 > mabs) a1 = mabs;
                    }
                }

                int dim_size = is_signed ? (2 * mabs + 1) : (mabs + 1);
                int lookup_a0 = is_signed ? (a0 + mabs) : a0;
                int lookup_a1 = is_signed ? (a1 + mabs) : a1;
                int idx = lookup_a0 * dim_size + lookup_a1;
                HuffEntry e = cb_luts[cb_off + idx];

                // For now, store codeword + length per thread for later packing
                // We'll use a simpler approach: thread 0 handles all spectral sequentially
                // (proper parallel scatter would need prefix sum on variable-length codes)
            }
        }
    }

    // ---- Thread 0: sequential spectral encoding (simpler, still GPU-fast) ----
    // Each frame's spectral data is small (<500 bytes), sequential on one GPU
    // thread is fast enough. The win is from running B frames in parallel.
    if (tid == 0u) {
        threadgroup uint32_t* raw = (threadgroup uint32_t*)shared_words;
        uint bp = hbits;

        for (int sb = 0; sb < num_sfb; sb++) {
            int cb = sfb_cb[sb];
            if (cb == 0 || cb < 1 || cb > 11) continue;

            int is_signed = cb_signed[cb];
            int dim = cb_dims[cb];
            int mabs = cb_max_abs[cb];
            int cb_off = cb_offsets[cb];
            int dim_size = is_signed ? (2 * mabs + 1) : (mabs + 1);

            int lo = sfb_offsets_buf[sb];
            int hi = sfb_offsets_buf[sb + 1];

            if (dim == 4) {
                for (int i = lo; i < hi; i += 4) {
                    int vals[4];
                    for (int k = 0; k < 4; k++) {
                        int v = (i+k < N) ? quantized[b*N + i+k] : 0;
                        if (v > 255) v = 255; if (v < -255) v = -255;
                        vals[k] = v;
                    }
                    int lookup[4];
                    if (is_signed) {
                        for (int k = 0; k < 4; k++) lookup[k] = vals[k] + mabs;
                    } else {
                        for (int k = 0; k < 4; k++) {
                            int av = (vals[k] >= 0) ? vals[k] : -vals[k];
                            if (av > mabs) av = mabs;
                            lookup[k] = av;
                        }
                    }
                    int idx = lookup[0]*dim_size*dim_size*dim_size + lookup[1]*dim_size*dim_size + lookup[2]*dim_size + lookup[3];
                    HuffEntry e = cb_luts[cb_off + idx];
                    write_bits_seq(raw, bp, e.code, e.bits); bp += e.bits;

                    if (!is_signed) {
                        for (int k = 0; k < 4; k++) {
                            int av = (vals[k] >= 0) ? vals[k] : -vals[k];
                            if (av > 0) {
                                write_bits_seq(raw, bp, (vals[k] < 0) ? 1 : 0, 1); bp += 1;
                            }
                        }
                    }
                }
            } else {
                for (int i = lo; i < hi; i += 2) {
                    int v0 = (i < N) ? quantized[b*N + i] : 0;
                    int v1 = (i+1 < N) ? quantized[b*N + i+1] : 0;
                    if (v0 > 255) v0 = 255; if (v0 < -255) v0 = -255;
                    if (v1 > 255) v1 = 255; if (v1 < -255) v1 = -255;

                    int a0, a1;
                    if (is_signed) {
                        a0 = v0 + mabs; a1 = v1 + mabs;
                    } else {
                        a0 = (v0 >= 0) ? v0 : -v0;
                        a1 = (v1 >= 0) ? v1 : -v1;
                        int clamp = (cb == 11) ? 16 : mabs;
                        if (a0 > clamp) a0 = clamp;
                        if (a1 > clamp) a1 = clamp;
                    }

                    int idx = a0 * dim_size + a1;
                    HuffEntry e = cb_luts[cb_off + idx];
                    write_bits_seq(raw, bp, e.code, e.bits); bp += e.bits;

                    if (!is_signed) {
                        // Sign bits: written for any non-zero value (using clamped abs)
                        if (a0 > 0) {
                            write_bits_seq(raw, bp, (v0 < 0) ? 1 : 0, 1); bp += 1;
                        }
                        if (a1 > 0) {
                            write_bits_seq(raw, bp, (v1 < 0) ? 1 : 0, 1); bp += 1;
                        }

                        // Escape coding for CB11
                        if (cb == 11) {
                            int orig0 = (v0 >= 0) ? v0 : -v0;
                            int orig1 = (v1 >= 0) ? v1 : -v1;
                            if (orig0 >= 16) {
                                int n = orig0;
                                int count = 0;
                                while (n >= (1 << (count + 5))) count++;
                                for (int c = 0; c < count; c++) {
                                    write_bits_seq(raw, bp, 1, 1); bp += 1;
                                }
                                write_bits_seq(raw, bp, 0, 1); bp += 1;
                                write_bits_seq(raw, bp, n, count + 4); bp += count + 4;
                            }
                            if (orig1 >= 16) {
                                int n = orig1;
                                int count = 0;
                                while (n >= (1 << (count + 5))) count++;
                                for (int c = 0; c < count; c++) {
                                    write_bits_seq(raw, bp, 1, 1); bp += 1;
                                }
                                write_bits_seq(raw, bp, 0, 1); bp += 1;
                                write_bits_seq(raw, bp, n, count + 4); bp += count + 4;
                            }
                        }
                    }
                }
            }
        }

        // ID_END
        write_bits_seq(raw, bp, 7, 3); bp += 3;

        // Convert uint32 words to bytes (big-endian) in output
        uint total_bytes = (bp + 7u) / 8u;
        uint total_words = (total_bytes + 3u) / 4u;
        device uint8_t* out = output_buf + b * (uint)max_frame_bytes;

        for (uint i = 0; i < total_words; i++) {
            uint32_t w = atomic_load_explicit(&shared_words[i], memory_order_relaxed);
            uint base = i * 4u;
            if (base < total_bytes) out[base] = (uint8_t)(w >> 24);
            if (base+1 < total_bytes) out[base+1] = (uint8_t)(w >> 16);
            if (base+2 < total_bytes) out[base+2] = (uint8_t)(w >> 8);
            if (base+3 < total_bytes) out[base+3] = (uint8_t)w;
        }

        frame_sizes[b] = (int32_t)total_bytes;
    }
}
