"""Python bridge to Metal GPU Huffman encoder/decoder via ctypes."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "metal" / "libmetal_huffman.dylib"

_c_int32_p = ctypes.POINTER(ctypes.c_int32)
_c_uint8_p = ctypes.POINTER(ctypes.c_uint8)


class MetalHuffman:
    _instance: MetalHuffman | None = None

    def __init__(self):
        if not _LIB_PATH.exists():
            raise FileNotFoundError(
                f"Metal library not found at {_LIB_PATH}. "
                "Run 'make' in the metal/ directory first."
            )
        self._lib = ctypes.CDLL(str(_LIB_PATH))

        self._lib.metal_huffman_create.restype = ctypes.c_void_p
        self._lib.metal_huffman_create.argtypes = []

        self._lib.metal_huffman_destroy.restype = None
        self._lib.metal_huffman_destroy.argtypes = [ctypes.c_void_p]

        self._lib.metal_huffman_max_frame_bytes.restype = ctypes.c_int32
        self._lib.metal_huffman_max_frame_bytes.argtypes = [
            ctypes.c_int32, ctypes.c_int32,
        ]

        self._lib.metal_huffman_encode.restype = ctypes.c_int
        self._lib.metal_huffman_encode.argtypes = [
            ctypes.c_void_p,  # ctx
            _c_int32_p,       # quantized
            _c_int32_p,       # scalefactors
            _c_int32_p,       # global_gains
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # B, N, num_sfb
            _c_uint8_p,       # output_buf
            ctypes.c_int32,   # max_frame_bytes
            _c_int32_p,       # frame_sizes
        ]

        self._lib.metal_huffman_decode.restype = ctypes.c_int
        self._lib.metal_huffman_decode.argtypes = [
            ctypes.c_void_p,  # ctx
            _c_uint8_p,       # compressed
            _c_int32_p,       # frame_offsets
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # B, N, num_sfb
            _c_int32_p,       # quantized_out
            _c_int32_p,       # scalefactors_out
            _c_int32_p,       # global_gains_out
        ]

        self._ctx = self._lib.metal_huffman_create()
        if not self._ctx:
            raise RuntimeError("Failed to create Metal Huffman context")

    @classmethod
    def shared(cls) -> MetalHuffman:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_frames(
        self,
        quantized: np.ndarray,
        scalefactors: np.ndarray,
        global_gains: np.ndarray,
    ) -> list[bytes]:
        B, N = quantized.shape
        num_sfb = scalefactors.shape[1]
        max_bytes = self._lib.metal_huffman_max_frame_bytes(N, num_sfb)

        q = np.ascontiguousarray(quantized, dtype=np.int32)
        sf = np.ascontiguousarray(scalefactors, dtype=np.int32)
        gg = np.ascontiguousarray(global_gains, dtype=np.int32)

        output = np.zeros(B * max_bytes, dtype=np.uint8)
        sizes = np.zeros(B, dtype=np.int32)

        rc = self._lib.metal_huffman_encode(
            self._ctx,
            q.ctypes.data_as(_c_int32_p),
            sf.ctypes.data_as(_c_int32_p),
            gg.ctypes.data_as(_c_int32_p),
            B, N, num_sfb,
            output.ctypes.data_as(_c_uint8_p),
            max_bytes,
            sizes.ctypes.data_as(_c_int32_p),
        )
        if rc != 0:
            raise RuntimeError(f"Metal encode failed (rc={rc})")

        result = []
        for b in range(B):
            start = b * max_bytes
            end = start + sizes[b]
            result.append(bytes(output[start:end]))
        return result

    def encode_adts_frames(
        self,
        quantized: np.ndarray,
        scalefactors: np.ndarray,
        global_gains: np.ndarray,
        window_seqs: np.ndarray,
        sample_rate: int = 44100,
    ) -> list[bytes]:
        """Encode all frames as ISO raw_data_blocks using Metal GPU."""
        from metal_aac.tables.scalefactor_bands import get_sfb_offsets
        from metal_aac.tables.huffman_tables import (
            CODEBOOKS, SF_CODES, SF_CODE_VALUES, SF_CODE_LENGTHS,
        )

        B, N = quantized.shape
        sfb_offsets = get_sfb_offsets(sample_rate)
        num_sfb = len(sfb_offsets) - 1
        max_bytes = self._lib.metal_huffman_max_frame_bytes(N, num_sfb)

        cb_dims_arr = [0, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2]
        cb_signed_arr = [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        cb_max_abs_arr = [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 16]

        # Build codebook LUTs: HuffEntry = uint32 code + uint8 bits + pad[3] = 8 bytes
        import struct as st
        from metal_aac.tables import huffman_tables as ht

        sizes = [0, 81, 81, 81, 81, 81, 81, 64, 64, 169, 169, 289]
        cb_offsets_arr = [0] * 12
        for i in range(2, 12):
            cb_offsets_arr[i] = cb_offsets_arr[i - 1] + sizes[i - 1]

        lut_data = bytearray()
        for cb_idx in range(1, 12):
            codes_arr = getattr(ht, f'CB{cb_idx}_CODES')
            bits_arr = getattr(ht, f'CB{cb_idx}_LENGTHS')
            for i in range(len(codes_arr)):
                lut_data += st.pack('<IB3x', codes_arr[i], bits_arr[i])

        sf_lut_data = bytearray()
        for i in range(121):
            sf_lut_data += st.pack('<IB3x', SF_CODE_VALUES[i], SF_CODE_LENGTHS[i])

        q = np.ascontiguousarray(quantized, dtype=np.int32)
        sf = np.ascontiguousarray(scalefactors, dtype=np.int32)
        gg = np.ascontiguousarray(global_gains, dtype=np.int32)
        ws = np.ascontiguousarray(window_seqs, dtype=np.int32)
        sfb_arr = np.array(sfb_offsets, dtype=np.int32)
        cb_off = np.array(cb_offsets_arr, dtype=np.int32)
        cb_dim = np.array(cb_dims_arr, dtype=np.int32)
        cb_sig = np.array(cb_signed_arr, dtype=np.int32)
        cb_mab = np.array(cb_max_abs_arr, dtype=np.int32)

        output = np.zeros(B * max_bytes, dtype=np.uint8)
        frame_sizes = np.zeros(B, dtype=np.int32)

        _c_float_p = ctypes.POINTER(ctypes.c_float)
        if not hasattr(self._lib, '_adts_setup_done'):
            self._lib.metal_encode_adts_frames.restype = ctypes.c_int
            self._lib.metal_encode_adts_frames.argtypes = [
                ctypes.c_void_p,
                _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p,
                ctypes.c_void_p, ctypes.c_int32,
                _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p,
                ctypes.c_void_p, ctypes.c_int32,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                _c_uint8_p, _c_int32_p,
            ]
            self._lib._adts_setup_done = True

        lut_bytes = bytes(lut_data)
        sf_bytes = bytes(sf_lut_data)

        rc = self._lib.metal_encode_adts_frames(
            self._ctx,
            q.ctypes.data_as(_c_int32_p),
            sf.ctypes.data_as(_c_int32_p),
            gg.ctypes.data_as(_c_int32_p),
            ws.ctypes.data_as(_c_int32_p),
            sfb_arr.ctypes.data_as(_c_int32_p),
            lut_bytes, len(lut_bytes),
            cb_off.ctypes.data_as(_c_int32_p),
            cb_dim.ctypes.data_as(_c_int32_p),
            cb_sig.ctypes.data_as(_c_int32_p),
            cb_mab.ctypes.data_as(_c_int32_p),
            sf_bytes, len(sf_bytes),
            B, N, num_sfb, max_bytes,
            output.ctypes.data_as(_c_uint8_p),
            frame_sizes.ctypes.data_as(_c_int32_p),
        )
        if rc != 0:
            raise RuntimeError(f"Metal ADTS encode failed (rc={rc})")

        result = []
        for b in range(B):
            start = b * max_bytes
            end = start + frame_sizes[b]
            result.append(bytes(output[start:end]))
        return result

    def decode_frames(
        self,
        frame_payloads: list[bytes],
        N: int = 1024,
        num_sfb: int = 49,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = len(frame_payloads)

        # Concatenate all payloads and build offset array
        offsets = np.zeros(B + 1, dtype=np.int32)
        total = 0
        for i, p in enumerate(frame_payloads):
            offsets[i] = total
            total += len(p)
        offsets[B] = total

        compressed = np.frombuffer(b"".join(frame_payloads), dtype=np.uint8)
        compressed = np.ascontiguousarray(compressed)

        quantized_out = np.zeros((B, N), dtype=np.int32)
        sf_out = np.zeros((B, num_sfb), dtype=np.int32)
        gg_out = np.zeros(B, dtype=np.int32)

        rc = self._lib.metal_huffman_decode(
            self._ctx,
            compressed.ctypes.data_as(_c_uint8_p),
            offsets.ctypes.data_as(_c_int32_p),
            B, N, num_sfb,
            quantized_out.ctypes.data_as(_c_int32_p),
            sf_out.ctypes.data_as(_c_int32_p),
            gg_out.ctypes.data_as(_c_int32_p),
        )
        if rc != 0:
            raise RuntimeError(f"Metal decode failed (rc={rc})")

        return quantized_out, sf_out, gg_out

    def quantize(
        self,
        mdct_coeffs: np.ndarray,
        masking_thresholds: np.ndarray,
        target_bits: int,
        sample_rate: int = 44100,
        max_iterations: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Full quantization on Metal GPU.

        Returns: (quantized, scalefactors, global_gains, total_bits)
        """
        from metal_aac.tables.scalefactor_bands import get_sfb_offsets

        B, N = mdct_coeffs.shape
        sfb_offsets = get_sfb_offsets(sample_rate)
        num_sfb = len(sfb_offsets) - 1

        sfb_starts = np.array(sfb_offsets[:-1], dtype=np.int32)
        sfb_ends = np.array(sfb_offsets[1:], dtype=np.int32)
        sfb_map = np.zeros(N, dtype=np.int32)
        for sb in range(num_sfb):
            sfb_map[sfb_offsets[sb] : sfb_offsets[sb + 1]] = sb

        mc = np.ascontiguousarray(mdct_coeffs, dtype=np.float32)
        mk = np.ascontiguousarray(masking_thresholds, dtype=np.float32)

        q_out = np.zeros((B, N), dtype=np.int32)
        sf_out = np.zeros((B, num_sfb), dtype=np.int32)
        gg_out = np.zeros(B, dtype=np.int32)
        tb_out = np.zeros(B, dtype=np.int32)

        _c_float_p = ctypes.POINTER(ctypes.c_float)

        if not hasattr(self._lib, '_quantize_setup_done'):
            self._lib.metal_huffman_quantize.restype = ctypes.c_int
            self._lib.metal_huffman_quantize.argtypes = [
                ctypes.c_void_p,
                _c_float_p, _c_float_p,
                _c_int32_p, _c_int32_p, _c_int32_p,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_int32, ctypes.c_int32,
                _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p,
            ]
            self._lib._quantize_setup_done = True

        rc = self._lib.metal_huffman_quantize(
            self._ctx,
            mc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            sfb_starts.ctypes.data_as(_c_int32_p),
            sfb_ends.ctypes.data_as(_c_int32_p),
            sfb_map.ctypes.data_as(_c_int32_p),
            B, N, num_sfb, target_bits, max_iterations,
            q_out.ctypes.data_as(_c_int32_p),
            sf_out.ctypes.data_as(_c_int32_p),
            gg_out.ctypes.data_as(_c_int32_p),
            tb_out.ctypes.data_as(_c_int32_p),
        )
        if rc != 0:
            raise RuntimeError(f"Metal quantize failed (rc={rc})")

        return q_out, sf_out, gg_out, tb_out

    _iso_decode_cache: dict | None = None

    @classmethod
    def _get_decode_luts(cls):
        if cls._iso_decode_cache is not None:
            return cls._iso_decode_cache
        from metal_aac.tables import huffman_tables as ht
        import struct as st

        cb_dims_arr = [0, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2]
        cb_signed_arr = [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        cb_max_abs_arr = [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 16]
        lut_offsets_arr = [0] * 12
        lut_max_bits_arr = [0] * 12
        spec_lut_data = bytearray()

        for cb_idx in range(1, 12):
            codes = getattr(ht, f'CB{cb_idx}_CODES')
            lengths = getattr(ht, f'CB{cb_idx}_LENGTHS')
            max_bits = max(lengths)
            lut_max_bits_arr[cb_idx] = max_bits
            lut_offsets_arr[cb_idx] = len(spec_lut_data) // 4
            lut_size = 1 << max_bits
            for prefix in range(lut_size):
                best_val, best_len = -1, 0
                for i, (code, length) in enumerate(zip(codes, lengths)):
                    if length == 0 or length > max_bits:
                        continue
                    if (prefix >> (max_bits - length)) == code:
                        best_val, best_len = i, length
                        break
                spec_lut_data += st.pack('<hh', best_val, best_len)

        sf_codes = ht.SF_CODE_VALUES
        sf_lengths = ht.SF_CODE_LENGTHS
        sf_max_bits = max(sf_lengths)
        sf_lut_data = bytearray()
        for prefix in range(1 << sf_max_bits):
            best_val, best_len = -1, 0
            for i, (code, length) in enumerate(zip(sf_codes, sf_lengths)):
                if length == 0 or length > sf_max_bits:
                    continue
                if (prefix >> (sf_max_bits - length)) == code:
                    best_val, best_len = i, length
                    break
            sf_lut_data += st.pack('<hh', best_val, best_len)

        cls._iso_decode_cache = {
            'spec_bytes': bytes(spec_lut_data),
            'sf_bytes': bytes(sf_lut_data),
            'lut_offsets': np.array(lut_offsets_arr, dtype=np.int32),
            'lut_max_bits': np.array(lut_max_bits_arr, dtype=np.int32),
            'cb_dims': np.array(cb_dims_arr, dtype=np.int32),
            'cb_signed': np.array(cb_signed_arr, dtype=np.int32),
            'cb_max_abs': np.array(cb_max_abs_arr, dtype=np.int32),
            'sf_max_bits': sf_max_bits,
        }
        return cls._iso_decode_cache

    def decode_iso_frames(
        self,
        payloads: list[bytes],
        sample_rate: int = 44100,
        N: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode ISO Huffman frames using native C with GCD parallelism."""
        from metal_aac.tables.scalefactor_bands import get_sfb_offsets

        sfb_offsets = get_sfb_offsets(sample_rate)
        num_sfb = len(sfb_offsets) - 1
        B = len(payloads)

        concat = b''.join(payloads)
        offsets = np.zeros(B + 1, dtype=np.int32)
        pos = 0
        for i, p in enumerate(payloads):
            offsets[i] = pos
            pos += len(p)
        offsets[B] = pos

        cache = self._get_decode_luts()

        if not hasattr(self._lib, '_iso_decode_setup'):
            self._lib.metal_decode_iso_frames.restype = ctypes.c_int
            self._lib.metal_decode_iso_frames.argtypes = [
                ctypes.POINTER(ctypes.c_uint8), _c_int32_p,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                _c_int32_p,
                ctypes.c_void_p, _c_int32_p, _c_int32_p,
                _c_int32_p, _c_int32_p, _c_int32_p,
                ctypes.c_void_p, ctypes.c_int32,
                _c_int32_p, _c_int32_p, _c_int32_p,
            ]
            self._lib._iso_decode_setup = True

        concat_arr = np.frombuffer(concat, dtype=np.uint8).copy()
        sfb_arr = np.array(sfb_offsets, dtype=np.int32)
        q_out = np.zeros(B * N, dtype=np.int32)
        sf_out = np.zeros(B * num_sfb, dtype=np.int32)
        gg_out = np.zeros(B, dtype=np.int32)

        rc = self._lib.metal_decode_iso_frames(
            concat_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offsets.ctypes.data_as(_c_int32_p),
            B, N, num_sfb,
            sfb_arr.ctypes.data_as(_c_int32_p),
            cache['spec_bytes'],
            cache['lut_offsets'].ctypes.data_as(_c_int32_p),
            cache['lut_max_bits'].ctypes.data_as(_c_int32_p),
            cache['cb_dims'].ctypes.data_as(_c_int32_p),
            cache['cb_signed'].ctypes.data_as(_c_int32_p),
            cache['cb_max_abs'].ctypes.data_as(_c_int32_p),
            cache['sf_bytes'], cache['sf_max_bits'],
            q_out.ctypes.data_as(_c_int32_p),
            sf_out.ctypes.data_as(_c_int32_p),
            gg_out.ctypes.data_as(_c_int32_p),
        )
        if rc != 0:
            raise RuntimeError(f"Native ISO decode failed (rc={rc})")

        return q_out.reshape(B, N), sf_out.reshape(B, num_sfb), gg_out

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.metal_huffman_destroy(self._ctx)
            self._ctx = None
