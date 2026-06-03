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

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.metal_huffman_destroy(self._ctx)
            self._ctx = None
