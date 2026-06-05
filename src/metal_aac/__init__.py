"""GPU-accelerated AAC-LC encoder/decoder for Apple Silicon."""

__version__ = "0.13.1"

from metal_aac.encoder import encode, EncoderConfig, EncoderResult
from metal_aac.decoder import decode, DecoderConfig, DecoderResult

__all__ = [
    "encode", "EncoderConfig", "EncoderResult",
    "decode", "DecoderConfig", "DecoderResult",
]
