"""Frame-level bitstream writer/reader.

Each encoded frame is packaged with a header containing:
- sync word (16 bits)
- frame length (16 bits)
- sample rate index (4 bits)
- channel config (4 bits)
- global gain (8 bits)
- num scalefactor bands (8 bits)

Followed by the entropy-coded spectral data.

This is a simplified frame format (not full ADTS) designed for
round-trip encode/decode testing. The focus of this research
is GPU optimization, not container format compliance.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

SYNC_WORD = 0xAAC1

SAMPLE_RATE_INDEX = {
    96000: 0,
    88200: 1,
    64000: 2,
    48000: 3,
    44100: 4,
    32000: 5,
    24000: 6,
    22050: 7,
    16000: 8,
    12000: 9,
    11025: 10,
    8000: 11,
}

INDEX_TO_SAMPLE_RATE = {v: k for k, v in SAMPLE_RATE_INDEX.items()}


@dataclass
class FrameHeader:
    sync: int
    frame_length: int  # total frame bytes including header
    sample_rate_index: int
    channel_config: int
    num_sfb: int

    HEADER_SIZE = 7  # bytes

    def to_bytes(self) -> bytes:
        return struct.pack(
            ">HHBBB",
            self.sync,
            self.frame_length,
            (self.sample_rate_index << 4) | (self.channel_config & 0x0F),
            self.num_sfb,
            0,  # reserved
        )

    @staticmethod
    def from_bytes(data: bytes) -> FrameHeader:
        sync, frame_length, sr_ch, num_sfb, _ = struct.unpack(
            ">HHBBB", data[:FrameHeader.HEADER_SIZE]
        )
        return FrameHeader(
            sync=sync,
            frame_length=frame_length,
            sample_rate_index=(sr_ch >> 4) & 0x0F,
            channel_config=sr_ch & 0x0F,
            num_sfb=num_sfb,
        )


class BitstreamWriter:
    """Write encoded frames to a byte buffer."""

    def __init__(self):
        self._buffer = bytearray()

    def write_frame(
        self,
        spectral_data: bytes,
        sample_rate: int,
        num_channels: int,
        num_sfb: int,
    ) -> None:
        header = FrameHeader(
            sync=SYNC_WORD,
            frame_length=FrameHeader.HEADER_SIZE + len(spectral_data),
            sample_rate_index=SAMPLE_RATE_INDEX.get(sample_rate, 4),
            channel_config=num_channels,
            num_sfb=num_sfb,
        )
        self._buffer.extend(header.to_bytes())
        self._buffer.extend(spectral_data)

    def get_bytes(self) -> bytes:
        return bytes(self._buffer)

    def get_bitrate(self, audio_duration: float) -> float:
        """Compute actual bitrate in kbps."""
        return len(self._buffer) * 8 / (audio_duration * 1000)


class BitstreamReader:
    """Read encoded frames from a byte buffer."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read_frame(self) -> tuple[FrameHeader, bytes] | None:
        """Read next frame. Returns None at end of stream."""
        if self._pos + FrameHeader.HEADER_SIZE > len(self._data):
            return None

        header = FrameHeader.from_bytes(self._data[self._pos :])
        if header.sync != SYNC_WORD:
            return None

        payload_start = self._pos + FrameHeader.HEADER_SIZE
        payload_end = self._pos + header.frame_length
        if payload_end > len(self._data):
            return None

        payload = self._data[payload_start:payload_end]
        self._pos = payload_end
        return header, payload

    def read_all_frames(self) -> list[tuple[FrameHeader, bytes]]:
        frames = []
        while True:
            result = self.read_frame()
            if result is None:
                break
            frames.append(result)
        return frames
