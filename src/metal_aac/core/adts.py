"""ADTS (Audio Data Transport Stream) frame format per ISO/IEC 14496-3.

Each ADTS frame wraps one raw_data_block with a 7-byte fixed header
(no CRC). This allows the bitstream to be self-synchronizing — a decoder
can find frame boundaries by scanning for the 0xFFF sync word.

Header layout (56 bits / 7 bytes, no CRC):
  syncword                 12 bits  0xFFF
  id                        1 bit   0 = MPEG-4, 1 = MPEG-2
  layer                     2 bits  always 0
  protection_absent         1 bit   1 = no CRC
  profile                   2 bits  0 = Main, 1 = LC, 2 = SSR, 3 = LTP
  sampling_frequency_index  4 bits  see SAMPLING_FREQ_TABLE
  private_bit               1 bit   0
  channel_configuration     3 bits  1 = mono, 2 = stereo, ...
  originality               1 bit   0
  home                      1 bit   0
  copyright_id_bit          1 bit   0
  copyright_id_start        1 bit   0
  aac_frame_length         13 bits  header + raw_data_block bytes
  adts_buffer_fullness     11 bits  0x7FF = VBR
  number_of_raw_data_blocks 2 bits  0 = 1 block per frame
"""

from __future__ import annotations

import struct

SYNC_WORD = 0xFFF

PROFILE_MAIN = 0
PROFILE_LC = 1
PROFILE_SSR = 2

SAMPLING_FREQ_TABLE = {
    96000: 0, 88200: 1, 64000: 2, 48000: 3,
    44100: 4, 32000: 5, 24000: 6, 22050: 7,
    16000: 8, 12000: 9, 11025: 10, 8000: 11,
}
SAMPLING_FREQ_INVERSE = {v: k for k, v in SAMPLING_FREQ_TABLE.items()}

ADTS_HEADER_SIZE = 7


def write_adts_header(
    raw_data_block_size: int,
    sample_rate: int = 44100,
    channel_config: int = 1,
    profile: int = PROFILE_LC,
) -> bytes:
    """Build a 7-byte ADTS fixed header (no CRC).

    raw_data_block_size: byte count of the raw_data_block payload.
    Returns 7 bytes.
    """
    frame_length = ADTS_HEADER_SIZE + raw_data_block_size
    sf_index = SAMPLING_FREQ_TABLE.get(sample_rate, 4)

    # Byte 0-1: syncword(12) + id(1) + layer(2) + protection_absent(1)
    b0 = 0xFF
    b1 = 0xF0 | (1 << 3) | (0 << 1) | 1  # id=1(MPEG2), layer=00, prot=1(noCRC)

    # Byte 2: profile(2) + sf_index(4) + private(1) + channel_config_hi(1)
    b2 = ((profile & 0x3) << 6) | ((sf_index & 0xF) << 2) | (0 << 1) | ((channel_config >> 2) & 0x1)

    # Byte 3: channel_config_lo(2) + originality(1) + home(1) + copyright_id_bit(1) + copyright_id_start(1) + frame_length_hi(2)
    b3 = ((channel_config & 0x3) << 6) | (0 << 5) | (0 << 4) | (0 << 3) | (0 << 2) | ((frame_length >> 11) & 0x3)

    # Byte 4: frame_length_mid(8)
    b4 = (frame_length >> 3) & 0xFF

    # Byte 5: frame_length_lo(3) + buffer_fullness_hi(5)
    buffer_fullness = 0  # 0 = CBR (matches afconvert convention)
    b5 = ((frame_length & 0x7) << 5) | ((buffer_fullness >> 6) & 0x1F)

    # Byte 6: buffer_fullness_lo(6) + num_raw_data_blocks(2)
    b6 = ((buffer_fullness & 0x3F) << 2) | 0  # 0 = 1 raw_data_block

    return bytes([b0, b1, b2, b3, b4, b5, b6])


def parse_adts_header(data: bytes) -> dict | None:
    """Parse a 7-byte ADTS header. Returns None if sync word not found."""
    if len(data) < ADTS_HEADER_SIZE:
        return None

    if data[0] != 0xFF or (data[1] & 0xF0) != 0xF0:
        return None

    protection_absent = data[1] & 0x1
    profile = (data[2] >> 6) & 0x3
    sf_index = (data[2] >> 2) & 0xF
    channel_config = ((data[2] & 0x1) << 2) | ((data[3] >> 6) & 0x3)

    frame_length = ((data[3] & 0x3) << 11) | (data[4] << 3) | ((data[5] >> 5) & 0x7)
    buffer_fullness = ((data[5] & 0x1F) << 6) | ((data[6] >> 2) & 0x3F)
    num_raw_blocks = data[6] & 0x3

    header_size = 7 if protection_absent else 9

    return {
        "profile": profile,
        "sampling_frequency_index": sf_index,
        "sample_rate": SAMPLING_FREQ_INVERSE.get(sf_index, 0),
        "channel_configuration": channel_config,
        "frame_length": frame_length,
        "header_size": header_size,
        "payload_size": frame_length - header_size,
        "buffer_fullness": buffer_fullness,
        "num_raw_data_blocks": num_raw_blocks + 1,
    }


class ADTSWriter:
    """Write ADTS frames to a byte buffer."""

    def __init__(self, sample_rate: int = 44100, num_channels: int = 1):
        self._buffer = bytearray()
        self._sample_rate = sample_rate
        self._channel_config = num_channels
        self._num_frames = 0

    def write_frame(self, raw_data_block: bytes) -> None:
        header = write_adts_header(
            len(raw_data_block),
            self._sample_rate,
            self._channel_config,
        )
        self._buffer.extend(header)
        self._buffer.extend(raw_data_block)
        self._num_frames += 1

    def get_bytes(self) -> bytes:
        return bytes(self._buffer)

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def get_bitrate(self, audio_duration: float) -> float:
        if audio_duration <= 0:
            return 0.0
        return len(self._buffer) * 8 / (audio_duration * 1000)


class ADTSReader:
    """Read ADTS frames from a byte buffer."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read_frame(self) -> tuple[dict, bytes] | None:
        """Read next ADTS frame. Returns (header_dict, raw_data_block) or None."""
        while self._pos + ADTS_HEADER_SIZE <= len(self._data):
            header = parse_adts_header(self._data[self._pos:])
            if header is None:
                self._pos += 1
                continue

            payload_start = self._pos + header["header_size"]
            payload_end = self._pos + header["frame_length"]
            if payload_end > len(self._data):
                return None

            payload = self._data[payload_start:payload_end]
            self._pos = payload_end
            return header, payload

        return None

    def read_all_frames(self) -> list[tuple[dict, bytes]]:
        frames = []
        while True:
            result = self.read_frame()
            if result is None:
                break
            frames.append(result)
        return frames
