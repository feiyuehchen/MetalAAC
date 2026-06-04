"""Tests for ADTS header format compliance."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metal_aac.core.adts import (
    ADTS_HEADER_SIZE,
    ADTSReader,
    ADTSWriter,
    parse_adts_header,
    write_adts_header,
)


class TestADTSHeader:
    def test_header_size(self):
        h = write_adts_header(100)
        assert len(h) == 7

    def test_sync_word(self):
        h = write_adts_header(100)
        assert h[0] == 0xFF
        assert (h[1] & 0xF0) == 0xF0

    def test_profile_lc(self):
        h = write_adts_header(100, profile=1)
        parsed = parse_adts_header(h)
        assert parsed["profile"] == 1

    def test_sample_rate_44100(self):
        h = write_adts_header(100, sample_rate=44100)
        parsed = parse_adts_header(h)
        assert parsed["sample_rate"] == 44100
        assert parsed["sampling_frequency_index"] == 4

    def test_sample_rate_48000(self):
        h = write_adts_header(100, sample_rate=48000)
        parsed = parse_adts_header(h)
        assert parsed["sample_rate"] == 48000

    def test_channel_mono(self):
        h = write_adts_header(100, channel_config=1)
        parsed = parse_adts_header(h)
        assert parsed["channel_configuration"] == 1

    def test_channel_stereo(self):
        h = write_adts_header(100, channel_config=2)
        parsed = parse_adts_header(h)
        assert parsed["channel_configuration"] == 2

    def test_frame_length(self):
        payload = 200
        h = write_adts_header(payload)
        parsed = parse_adts_header(h)
        assert parsed["frame_length"] == ADTS_HEADER_SIZE + payload

    def test_roundtrip(self):
        for payload_size in [50, 100, 500, 1000, 4000]:
            h = write_adts_header(payload_size, sample_rate=44100, channel_config=1)
            parsed = parse_adts_header(h)
            assert parsed["payload_size"] == payload_size

    def test_invalid_sync(self):
        assert parse_adts_header(b"\x00\x00\x00\x00\x00\x00\x00") is None


class TestADTSWriter:
    def test_single_frame(self):
        writer = ADTSWriter(sample_rate=44100, num_channels=1)
        writer.write_frame(b"\x00" * 100)
        data = writer.get_bytes()
        assert len(data) == ADTS_HEADER_SIZE + 100
        assert writer.num_frames == 1

    def test_multiple_frames(self):
        writer = ADTSWriter()
        for _ in range(10):
            writer.write_frame(b"\x00" * 50)
        assert writer.num_frames == 10
        assert len(writer.get_bytes()) == 10 * (ADTS_HEADER_SIZE + 50)


class TestADTSReader:
    def test_read_written_frames(self):
        writer = ADTSWriter(sample_rate=44100)
        payloads = [bytes(range(i, i + 20)) for i in range(5)]
        for p in payloads:
            writer.write_frame(p)

        reader = ADTSReader(writer.get_bytes())
        frames = reader.read_all_frames()
        assert len(frames) == 5
        for i, (hdr, payload) in enumerate(frames):
            assert payload == payloads[i]
            assert hdr["sample_rate"] == 44100

    def test_empty_stream(self):
        reader = ADTSReader(b"")
        assert reader.read_all_frames() == []
