"""AAC scalefactor band tables for supported sample rates.

Band boundaries define groups of MDCT coefficients that share a single
scalefactor. Values are cumulative sample indices.

Long window: 1024 MDCT lines.
Short window: 128 MDCT lines per window group (8 groups per frame).

Source: ISO/IEC 14496-3 Table 4.110.
"""

from __future__ import annotations

# ---- Long window (1024 lines) ----

SFB_44100_LONG: list[int] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88,
    96, 108, 120, 132, 144, 160, 176, 196, 216, 240, 264, 292, 320,
    352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736,
    768, 800, 832, 864, 896, 928, 1024,
]

SFB_48000_LONG: list[int] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88,
    96, 108, 120, 132, 144, 160, 176, 196, 216, 240, 264, 292, 320,
    352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736,
    768, 800, 832, 864, 896, 928, 1024,
]

SFB_32000_LONG: list[int] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88,
    96, 108, 120, 132, 144, 160, 176, 196, 216, 240, 264, 292, 320,
    352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736,
    768, 800, 832, 864, 896, 928, 960, 1024,
]

SFB_TABLES: dict[int, list[int]] = {
    32000: SFB_32000_LONG,
    44100: SFB_44100_LONG,
    48000: SFB_48000_LONG,
}


def get_sfb_offsets(sample_rate: int) -> list[int]:
    if sample_rate not in SFB_TABLES:
        raise ValueError(
            f"Unsupported sample rate {sample_rate}. "
            f"Supported: {sorted(SFB_TABLES.keys())}"
        )
    return SFB_TABLES[sample_rate]


def get_num_sfb(sample_rate: int) -> int:
    return len(get_sfb_offsets(sample_rate)) - 1


# ---- Short window (128 lines per group) ----

SFB_44100_SHORT: list[int] = [
    0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128,
]

SFB_48000_SHORT: list[int] = [
    0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128,
]

SFB_32000_SHORT: list[int] = [
    0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128,
]

SFB_SHORT_TABLES: dict[int, list[int]] = {
    32000: SFB_32000_SHORT,
    44100: SFB_44100_SHORT,
    48000: SFB_48000_SHORT,
}


def get_sfb_offsets_short(sample_rate: int) -> list[int]:
    if sample_rate not in SFB_SHORT_TABLES:
        raise ValueError(
            f"Unsupported sample rate {sample_rate}. "
            f"Supported: {sorted(SFB_SHORT_TABLES.keys())}"
        )
    return SFB_SHORT_TABLES[sample_rate]


def get_num_sfb_short(sample_rate: int) -> int:
    return len(get_sfb_offsets_short(sample_rate)) - 1
