"""Window functions for AAC MDCT per ISO/IEC 14496-3 section 11.2.1.

Supports long (2048), short (256), and transition (long-start, long-stop)
window shapes. KBD and sine windows both satisfy the Princen-Bradley
condition for perfect reconstruction in MDCT overlap-add.
"""

from __future__ import annotations

import numpy as np


def kbd_window(n: int, alpha: float = 4.0) -> np.ndarray:
    """Kaiser-Bessel Derived window."""
    half = n // 2
    kb = np.kaiser(half + 1, np.pi * alpha)
    cumsum = np.cumsum(kb)
    w_half = np.sqrt(cumsum[:-1] / cumsum[-1])
    return np.concatenate([w_half, w_half[::-1]]).astype(np.float32)


def sine_window(n: int) -> np.ndarray:
    """Sine window: w[k] = sin(pi * (k + 0.5) / N)."""
    k = np.arange(n)
    return np.sin(np.pi * (k + 0.5) / n).astype(np.float32)


def long_start_window(n_long: int = 2048, n_short: int = 256, name: str = "kbd") -> np.ndarray:
    """Transition window: long → short (ISO 14496-3 Figure 4.19).

    First half = long window rising slope.
    Middle = ones.
    Last quarter = short window falling slope.
    Tail = zeros.
    """
    w_long = get_window(name, n_long)
    w_short = get_window(name, n_short)
    half_long = n_long // 2
    half_short = n_short // 2

    w = np.zeros(n_long, dtype=np.float32)
    w[:half_long] = w_long[:half_long]
    w[half_long: half_long + (half_long - half_short)] = 1.0
    w[half_long + (half_long - half_short): half_long + (half_long - half_short) + half_short] = w_short[half_short:]
    return w


def long_stop_window(n_long: int = 2048, n_short: int = 256, name: str = "kbd") -> np.ndarray:
    """Transition window: short → long (ISO 14496-3 Figure 4.19).

    Head = zeros.
    First quarter = short window rising slope.
    Middle = ones.
    Second half = long window falling slope.
    """
    w_long = get_window(name, n_long)
    w_short = get_window(name, n_short)
    half_long = n_long // 2
    half_short = n_short // 2

    w = np.zeros(n_long, dtype=np.float32)
    w[half_short: half_short + half_short] = w_short[:half_short]
    w[half_short + half_short: half_long] = 1.0
    w[half_long:] = w_long[half_long:]
    return w


def get_window(name: str, n: int = 2048) -> np.ndarray:
    if name == "kbd":
        return kbd_window(n)
    elif name == "sine":
        return sine_window(n)
    else:
        raise ValueError(f"Unknown window: {name}. Use 'kbd' or 'sine'.")


def get_window_for_sequence(
    sequence: int,
    n_long: int = 2048,
    n_short: int = 256,
    name: str = "kbd",
) -> np.ndarray:
    """Get the appropriate window shape for a given window sequence.

    sequence: WindowSequence enum value (0=ONLY_LONG, 1=LONG_START,
              2=EIGHT_SHORT, 3=LONG_STOP)
    """
    if sequence == 0:  # ONLY_LONG
        return get_window(name, n_long)
    elif sequence == 1:  # LONG_START
        return long_start_window(n_long, n_short, name)
    elif sequence == 2:  # EIGHT_SHORT
        return get_window(name, n_short)
    elif sequence == 3:  # LONG_STOP
        return long_stop_window(n_long, n_short, name)
    else:
        raise ValueError(f"Unknown window sequence: {sequence}")
