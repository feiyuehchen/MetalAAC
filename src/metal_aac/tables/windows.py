"""Window functions for AAC MDCT.

AAC-LC uses either the Kaiser-Bessel Derived (KBD) window or the sine window,
both of length 2048 for long blocks.
"""

from __future__ import annotations

import numpy as np


def kbd_window(n: int, alpha: float = 4.0) -> np.ndarray:
    """Kaiser-Bessel Derived window.

    The KBD window is constructed from a Kaiser-Bessel window by accumulating
    and normalizing. It satisfies the Princen-Bradley condition for perfect
    reconstruction in MDCT overlap-add.
    """
    half = n // 2
    kb = np.kaiser(half + 1, np.pi * alpha)
    cumsum = np.cumsum(kb)
    w_half = np.sqrt(cumsum[:-1] / cumsum[-1])
    return np.concatenate([w_half, w_half[::-1]]).astype(np.float32)


def sine_window(n: int) -> np.ndarray:
    """Sine window: w[k] = sin(pi * (k + 0.5) / N).

    Also satisfies Princen-Bradley for MDCT overlap-add.
    """
    k = np.arange(n)
    return np.sin(np.pi * (k + 0.5) / n).astype(np.float32)


def get_window(name: str, n: int = 2048) -> np.ndarray:
    if name == "kbd":
        return kbd_window(n)
    elif name == "sine":
        return sine_window(n)
    else:
        raise ValueError(f"Unknown window: {name}. Use 'kbd' or 'sine'.")
