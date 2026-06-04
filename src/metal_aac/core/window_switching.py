"""Window switching for AAC-LC per ISO/IEC 14496-3 section 11.2.

The encoder selects one of four window sequences per frame based on
transient detection. Short windows (8x256) suppress pre-echo on attacks;
long windows (2048) give better frequency resolution on stationary signals.

Window sequences:
  ONLY_LONG_SEQUENCE   (0): 2048-pt, steady state
  LONG_START_SEQUENCE  (1): 2048-pt, transition into short
  EIGHT_SHORT_SEQUENCE (2): 8x256-pt, transient region
  LONG_STOP_SEQUENCE   (3): 2048-pt, transition out of short

State machine (ISO 14496-3 Figure 4.18):
  current \\ next_transient  | False              | True
  -------------------------+--------------------+--------------------
  ONLY_LONG                | ONLY_LONG          | LONG_START
  LONG_START               | EIGHT_SHORT        | EIGHT_SHORT
  EIGHT_SHORT              | LONG_STOP          | EIGHT_SHORT
  LONG_STOP                | ONLY_LONG          | LONG_START
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class WindowSequence(IntEnum):
    ONLY_LONG = 0
    LONG_START = 1
    EIGHT_SHORT = 2
    LONG_STOP = 3


NEXT_WINDOW_SEQUENCE: dict[tuple[WindowSequence, bool], WindowSequence] = {
    (WindowSequence.ONLY_LONG, False): WindowSequence.ONLY_LONG,
    (WindowSequence.ONLY_LONG, True): WindowSequence.LONG_START,
    (WindowSequence.LONG_START, False): WindowSequence.EIGHT_SHORT,
    (WindowSequence.LONG_START, True): WindowSequence.EIGHT_SHORT,
    (WindowSequence.EIGHT_SHORT, False): WindowSequence.LONG_STOP,
    (WindowSequence.EIGHT_SHORT, True): WindowSequence.EIGHT_SHORT,
    (WindowSequence.LONG_STOP, False): WindowSequence.ONLY_LONG,
    (WindowSequence.LONG_STOP, True): WindowSequence.LONG_START,
}


def detect_transients_cpu(
    pcm_frames: np.ndarray,
    threshold: float = 8.0,
) -> np.ndarray:
    """Detect transients by comparing energy in frame halves.

    pcm_frames: (B, frame_size) raw PCM (pre-window)
    Returns: (B,) bool array, True where transient detected.
    """
    half = pcm_frames.shape[1] // 2
    energy_first = np.sum(pcm_frames[:, :half] ** 2, axis=1) + 1e-20
    energy_second = np.sum(pcm_frames[:, half:] ** 2, axis=1) + 1e-20
    ratio = energy_second / energy_first
    return ratio > threshold


def detect_transients_gpu(
    pcm_frames: mx.array,
    threshold: float = 8.0,
) -> mx.array:
    """GPU transient detection via MLX energy ratio."""
    half = pcm_frames.shape[1] // 2
    energy_first = mx.sum(pcm_frames[:, :half] ** 2, axis=1) + 1e-20
    energy_second = mx.sum(pcm_frames[:, half:] ** 2, axis=1) + 1e-20
    ratio = energy_second / energy_first
    return ratio > threshold


def compute_window_sequences(
    transients: np.ndarray,
) -> np.ndarray:
    """Apply the window switching state machine across a sequence of frames.

    transients: (B,) bool array from detect_transients
    Returns: (B,) array of WindowSequence values
    """
    n_frames = len(transients)
    sequences = np.zeros(n_frames, dtype=np.int32)
    state = WindowSequence.ONLY_LONG

    for i in range(n_frames):
        is_transient = bool(transients[i])
        # Look ahead: if NEXT frame has a transient, current should start transitioning
        next_transient = bool(transients[i + 1]) if i + 1 < n_frames else False
        attack = is_transient or next_transient

        state = NEXT_WINDOW_SEQUENCE[(state, attack)]
        sequences[i] = int(state)

    return sequences
