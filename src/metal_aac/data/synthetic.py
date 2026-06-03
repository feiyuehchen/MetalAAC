"""Deterministic synthetic test signal generation.

All signals defined in DATASET.md v1. Re-running must produce bit-identical output.
"""

from __future__ import annotations

import numpy as np


def generate_test_signals(
    sample_rate: int = 44100,
    include_1hr: bool = False,
) -> dict[str, dict]:
    signals = {}

    def _add(name: str, pcm: np.ndarray, duration: float, desc: str):
        signals[name] = {
            "pcm": pcm.astype(np.float32),
            "sample_rate": sample_rate,
            "duration": duration,
            "description": desc,
        }

    sr = sample_rate

    # Sine waves
    for freq, name in [(440, "sine_440"), (1000, "sine_1k"), (4000, "sine_4k")]:
        dur = 1.0
        t = np.arange(int(sr * dur)) / sr
        _add(name, 0.9 * np.sin(2 * np.pi * freq * t), dur, f"{freq} Hz sine wave")

    # Multi-tone
    dur = 1.0
    t = np.arange(int(sr * dur)) / sr
    multi = 0.3 * (
        np.sin(2 * np.pi * 440 * t)
        + np.sin(2 * np.pi * 1000 * t)
        + np.sin(2 * np.pi * 4000 * t)
    )
    _add("multitone", multi, dur, "440 + 1000 + 4000 Hz")

    # Chirp
    dur = 2.0
    t = np.arange(int(sr * dur)) / sr
    f0, f1 = 20.0, 20000.0
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * dur))
    _add("chirp", 0.9 * np.sin(phase), dur, "Linear sweep 20-20000 Hz")

    # White noise
    rng = np.random.RandomState(42)
    dur = 1.0
    white = 0.5 * rng.randn(int(sr * dur)).astype(np.float32)
    white = np.clip(white, -1.0, 1.0)
    _add("white_noise", white, dur, "Gaussian white noise (seed=42)")

    # Pink noise via spectral shaping
    rng2 = np.random.RandomState(42)
    dur = 1.0
    n = int(sr * dur)
    white = rng2.randn(n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    freqs[0] = 1.0  # avoid division by zero
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n)
    pink = 0.5 * pink / (np.max(np.abs(pink)) + 1e-8)
    pink = np.clip(pink, -1.0, 1.0)
    _add("pink_noise", pink, dur, "1/f noise (seed=42)")

    # Silence
    _add("silence", np.zeros(int(sr * 0.5), dtype=np.float32), 0.5, "All zeros")

    # Impulse
    dur = 1.0
    imp = np.zeros(int(sr * dur), dtype=np.float32)
    imp[int(sr * 0.5)] = 1.0
    _add("impulse", imp, dur, "Single sample = 1.0 at t=0.5s")

    # Longer synthetic signals for throughput benchmarks
    for target_dur, name_suffix in [(60.0, "1min"), (300.0, "5min")]:
        t_long = np.arange(int(sr * target_dur)) / sr
        freqs_cycle = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        note_len = 0.5  # seconds per note
        sig = np.zeros_like(t_long)
        for j, s in enumerate(t_long):
            f = freqs_cycle[int(s / note_len) % len(freqs_cycle)]
            sig[j] = (
                0.4 * np.sin(2 * np.pi * f * s)
                + 0.2 * np.sin(2 * np.pi * 2 * f * s)
                + 0.1 * np.sin(2 * np.pi * 3 * f * s)
            )
        _add(f"music_{name_suffix}", sig.astype(np.float32), target_dur,
             f"Synthetic harmonic sequence ({name_suffix})")

    # Long synthetic harmonic sequence
    dur = 10.0
    t = np.arange(int(sr * dur)) / sr
    freqs_seq = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    note_dur = dur / len(freqs_seq)
    music = np.zeros_like(t)
    for i, f in enumerate(freqs_seq):
        start = int(i * note_dur * sr)
        end = int((i + 1) * note_dur * sr)
        seg_t = np.arange(end - start) / sr
        envelope = np.exp(-2.0 * seg_t / note_dur)
        note = envelope * (
            np.sin(2 * np.pi * f * seg_t)
            + 0.5 * np.sin(2 * np.pi * 2 * f * seg_t)
            + 0.25 * np.sin(2 * np.pi * 3 * f * seg_t)
        )
        music[start:end] = note
    music = 0.8 * music / (np.max(np.abs(music)) + 1e-8)
    _add("long_music", music, dur, "Synthetic harmonic sequence")

    if include_1hr:
        dur_1hr = 3600.0
        t_1hr = np.arange(int(sr * dur_1hr)) / sr
        freqs_cycle = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        note_len = 0.5
        sig_1hr = np.zeros_like(t_1hr)
        for j, s in enumerate(t_1hr):
            f = freqs_cycle[int(s / note_len) % len(freqs_cycle)]
            sig_1hr[j] = (
                0.4 * np.sin(2 * np.pi * f * s)
                + 0.2 * np.sin(2 * np.pi * 2 * f * s)
                + 0.1 * np.sin(2 * np.pi * 3 * f * s)
            )
        _add("music_1hr", sig_1hr.astype(np.float32), dur_1hr,
             "Synthetic harmonic sequence (1hr)")

    return signals
