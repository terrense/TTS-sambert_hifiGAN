#!/usr/bin/env python3
"""
Plot waveform from a WAV file.

Usage:
  python plot_waveform.py path/to/audio.wav
  python plot_waveform.py path/to/audio.wav --seconds 5
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import soundfile as sf
except ImportError as e:
    raise SystemExit(
        "Missing dependency: soundfile\n"
        "Install with: pip install soundfile matplotlib numpy"
    ) from e


def to_mono(y: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono if it's multi-channel.
    soundfile returns:
      - shape (T,) for mono
      - shape (T, C) for multi-channel
    """
    if y.ndim == 1:
        return y
    # average channels
    return y.mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=str, help="Path to a .wav file")
    parser.add_argument("--seconds", type=float, default=None, help="Plot only the first N seconds")
    parser.add_argument("--mono", action="store_true", help="Force mono by averaging channels")
    args = parser.parse_args()

    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        raise SystemExit(f"File not found: {wav_path}")

    y, sr = sf.read(str(wav_path), always_2d=False)  # y: (T,) or (T,C)
    if args.mono:
        y_plot = to_mono(y)
        title_ch = "mono"
    else:
        y_plot = y
        title_ch = "original channels"

    # Truncate if needed
    if args.seconds is not None:
        n = int(round(args.seconds * sr))
        if y_plot.ndim == 1:
            y_plot = y_plot[:n]
        else:
            y_plot = y_plot[:n, :]

    # Time axis
    T = y_plot.shape[0] if y_plot.ndim == 1 else y_plot.shape[0]
    t = np.arange(T) / sr

    plt.figure(figsize=(12, 4))

    if y_plot.ndim == 1:
        plt.plot(t, y_plot, linewidth=0.8)
        plt.ylabel("Amplitude")
        plt.title(f"Waveform ({title_ch}) | sr={sr} Hz | samples={T}")
    else:
        # multi-channel: plot each channel with a vertical offset to avoid overlap
        C = y_plot.shape[1]
        # offset based on robust amplitude scale
        scale = np.percentile(np.abs(y_plot), 99) + 1e-9
        offset = 2.2 * scale
        for c in range(C):
            plt.plot(t, y_plot[:, c] + c * offset, linewidth=0.8, label=f"ch{c}")
        plt.ylabel("Amplitude (with offsets)")
        plt.title(f"Waveform ({title_ch}) | sr={sr} Hz | channels={C} | samples={T}")
        plt.legend(loc="upper right", frameon=False)

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
