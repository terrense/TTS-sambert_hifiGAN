#!/usr/bin/env python3
"""
Plot waveform + log-mel spectrogram from a WAV file.

Usage:
  python plot_waveform_mel.py path/to/audio.wav
  python plot_waveform_mel.py path/to/audio.wav --seconds 5
  python plot_waveform_mel.py path/to/audio.wav --mono
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf
import librosa
import librosa.display


def to_mono(y: np.ndarray) -> np.ndarray:
    """soundfile returns (T,) for mono or (T, C) for multi-channel."""
    if y.ndim == 1:
        return y
    return y.mean(axis=1)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Convert int PCM to float32 in [-1, 1] if needed."""
    if np.issubdtype(y.dtype, np.integer):
        maxv = np.iinfo(y.dtype).max
        return (y.astype(np.float32) / float(maxv)).clip(-1.0, 1.0)
    return y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Plot waveform + log-mel spectrogram.")
    parser.add_argument("wav_path", type=str, help="Path to a .wav file")
    parser.add_argument("--seconds", type=float, default=None, help="Plot only the first N seconds")
    parser.add_argument("--mono", action="store_true", help="Force mono waveform plot (average channels)")

    # Mel/STFT params (common defaults)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=None, help="Default: sr/2")

    args = parser.parse_args()

    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        raise SystemExit(f"File not found: {wav_path}")

    y, sr = sf.read(str(wav_path), always_2d=False)  # (T,) or (T,C)

    # Optionally truncate
    if args.seconds is not None:
        n = int(round(args.seconds * sr))
        y = y[:n] if y.ndim == 1 else y[:n, :]

    # Waveform data for plotting
    if args.mono:
        y_wave = to_mono(y)
        waveform_mode = "mono"
    else:
        y_wave = y
        waveform_mode = "original channels"

    # Mel is almost always computed on mono in TTS pipelines
    y_mono = normalize_audio(to_mono(y))

    fmax = (sr / 2.0) if args.fmax is None else args.fmax

    # Compute log-mel
    S_mel = librosa.feature.melspectrogram(
        y=y_mono,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=fmax,
        power=2.0,  # power mel
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)  # log-mel in dB

    # Time axis for waveform
    T = y_wave.shape[0] if y_wave.ndim == 1 else y_wave.shape[0]
    t = np.arange(T) / sr

    plt.figure(figsize=(14, 8))

    # --- Panel 1: waveform ---
    ax1 = plt.subplot(2, 1, 1)
    if y_wave.ndim == 1:
        ax1.plot(t, y_wave, linewidth=0.8)
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"Waveform ({waveform_mode}) | sr={sr} Hz | samples={T}")
    else:
        C = y_wave.shape[1]
        scale = np.percentile(np.abs(y_wave), 99) + 1e-9
        offset = 2.2 * scale
        for c in range(C):
            ax1.plot(t, y_wave[:, c] + c * offset, linewidth=0.8, label=f"ch{c}")
        ax1.set_ylabel("Amplitude (with offsets)")
        ax1.set_title(f"Waveform ({waveform_mode}) | sr={sr} Hz | channels={C} | samples={T}")
        ax1.legend(loc="upper right", frameon=False)
    ax1.set_xlabel("Time (s)")

    # --- Panel 2: log-mel spectrogram ---
    ax2 = plt.subplot(2, 1, 2)
    img = librosa.display.specshow(
        S_mel_db,
        sr=sr,
        hop_length=args.hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=args.fmin,
        fmax=fmax,
        ax=ax2,
    )
    ax2.set_title(
        f"Log-Mel Spectrogram | n_mels={args.n_mels}, n_fft={args.n_fft}, hop={args.hop_length}, win={args.win_length}"
    )
    plt.colorbar(img, ax=ax2, format="%+2.0f dB")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
