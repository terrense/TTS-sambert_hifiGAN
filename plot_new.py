#!/usr/bin/env python3
"""
plot_audio_views.py

Plot a set of common audio visualizations from a WAV file:
1) Waveform (original channels with offsets)
2) Waveform (mono used for analysis)
3) STFT magnitude (linear, clipped for visibility)
4) STFT magnitude (dB)
5) (optional) STFT phase
6) Mel spectrogram (power, clipped for visibility)
7) Log-mel spectrogram (dB)

Key visualization fixes:
- Linear-scale spectrograms often look "black" due to huge dynamic range.
  We clip vmax to a high percentile (e.g., 99th) to make structure visible.
- For speech, showing only up to 8 kHz is often more informative.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf
import librosa
import librosa.display


def to_mono(y: np.ndarray) -> np.ndarray:
    """y: (N,) or (N,C). Return mono (N,)"""
    if y.ndim == 1:
        return y
    # soundfile may return (N, C)
    return y.mean(axis=1)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Normalize integer PCM to [-1,1] float. Leave float as float32."""
    if np.issubdtype(y.dtype, np.integer):
        maxv = np.iinfo(y.dtype).max
        y = y.astype(np.float32) / float(maxv)
    return y.astype(np.float32)


def percentile_clip(x: np.ndarray, pct: float) -> float:
    """Return a robust vmax for visualization; ignores extreme outliers."""
    pct = float(pct)
    pct = min(max(pct, 0.0), 100.0)
    return float(np.percentile(x, pct))


def main():
    p = argparse.ArgumentParser(
        description="Plot waveform + STFT + mel views with robust visualization defaults."
    )
    p.add_argument("wav_path", type=str, help="Path to wav file")
    p.add_argument("--seconds", type=float, default=None, help="Only plot first N seconds")
    p.add_argument("--sr", type=int, default=None, help="Resample to target sr (default: keep original)")
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=80)
    p.add_argument("--fmin", type=float, default=0.0)
    p.add_argument("--fmax", type=float, default=None, help="Mel fmax (default: sr/2)")
    p.add_argument("--fmax_vis", type=float, default=8000.0, help="Visualization max frequency (Hz). 0 means no limit.")
    p.add_argument("--linear_clip_percentile", type=float, default=99.0, help="Percentile for vmax clipping on linear plots")
    p.add_argument("--vmin_db", type=float, default=-80.0, help="Lower bound (dB) for dB plots")
    p.add_argument("--show_phase", action="store_true", help="Also plot STFT phase")
    p.add_argument("--skip_linear", action="store_true", help="Skip linear-scale spectrograms (magnitude/power)")
    args = p.parse_args()

    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        raise SystemExit(f"File not found: {wav_path}")

    # ---- Load audio (keep channels) ----
    y, sr0 = sf.read(str(wav_path), always_2d=False)
    y = normalize_audio(y)

    # Truncate before resample (original timeline)
    if args.seconds is not None:
        n0 = int(round(args.seconds * sr0))
        y = y[:n0] if y.ndim == 1 else y[:n0, :]

    # Mono for analysis
    y_mono0 = to_mono(y)

    # Resample if requested
    if args.sr is not None and args.sr != sr0:
        y_mono = librosa.resample(y_mono0, orig_sr=sr0, target_sr=args.sr)
        sr = args.sr
    else:
        y_mono = y_mono0
        sr = sr0

    # Truncate after resample (analysis timeline)
    if args.seconds is not None:
        n = int(round(args.seconds * sr))
        y_mono = y_mono[:n]

    # Frequency limits
    fmax = (sr / 2.0) if args.fmax is None else float(args.fmax)
    fmax_vis = float(args.fmax_vis)
    if fmax_vis < 0:
        fmax_vis = 0.0

    # ---- STFT ----
    X = librosa.stft(
        y_mono,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    S_mag = np.abs(X)                       # (F, T)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)  # dB
    phase = np.angle(X)                     # [-pi, pi]

    # ---- Mel ----
    S_mel = librosa.feature.melspectrogram(
        y=y_mono,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=fmax,
        power=2.0,  # power spectrogram
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    # ---- Notes ----
    frame_ms = 1000.0 * args.hop_length / sr
    win_ms = 1000.0 * args.win_length / sr
    dur_s = len(y_mono) / sr
    notes = (
        f"sr={sr} Hz | dur={dur_s:.2f}s | hop={args.hop_length} ({frame_ms:.2f}ms/frame) | "
        f"win={args.win_length} ({win_ms:.2f}ms) | n_fft={args.n_fft} | n_mels={args.n_mels}"
    )

    # ---- Build panels dynamically ----
    panels = []

    def panel_waveform_original(ax):
        if y.ndim == 1:
            t0 = np.arange(len(y)) / sr0
            ax.plot(t0, y, linewidth=0.7)
            ax.set_title("Waveform (mono in file)")
        else:
            t0 = np.arange(y.shape[0]) / sr0
            # Offset channels so you see them separately
            scale = np.percentile(np.abs(y), 99) + 1e-9
            offset = 2.2 * scale
            for c in range(y.shape[1]):
                ax.plot(t0, y[:, c] + c * offset, linewidth=0.7, label=f"ch{c}")
            ax.legend(loc="upper right", frameon=False)
            ax.set_title("Waveform (original channels with offsets) — channels=2 often means stereo")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    panels.append(panel_waveform_original)

    def panel_waveform_mono(ax):
        t = np.arange(len(y_mono)) / sr
        ax.plot(t, y_mono, linewidth=0.7)
        ax.set_title("Waveform (mono used for STFT/mel)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    panels.append(panel_waveform_mono)

    if not args.skip_linear:
        def panel_stft_mag(ax):
            vmax = percentile_clip(S_mag, args.linear_clip_percentile)
            img = librosa.display.specshow(
                S_mag,
                sr=sr,
                hop_length=args.hop_length,
                x_axis="time",
                y_axis="linear",
                ax=ax,
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_title(f"STFT Magnitude (linear, clipped @P{args.linear_clip_percentile:g})")
            if fmax_vis > 0:
                ax.set_ylim(0, min(fmax_vis, sr / 2))
            plt.colorbar(img, ax=ax, format="%.2f")

        panels.append(panel_stft_mag)

    def panel_stft_db(ax):
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=args.hop_length,
            x_axis="time",
            y_axis="linear",
            ax=ax,
            vmin=args.vmin_db,
            vmax=0.0,
        )
        ax.set_title(f"STFT Magnitude (dB) [vmin={args.vmin_db:g} dB]")
        if fmax_vis > 0:
            ax.set_ylim(0, min(fmax_vis, sr / 2))
        plt.colorbar(img, ax=ax, format="%+2.0f dB")

    panels.append(panel_stft_db)

    if args.show_phase:
        def panel_phase(ax):
            img = librosa.display.specshow(
                phase,
                sr=sr,
                hop_length=args.hop_length,
                x_axis="time",
                y_axis="linear",
                ax=ax,
            )
            ax.set_title("STFT Phase (wrapped) — usually not used by mel/log-mel models")
            if fmax_vis > 0:
                ax.set_ylim(0, min(fmax_vis, sr / 2))
            plt.colorbar(img, ax=ax, format="%.2f")

        panels.append(panel_phase)

    if not args.skip_linear:
        def panel_mel_power(ax):
            vmax = percentile_clip(S_mel, args.linear_clip_percentile)
            img = librosa.display.specshow(
                S_mel,
                sr=sr,
                hop_length=args.hop_length,
                x_axis="time",
                y_axis="mel",
                fmin=args.fmin,
                fmax=fmax,
                ax=ax,
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_title(f"Mel Spectrogram (power, clipped @P{args.linear_clip_percentile:g})")
            plt.colorbar(img, ax=ax, format="%.2f")

        panels.append(panel_mel_power)

    def panel_mel_db(ax):
        img = librosa.display.specshow(
            S_mel_db,
            sr=sr,
            hop_length=args.hop_length,
            x_axis="time",
            y_axis="mel",
            fmin=args.fmin,
            fmax=fmax,
            ax=ax,
            vmin=args.vmin_db,
            vmax=0.0,
        )
        ax.set_title(f"Log-Mel Spectrogram (dB) [vmin={args.vmin_db:g} dB] — common ML/TTS feature")
        plt.colorbar(img, ax=ax, format="%+2.0f dB")

    panels.append(panel_mel_db)

    # ---- Plot ----
    nrows = len(panels)
    plt.figure(figsize=(16, 2.5 * nrows))
    plt.suptitle(f"Audio Views\n{notes}", y=0.995)

    for i, fn in enumerate(panels, start=1):
        ax = plt.subplot(nrows, 1, i)
        fn(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main()
