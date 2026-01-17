#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf
import librosa
import librosa.display


def to_mono(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 1 else y.mean(axis=1)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    if np.issubdtype(y.dtype, np.integer):
        maxv = np.iinfo(y.dtype).max
        y = y.astype(np.float32) / float(maxv)
    return y.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Plot waveform + STFT + mel views (robust subplot layout).")
    p.add_argument("wav_path", type=str)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--sr", type=int, default=None, help="Resample to this sr (default: keep original)")
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=80)
    p.add_argument("--fmin", type=float, default=0.0)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--no_phase", action="store_true", help="Skip phase panel")
    p.add_argument("--skip_mel_power", action="store_true", help="Skip raw mel(power) panel; keep log-mel only")
    args = p.parse_args()

    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        raise SystemExit(f"File not found: {wav_path}")

    # Load (keep channels)
    y, sr0 = sf.read(str(wav_path), always_2d=False)
    y = normalize_audio(y)

    # Truncate (before resample)
    if args.seconds is not None:
        n0 = int(round(args.seconds * sr0))
        y = y[:n0] if y.ndim == 1 else y[:n0, :]

    y_mono0 = to_mono(y)

    # Resample mono for analysis
    if args.sr is not None and args.sr != sr0:
        y_mono = librosa.resample(y_mono0, orig_sr=sr0, target_sr=args.sr)
        sr = args.sr
    else:
        y_mono = y_mono0
        sr = sr0

    if args.seconds is not None:
        n = int(round(args.seconds * sr))
        y_mono = y_mono[:n]

    fmax = (sr / 2.0) if args.fmax is None else args.fmax

    # STFT
    X = librosa.stft(
        y_mono,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    S_mag = np.abs(X)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    phase = np.angle(X)

    # Mel
    S_mel = librosa.feature.melspectrogram(
        y=y_mono,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=fmax,
        power=2.0,
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    # Notes
    frame_ms = 1000.0 * args.hop_length / sr
    win_ms = 1000.0 * args.win_length / sr
    dur_s = len(y_mono) / sr
    notes = (
        f"sr={sr} Hz, dur={dur_s:.2f}s | hop={args.hop_length} ({frame_ms:.2f}ms/frame), "
        f"win={args.win_length} ({win_ms:.2f}ms), n_fft={args.n_fft}, n_mels={args.n_mels}"
    )

    # --- Build panels list: each item is (title, plot_fn) ---
    panels = []

    def panel_waveform_original(ax):
        if y.ndim == 1:
            t0 = np.arange(len(y)) / sr0
            ax.plot(t0, y, linewidth=0.7)
            ax.set_title("Waveform (mono in file)")
        else:
            t0 = np.arange(y.shape[0]) / sr0
            scale = np.percentile(np.abs(y), 99) + 1e-9
            offset = 2.2 * scale
            for c in range(y.shape[1]):
                ax.plot(t0, y[:, c] + c * offset, linewidth=0.7, label=f"ch{c}")
            ax.legend(loc="upper right", frameon=False)
            ax.set_title("Waveform (original channels, offsets) — channels=2 usually means stereo")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    panels.append(("waveform_original", panel_waveform_original))

    def panel_waveform_mono(ax):
        t = np.arange(len(y_mono)) / sr
        ax.plot(t, y_mono, linewidth=0.7)
        ax.set_title("Waveform (mono used for STFT/mel)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    panels.append(("waveform_mono", panel_waveform_mono))

    def panel_stft_mag(ax):
        img = librosa.display.specshow(
            S_mag, sr=sr, hop_length=args.hop_length,
            x_axis="time", y_axis="linear", ax=ax
        )
        ax.set_title("STFT Magnitude Spectrogram (linear freq, not log)")
        plt.colorbar(img, ax=ax, format="%.2f")

    panels.append(("stft_mag", panel_stft_mag))

    def panel_stft_db(ax):
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=args.hop_length,
            x_axis="time", y_axis="linear", ax=ax
        )
        ax.set_title("Log-Magnitude Spectrogram (dB, linear freq)")
        plt.colorbar(img, ax=ax, format="%+2.0f dB")

    panels.append(("stft_db", panel_stft_db))

    if not args.no_phase:
        def panel_phase(ax):
            img = librosa.display.specshow(
                phase, sr=sr, hop_length=args.hop_length,
                x_axis="time", y_axis="linear", ax=ax
            )
            ax.set_title("STFT Phase (wrapped) — usually discarded by mel/log-mel")
            plt.colorbar(img, ax=ax, format="%.2f")
        panels.append(("phase", panel_phase))

    if not args.skip_mel_power:
        def panel_mel_power(ax):
            img = librosa.display.specshow(
                S_mel, sr=sr, hop_length=args.hop_length,
                x_axis="time", y_axis="mel",
                fmin=args.fmin, fmax=fmax, ax=ax
            )
            ax.set_title("Mel Spectrogram (power)")
            plt.colorbar(img, ax=ax, format="%.2f")
        panels.append(("mel_power", panel_mel_power))

    def panel_mel_db(ax):
        img = librosa.display.specshow(
            S_mel_db, sr=sr, hop_length=args.hop_length,
            x_axis="time", y_axis="mel",
            fmin=args.fmin, fmax=fmax, ax=ax
        )
        ax.set_title("Log-Mel Spectrogram (dB) — common TTS feature")
        plt.colorbar(img, ax=ax, format="%+2.0f dB")

    panels.append(("mel_db", panel_mel_db))

    # --- Plot with dynamic nrows ---
    nrows = len(panels)
    plt.figure(figsize=(14, 2.6 * nrows))
    plt.suptitle(f"Audio Views (robust layout)\n{notes}", y=0.995)

    for i, (_, fn) in enumerate(panels, start=1):
        ax = plt.subplot(nrows, 1, i)
        fn(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main()
