"""
utils.py
--------
Visualization utilities: waveform, spectrogram, and noise/speech pie chart.
All functions return Matplotlib or Plotly figure objects for use in Streamlit.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────
# COLOUR PALETTE  (consistent across plots)
# ─────────────────────────────────────────────
PALETTE = {
    "speech":     "#00d2aa",   # cyan-teal
    "noise":      "#ff4d6d",   # neon-red
    "original":   "#5a90c0",   # steel blue
    "enhanced":   "#00d2ff",   # electric cyan
    "bg":         "#020610",   # deep space black
    "bg2":        "#050d1e",   # card bg
    "grid":       "#0a1828",   # subtle grid
    "text":       "#c0d8f0",   # cool white
    "accent":     "#7b2dff",   # violet
}


def _mpl_style():
    """Apply a dark, minimal style to all Matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg2"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "grid.color":        PALETTE["grid"],
        "text.color":        PALETTE["text"],
        "font.family":       "monospace",
        "font.size":         9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ─────────────────────────────────────────────
# 1. WAVEFORM PLOT
# ─────────────────────────────────────────────
def plot_waveform(audio_original: np.ndarray,
                  audio_enhanced: np.ndarray,
                  sr: int) -> plt.Figure:
    """
    Side-by-side waveform comparison: original vs enhanced audio.
    Handles the case where ISTFT produces a slightly different length.
    """
    _mpl_style()
    # Align lengths (ISTFT can return N±hop samples)
    min_len = min(len(audio_original), len(audio_enhanced))
    audio_original = audio_original[:min_len]
    audio_enhanced = audio_enhanced[:min_len]

    time = np.linspace(0, min_len / sr, min_len)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    fig.suptitle("Waveform Comparison", color=PALETTE["text"], fontsize=13, fontweight="bold")

    axes[0].plot(time, audio_original, color=PALETTE["original"], linewidth=0.6, alpha=0.85)
    axes[0].set_title("Original Audio", color=PALETTE["text"], fontsize=10)
    axes[0].set_ylabel("Amplitude", color=PALETTE["text"])
    axes[0].set_xlim(0, time[-1])
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time, audio_enhanced, color=PALETTE["enhanced"], linewidth=0.6, alpha=0.85)
    axes[1].set_title("Enhanced Audio", color=PALETTE["text"], fontsize=10)
    axes[1].set_ylabel("Amplitude", color=PALETTE["text"])
    axes[1].set_xlabel("Time (s)", color=PALETTE["text"])
    axes[1].set_xlim(0, time[-1])
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 2. SPECTROGRAM PLOT
# ─────────────────────────────────────────────
def plot_spectrogram(audio: np.ndarray, sr: int,
                     title: str = "Spectrogram",
                     n_fft: int = 1024,
                     hop_length: int = 256) -> plt.Figure:
    """
    Log-power mel spectrogram using librosa.

    Parameters
    ----------
    audio      : audio signal
    sr         : sample rate
    title      : subplot title
    n_fft      : FFT window size
    hop_length : hop size between frames

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _mpl_style()
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr,
                                               n_fft=n_fft,
                                               hop_length=hop_length,
                                               n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length,
                                    x_axis="time", y_axis="mel",
                                    fmax=sr // 2, ax=ax,
                                    cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB",
                 label="Power (dB)")
    ax.set_title(title, color=PALETTE["text"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)", color=PALETTE["text"])
    ax.set_ylabel("Frequency (Hz)", color=PALETTE["text"])
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 3. NOISE vs SPEECH PIE CHART
# ─────────────────────────────────────────────
def generate_noise_speech_chart(speech_ratio: float,
                                 noise_ratio: float) -> go.Figure:
    """
    Interactive Plotly donut chart showing detected speech vs noise percentages.

    Parameters
    ----------
    speech_ratio : float  – 0.0 to 1.0
    noise_ratio  : float  – 0.0 to 1.0

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    labels = ["Speech", "Noise"]
    values = [round(speech_ratio * 100, 1), round(noise_ratio * 100, 1)]
    colors = [PALETTE["speech"], PALETTE["noise"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors,
                    line=dict(color=PALETTE["bg"], width=2)),
        textfont=dict(size=14, color=PALETTE["text"]),
        hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>",
    )])

    fig.update_layout(
        title=dict(text="Speech vs Noise",
                   font=dict(color=PALETTE["text"], size=13,
                             family="JetBrains Mono, monospace"),
                   x=0.5),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        legend=dict(font=dict(color=PALETTE["text"], family="monospace"),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor=PALETTE["grid"]),
        margin=dict(t=50, b=10, l=10, r=10),
        annotations=[dict(
            text=f"<b style='font-size:16px'>{values[0]}%</b><br>speech",
            x=0.5, y=0.5,
            font=dict(size=13, color=PALETTE["text"],
                      family="JetBrains Mono, monospace"),
            showarrow=False,
        )],
    )
    return fig


# ─────────────────────────────────────────────
# 4. FFT FREQUENCY SPECTRUM  (bonus utility)
# ─────────────────────────────────────────────
def plot_fft_spectrum(audio_original: np.ndarray,
                      audio_enhanced: np.ndarray,
                      sr: int) -> go.Figure:
    """
    Overlay FFT magnitude spectra of original and enhanced audio
    using Plotly for interactivity.
    """
    def _spectrum(sig):
        n = len(sig)
        freq = np.fft.rfftfreq(n, d=1.0 / sr)
        mag = np.abs(np.fft.rfft(sig)) / n
        mag_db = 20 * np.log10(mag + 1e-10)
        return freq, mag_db

    # Align lengths before computing spectra
    min_len = min(len(audio_original), len(audio_enhanced))
    freq_o, mag_o = _spectrum(audio_original[:min_len])
    freq_e, mag_e = _spectrum(audio_enhanced[:min_len])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_o, y=mag_o,
                             name="Original",
                             line=dict(color=PALETTE["original"], width=1.2),
                             opacity=0.75))
    fig.add_trace(go.Scatter(x=freq_e, y=mag_e,
                             name="Enhanced",
                             line=dict(color=PALETTE["enhanced"], width=1.5)))

    fig.update_layout(
        title=dict(text="FFT Frequency Spectrum — Original vs Enhanced",
                   font=dict(color=PALETTE["text"], size=12,
                             family="JetBrains Mono, monospace"),
                   x=0.5),
        xaxis=dict(title="Frequency (Hz)", color=PALETTE["text"],
                   gridcolor=PALETTE["grid"], range=[0, sr // 2],
                   tickfont=dict(family="monospace", size=9)),
        yaxis=dict(title="Magnitude (dB)", color=PALETTE["text"],
                   gridcolor=PALETTE["grid"],
                   tickfont=dict(family="monospace", size=9)),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg2"],
        legend=dict(font=dict(color=PALETTE["text"], family="monospace"),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=40, l=55, r=20),
    )
    return fig