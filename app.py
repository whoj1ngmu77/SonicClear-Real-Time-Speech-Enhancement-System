"""
app.py  —  Speech Enhancement System Dashboard
Dark sci-fi / audio-lab aesthetic with glassmorphism, animated waveforms,
glow effects, and cinematic layout.
Run: streamlit run app.py
"""

import os, io, tempfile
import numpy as np
import streamlit as st
import soundfile as sf

from dsp_processing import run_full_pipeline
from utils import plot_waveform, plot_spectrogram, generate_noise_speech_chart, plot_fft_spectrum
from about_page import render_about

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SonicClear · Speech Enhancement",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

/* ═══ RESET & BASE ═══════════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background: #020610 !important;
    color: #c8d8f0;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ═══ ANIMATED HERO HEADER ══════════════════════════════════════════════════ */
.hero-wrap {
    position: relative;
    width: 100%;
    overflow: hidden;
    background: linear-gradient(160deg, #020a18 0%, #040d1f 40%, #060818 100%);
    padding: 3.5rem 4rem 2.5rem;
    border-bottom: 1px solid rgba(0,210,255,0.12);
}

/* Animated grid lines */
.hero-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,210,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,210,255,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: gridscroll 20s linear infinite;
}
@keyframes gridscroll {
    0%   { transform: translateY(0); }
    100% { transform: translateY(40px); }
}

/* Glowing orbs */
.hero-wrap::after {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,180,255,0.12) 0%, transparent 70%);
    animation: orb1 8s ease-in-out infinite alternate;
}
@keyframes orb1 {
    0%   { transform: translate(0,0); opacity: 0.6; }
    100% { transform: translate(80px,40px); opacity: 1; }
}

.hero-inner {
    position: relative;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}

.hero-left {}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0,210,255,0.08);
    border: 1px solid rgba(0,210,255,0.25);
    border-radius: 100px;
    padding: 0.25rem 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #00d2ff;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero-badge-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00d2ff;
    animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.7); }
}

.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1.05;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #a8d8ff 50%, #00d2ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.02em;
}

.hero-sub {
    font-size: 0.95rem;
    color: #6a8fb0;
    font-weight: 300;
    letter-spacing: 0.03em;
    margin: 0;
}

/* Animated sound bars on the right */
.soundbars {
    display: flex;
    align-items: flex-end;
    gap: 4px;
    height: 60px;
}
.soundbars span {
    display: block;
    width: 5px;
    background: linear-gradient(to top, #00d2ff, #7b2dff);
    border-radius: 3px 3px 0 0;
    animation: bar var(--dur, 1s) ease-in-out infinite alternate;
    opacity: 0.85;
}
@keyframes bar {
    from { height: var(--min, 8px); }
    to   { height: var(--max, 50px); }
}

/* ═══ NAV TABS ═══════════════════════════════════════════════════════════════ */
.stTabs {
    padding: 0 4rem;
    background: #020610;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid rgba(0,210,255,0.1);
    padding-top: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #4a6a8a;
    font-family: 'Exo 2', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 0.7rem 1.8rem;
    letter-spacing: 0.04em;
    border-bottom: 2px solid transparent !important;
    transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #a0c8e8 !important;
    background: rgba(0,210,255,0.03) !important;
}
.stTabs [aria-selected="true"] {
    color: #00d2ff !important;
    border-bottom: 2px solid #00d2ff !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 2.5rem 4rem 4rem;
}

/* ═══ SECTION TITLE ══════════════════════════════════════════════════════════ */
.sec-title {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 2rem 0 1rem;
}
.sec-title-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,210,255,0.3), transparent);
}
.sec-title-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: #00d2ff;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    white-space: nowrap;
}

/* ═══ UPLOAD ZONE ════════════════════════════════════════════════════════════ */
.upload-zone-wrap {
    background: linear-gradient(135deg, rgba(0,30,60,0.6), rgba(10,10,30,0.8));
    border: 1.5px dashed rgba(0,210,255,0.25);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.upload-zone-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,210,255,0.5), transparent);
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0%   { left: -60%; }
    100% { left: 160%; }
}
.upload-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
    display: block;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-8px); }
}
.upload-text { color: #a0b8d0; font-size: 0.95rem; }
.upload-hint { color: #3a5a7a; font-size: 0.8rem; margin-top: 0.3rem; }

/* Hide default uploader UI and overlay with ours */
.stFileUploader { position: relative; }
.stFileUploader > div {
    background: transparent !important;
    border: none !important;
}
.stFileUploader label { display: none !important; }

/* ═══ METRIC CARDS ═══════════════════════════════════════════════════════════ */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: linear-gradient(135deg, rgba(0,20,40,0.9), rgba(5,10,25,0.95));
    border: 1px solid rgba(0,210,255,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, border-color 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(0,210,255,0.3);
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg, #00d2ff, #7b2dff));
    border-radius: 0 0 16px 16px;
}
.metric-icon {
    font-size: 1.5rem;
    margin-bottom: 0.4rem;
    display: block;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3a6080;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.metric-sub {
    font-size: 0.75rem;
    color: #3a5a7a;
}

/* ═══ AUDIO COMPARE CARDS ════════════════════════════════════════════════════ */
.audio-card {
    background: linear-gradient(135deg, rgba(0,15,35,0.95), rgba(5,8,20,0.98));
    border: 1px solid rgba(0,210,255,0.1);
    border-radius: 20px;
    padding: 1.8rem;
    position: relative;
    overflow: hidden;
}
.audio-card-orig { border-color: rgba(100,160,255,0.15); }
.audio-card-enh  { border-color: rgba(0,210,180,0.2); }
.audio-card-enh::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,210,150,0.08) 0%, transparent 70%);
}
.audio-card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.audio-label-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}

/* ═══ ENGINE BADGE ═══════════════════════════════════════════════════════════ */
.engine-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0,210,150,0.08);
    border: 1px solid rgba(0,210,150,0.25);
    border-radius: 100px;
    padding: 0.4rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #00d2aa;
    margin: 0.5rem 0 1.5rem;
}
.engine-badge-warn {
    background: rgba(255,180,0,0.08);
    border-color: rgba(255,180,0,0.25);
    color: #ffb400;
}

/* ═══ CHART CARDS ════════════════════════════════════════════════════════════ */
.chart-card {
    background: linear-gradient(135deg, rgba(0,12,28,0.95), rgba(3,6,18,0.98));
    border: 1px solid rgba(0,210,255,0.08);
    border-radius: 20px;
    padding: 1.5rem 1.8rem 1.8rem;
    margin-bottom: 1.5rem;
}
.chart-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #a0c0e0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.chart-title-icon { font-size: 1rem; }

/* ═══ DOWNLOAD BUTTON ════════════════════════════════════════════════════════ */
.stDownloadButton > button {
    background: linear-gradient(135deg, #00b890, #00d2ff) !important;
    color: #020610 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(0,210,200,0.2) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,210,200,0.35) !important;
}

/* ═══ PROCESSING SPINNER ═════════════════════════════════════════════════════ */
.stSpinner > div {
    border-color: #00d2ff transparent transparent !important;
}

/* ═══ EMPTY STATE ════════════════════════════════════════════════════════════ */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
}
.empty-icon {
    font-size: 4rem;
    display: block;
    margin-bottom: 1rem;
    opacity: 0.4;
    animation: float 3s ease-in-out infinite;
}
.empty-text {
    color: #2a4a6a;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* ═══ PROCESSING STEPS ANIMATION ═════════════════════════════════════════════*/
.steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin: 1.5rem 0;
}
.step-pill {
    background: rgba(0,15,35,0.8);
    border: 1px solid rgba(0,210,255,0.1);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.82rem;
    color: #5a80a0;
    animation: stepin 0.4s ease forwards;
    animation-delay: var(--d, 0s);
    opacity: 0;
    transform: translateY(6px);
}
@keyframes stepin {
    to { opacity: 1; transform: translateY(0); }
}
.step-pill.done {
    border-color: rgba(0,210,150,0.25);
    color: #00d2aa;
    background: rgba(0,50,40,0.3);
}
.step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #00d2ff;
    background: rgba(0,210,255,0.08);
    border-radius: 50%;
    width: 22px; height: 22px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

/* ═══ DIVIDER ════════════════════════════════════════════════════════════════ */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,210,255,0.2), rgba(123,45,255,0.15), transparent);
    margin: 2rem 0;
    border: none;
}

/* ═══ STREAMLIT OVERRIDES ════════════════════════════════════════════════════ */
.stAudio audio {
    width: 100%;
    border-radius: 10px;
    filter: invert(0.85) hue-rotate(180deg);
}
.stPlotlyChart { border-radius: 16px; overflow: hidden; }
.stFileUploader [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── ANIMATED HERO HEADER ──────────────────────────────────────────────────────
bar_configs = [
    (12, 45, "0.3s"), (8, 55, "0.7s"), (15, 60, "0.5s"), (6, 40, "1.1s"),
    (18, 52, "0.4s"), (10, 48, "0.9s"), (14, 58, "0.6s"), (7, 35, "1.3s"),
    (20, 62, "0.2s"), (9, 44, "0.8s"), (16, 56, "0.35s"),(11, 50, "1.0s"),
]
bars_html = "".join(
    f'<span style="--min:{mn}px;--max:{mx}px;--dur:{dur};animation-delay:{dur}"></span>'
    for mn, mx, dur in bar_configs
)

st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-inner">
    <div class="hero-left">
      <div class="hero-badge">
        <span class="hero-badge-dot"></span>
        DSP · AI · Real-time Processing
      </div>
      <h1 class="hero-title">SonicClear<br>Enhancement Lab</h1>
      <p class="hero-sub">Background Noise Removal · Voice Amplification · Echo Cancellation</p>
    </div>
    <div class="soundbars">{bars_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab_upload, tab_about = st.tabs(["⚡  Enhance Audio", "◎  About & Pipeline"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & ENHANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:

    # ── Upload Zone ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-title">
      <span class="sec-title-text">01 · Input Audio</span>
      <div class="sec-title-line"></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone-wrap">
      <span class="upload-icon">🎵</span>
      <div class="upload-text">Drop your audio file here or click to browse</div>
      <div class="upload-hint">Supports .WAV · .MP3 &nbsp;|&nbsp; Stereo auto-converted to 16 kHz mono</div>
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        " ", type=["wav", "mp3"], label_visibility="collapsed"
    )

    # ── Empty state ───────────────────────────────────────────────────────────
    if uploaded_file is None:
        st.markdown("""
        <div class="empty-state">
          <span class="empty-icon">🎙️</span>
          <div class="empty-text">Upload a noisy audio file above to begin enhancement</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Save & process ────────────────────────────────────────────────────────
    suffix = ".wav" if uploaded_file.name.lower().endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        input_path = tmp.name

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"enhanced_{uploaded_file.name.replace('.mp3', '.wav')}")

    # Animated processing steps
    st.markdown("""
    <div class="sec-title" style="margin-top:2rem">
      <span class="sec-title-text">02 · Processing Pipeline</span>
      <div class="sec-title-line"></div>
    </div>
    <div class="steps-grid">
      <div class="step-pill done" style="--d:0.05s"><span class="step-num">1</span> Load &amp; Resample</div>
      <div class="step-pill done" style="--d:0.10s"><span class="step-num">2</span> Normalise</div>
      <div class="step-pill done" style="--d:0.15s"><span class="step-num">3</span> VAD Analysis</div>
      <div class="step-pill done" style="--d:0.20s"><span class="step-num">4</span> Bandpass Filter</div>
      <div class="step-pill done" style="--d:0.25s"><span class="step-num">5</span> Echo Cancel</div>
      <div class="step-pill done" style="--d:0.30s"><span class="step-num">6</span> BG Noise Removal</div>
      <div class="step-pill done" style="--d:0.35s"><span class="step-num">7</span> 2nd Pass Denoise</div>
      <div class="step-pill done" style="--d:0.40s"><span class="step-num">8</span> VAD Gate</div>
      <div class="step-pill done" style="--d:0.45s"><span class="step-num">9</span> Voice Amplify</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔬  Running enhancement pipeline…"):
        try:
            result = run_full_pipeline(input_path, output_path=output_path)
        except Exception as exc:
            st.error(f"❌ Processing error: {exc}")
            st.stop()

    audio_original = result["audio_original"]
    audio_enhanced = result["audio_enhanced"]
    sr             = result["sr"]
    speech_pct     = round(result["speech_ratio"] * 100, 1)
    noise_pct      = round(result["noise_ratio"]  * 100, 1)
    engine_used    = result.get("engine_used", "DSP fallback")
    duration_s     = len(audio_original) / sr

    _n = min(len(audio_original), len(audio_enhanced))
    _o, _e = audio_original[:_n], audio_enhanced[:_n]
    snr_db = 10 * np.log10(np.mean(_o**2) / (np.var(_o - _e) + 1e-10))

    # Engine badge
    if "noisereduce" in engine_used:
        st.markdown(f'<div class="engine-badge">✦ Active Engine: {engine_used}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="engine-badge engine-badge-warn">⚠ DSP Fallback · '
            'Run <code>pip install noisereduce</code> for best quality</div>',
            unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-title">
      <span class="sec-title-text">03 · Analysis Metrics</span>
      <div class="sec-title-line"></div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-card" style="--accent: linear-gradient(90deg,#00d2ff,#0080ff)">
        <span class="metric-icon">🗣️</span>
        <div class="metric-label">Speech Detected</div>
        <div class="metric-value" style="color:#00d2ff">{speech_pct}%</div>
        <div class="metric-sub">of total frames</div>
      </div>
      <div class="metric-card" style="--accent: linear-gradient(90deg,#ff4d6d,#ff8c42)">
        <span class="metric-icon">🔊</span>
        <div class="metric-label">Noise Detected</div>
        <div class="metric-value" style="color:#ff4d6d">{noise_pct}%</div>
        <div class="metric-sub">background noise</div>
      </div>
      <div class="metric-card" style="--accent: linear-gradient(90deg,#7b2dff,#00d2ff)">
        <span class="metric-icon">📈</span>
        <div class="metric-label">SNR Gain</div>
        <div class="metric-value" style="color:#a06fff">{snr_db:.1f} dB</div>
        <div class="metric-sub">improvement</div>
      </div>
      <div class="metric-card" style="--accent: linear-gradient(90deg,#00d2aa,#00b890)">
        <span class="metric-icon">⏱️</span>
        <div class="metric-label">Duration</div>
        <div class="metric-value" style="color:#00d2aa">{duration_s:.1f}s</div>
        <div class="metric-sub">@ {sr:,} Hz</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Audio Compare ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-title">
      <span class="sec-title-text">04 · Before / After</span>
      <div class="sec-title-line"></div>
    </div>""", unsafe_allow_html=True)

    col_o, col_e = st.columns(2)

    with col_o:
        st.markdown("""
        <div class="audio-card audio-card-orig">
          <div class="audio-card-label" style="color:#5a90c0">
            <span class="audio-label-dot" style="background:#5a90c0"></span>
            Original — Noisy Input
          </div>
        </div>""", unsafe_allow_html=True)
        st.audio(input_path)

    with col_e:
        st.markdown("""
        <div class="audio-card audio-card-enh">
          <div class="audio-card-label" style="color:#00d2aa">
            <span class="audio-label-dot" style="background:#00d2aa; box-shadow:0 0 8px #00d2aa"></span>
            Enhanced — Noise Removed ✦
          </div>
        </div>""", unsafe_allow_html=True)
        enhanced_buf = io.BytesIO()
        sf.write(enhanced_buf, audio_enhanced, sr, format="WAV")
        enhanced_buf.seek(0)
        st.audio(enhanced_buf, format="audio/wav")

    # Download button
    st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
    enhanced_buf.seek(0)
    st.download_button(
        label="⬇  Download Enhanced Audio (.wav)",
        data=enhanced_buf,
        file_name=f"enhanced_{uploaded_file.name.replace('.mp3','.wav')}",
        mime="audio/wav",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

    # ── Charts Row 1: Pie + FFT ───────────────────────────────────────────────
    st.markdown("""
    <div class="sec-title">
      <span class="sec-title-text">05 · Frequency & Composition Analysis</span>
      <div class="sec-title-line"></div>
    </div>""", unsafe_allow_html=True)

    ch1, ch2 = st.columns([1, 1.8])

    with ch1:
        st.markdown('<div class="chart-card"><div class="chart-title"><span class="chart-title-icon">◔</span> Speech vs Noise</div>', unsafe_allow_html=True)
        pie_fig = generate_noise_speech_chart(result["speech_ratio"], result["noise_ratio"])
        st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="chart-card"><div class="chart-title"><span class="chart-title-icon">≋</span> FFT Frequency Spectrum</div>', unsafe_allow_html=True)
        fft_fig = plot_fft_spectrum(audio_original, audio_enhanced, sr)
        st.plotly_chart(fft_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Waveform ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-title">
      <span class="sec-title-text">06 · Waveform Comparison</span>
      <div class="sec-title-line"></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    wave_fig = plot_waveform(audio_original, audio_enhanced, sr)
    st.pyplot(wave_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Spectrograms ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-title">
      <span class="sec-title-text">07 · Spectrogram Analysis</span>
      <div class="sec-title-line"></div>
    </div>""", unsafe_allow_html=True)

    sp1, sp2 = st.columns(2)
    with sp1:
        st.markdown('<div class="chart-card"><div class="chart-title">◈ Original Spectrogram</div>', unsafe_allow_html=True)
        st.pyplot(plot_spectrogram(audio_original, sr, title="Original"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with sp2:
        st.markdown('<div class="chart-card"><div class="chart-title">◈ Enhanced Spectrogram</div>', unsafe_allow_html=True)
        st.pyplot(plot_spectrogram(audio_enhanced, sr, title="Enhanced"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    try:
        os.unlink(input_path)
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    render_about()