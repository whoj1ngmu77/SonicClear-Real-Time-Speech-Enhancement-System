"""
about_page.py  —  About & Pipeline tab (sci-fi aesthetic)
"""
import streamlit as st


def render_about():
    st.markdown("""
    <style>
    /* ── About-specific styles ───────────────────────────────────────────── */
    .about-hero {
        background: linear-gradient(135deg, rgba(0,15,35,0.9), rgba(5,5,20,0.95));
        border: 1px solid rgba(0,210,255,0.1);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .about-hero::before {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(123,45,255,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .about-hero-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #7b2dff;
        margin-bottom: 0.7rem;
    }
    .about-hero-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.1;
        background: linear-gradient(135deg, #ffffff 0%, #b0d0ff 60%, #7b2dff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 1rem;
    }
    .about-hero-desc {
        color: #6a8fb0;
        font-size: 0.95rem;
        line-height: 1.75;
        max-width: 700px;
        font-weight: 300;
    }

    /* ── Feature Cards ───────────────────────────────────────────────────── */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, rgba(0,12,30,0.9), rgba(3,6,18,0.95));
        border: 1px solid rgba(0,210,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: rgba(0,210,255,0.22);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .feature-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: var(--top-line, linear-gradient(90deg, transparent, rgba(0,210,255,0.2), transparent));
    }
    .feature-emoji {
        font-size: 1.8rem;
        margin-bottom: 0.7rem;
        display: block;
    }
    .feature-name {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #c0d8f0;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .feature-desc {
        font-size: 0.82rem;
        color: #4a6a8a;
        line-height: 1.6;
    }

    /* ── Pipeline ────────────────────────────────────────────────────────── */
    .pipeline-wrap {
        background: linear-gradient(135deg, rgba(0,12,30,0.9), rgba(3,6,18,0.95));
        border: 1px solid rgba(0,210,255,0.08);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin: 1.5rem 0;
    }
    .pipeline-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(0,210,255,0.05);
        animation: fadein 0.4s ease forwards;
        animation-delay: var(--d, 0s);
        opacity: 0;
    }
    .pipeline-step:last-child { border-bottom: none; }
    @keyframes fadein { to { opacity: 1; } }
    .ps-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #00d2ff;
        background: rgba(0,210,255,0.07);
        border: 1px solid rgba(0,210,255,0.15);
        border-radius: 6px;
        padding: 0.2rem 0.5rem;
        white-space: nowrap;
        flex-shrink: 0;
        margin-top: 0.1rem;
    }
    .ps-arrow {
        color: #1a3a5a;
        font-size: 0.8rem;
        flex-shrink: 0;
        margin-top: 0.15rem;
    }
    .ps-name {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #a0c0e0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .ps-desc {
        font-size: 0.8rem;
        color: #3a5a7a;
        margin-top: 0.1rem;
        line-height: 1.5;
    }

    /* ── Tech Stack ──────────────────────────────────────────────────────── */
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin: 1.5rem 0;
    }
    .tech-pill {
        background: rgba(0,10,25,0.8);
        border: 1px solid rgba(0,210,255,0.1);
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .tech-icon {
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    .tech-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        color: #2a4a6a;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .tech-value {
        font-size: 0.85rem;
        color: #a0c0e0;
        font-weight: 600;
        margin-top: 0.1rem;
    }

    /* ── Reuse sec-title from app.py ─────────────────────────────────────── */
    .sec-title {
        display: flex; align-items: center; gap: 0.7rem; margin: 2rem 0 1rem;
    }
    .sec-title-line {
        flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(0,210,255,0.3), transparent);
    }
    .sec-title-text {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        font-weight: 700; color: #00d2ff; text-transform: uppercase;
        letter-spacing: 0.15em; white-space: nowrap;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="about-hero">
      <div class="about-hero-tag">◆ DSP + AI · Academic Research Project</div>
      <h1 class="about-hero-title">Speech Enhancement<br>System</h1>
      <p class="about-hero-desc">
        A multi-stage signal processing and AI pipeline that takes noisy real-world audio
        — contaminated by traffic, crowd noise, echo, or faint speech — and outputs
        clean, intelligible speech. Designed to simulate the technology inside
        call-centre noise suppression, hearing aids, and voice assistant preprocessing.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Features ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-title">
      <span class="sec-title-text">◈ Key Features</span>
      <div class="sec-title-line"></div>
    </div>""", unsafe_allow_html=True)

    features = [
        ("🔇", "Noise Suppression",     "noisereduce (stationary + non-stationary) removes traffic, crowd, HVAC backgrounds",
         "linear-gradient(90deg,#00d2ff,#0060ff)"),
        ("🔊", "Voice Amplification",   "Global RMS gain + dynamic compression boosts faint speakers without clipping",
         "linear-gradient(90deg,#00d2aa,#00a870)"),
        ("🗣️", "VAD Detection",         "Energy + ZCR-based frame classification measures speech vs noise percentage",
         "linear-gradient(90deg,#7b2dff,#4a00c0)"),
        ("↩️", "Echo Cancellation",     "LMS adaptive filter continuously learns and subtracts the echo path",
         "linear-gradient(90deg,#ff4d6d,#c0003a)"),
        ("📊", "Spectral Analysis",      "Interactive FFT spectrum and mel spectrogram before/after comparison",
         "linear-gradient(90deg,#ffb400,#ff6600)"),
        ("🎚️", "VAD Gating",            "Soft-mutes silent gaps post-suppression to eliminate residual bleed",
         "linear-gradient(90deg,#00d2ff,#7b2dff)"),
    ]

    cols = st.columns(3)
    for i, (icon, name, desc, line) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card" style="--top-line:{line}">
              <span class="feature-emoji">{icon}</span>
              <div class="feature-name">{name}</div>
              <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-title">
      <span class="sec-title-text">◈ Processing Pipeline</span>
      <div class="sec-title-line"></div>
    </div>""", unsafe_allow_html=True)

    steps = [
        ("01", "Load Audio",           "Read .wav / .mp3, stereo → mono, resample to 16 kHz"),
        ("02", "Pre-Processing",       "Peak-normalise to [-1, 1] for consistent amplitude"),
        ("03", "VAD Analysis",         "Energy + ZCR per 20 ms frame → speech % / noise %"),
        ("04", "Bandpass Filter",      "Butterworth 80–7500 Hz — removes sub-bass rumble and ultrasonic hiss"),
        ("05", "Echo Cancellation",    "LMS adaptive filter models and subtracts room echo"),
        ("06", "BG Noise Removal ×2",  "noisereduce pass 1 (stationary) → pass 2 (non-stationary) · 100% prop decrease"),
        ("07", "VAD Gating",           "Soft-mute non-speech frames (gain floor 0.02) to kill residual noise"),
        ("08", "Voice Amplification",  "Global gain → dynamic compression → soft tanh clip → 5% headroom"),
    ]

    delays = ["0.05s","0.10s","0.15s","0.20s","0.25s","0.30s","0.35s","0.40s"]
    steps_html = ""
    for (num, name, desc), d in zip(steps, delays):
        steps_html += f"""
        <div class="pipeline-step" style="--d:{d}">
          <span class="ps-num">{num}</span>
          <span class="ps-arrow">▶</span>
          <div>
            <div class="ps-name">{name}</div>
            <div class="ps-desc">{desc}</div>
          </div>
        </div>"""

    st.markdown(f'<div class="pipeline-wrap">{steps_html}</div>', unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-title">
      <span class="sec-title-text">◈ Technology Stack</span>
      <div class="sec-title-line"></div>
    </div>""", unsafe_allow_html=True)

    tech = [
        ("🐍", "Language",       "Python 3.10+"),
        ("🖥️", "UI Framework",  "Streamlit"),
        ("🎵", "Audio I/O",      "librosa · soundfile"),
        ("🔬", "DSP Engine",     "scipy.signal · numpy"),
        ("🤖", "AI Denoise",     "noisereduce ≥ 3.0"),
        ("📊", "Visualisation",  "matplotlib · plotly"),
    ]

    st.markdown('<div class="tech-grid">', unsafe_allow_html=True)
    tech_html = ""
    for icon, label, value in tech:
        tech_html += f"""
        <div class="tech-pill">
          <span class="tech-icon">{icon}</span>
          <div>
            <div class="tech-label">{label}</div>
            <div class="tech-value">{value}</div>
          </div>
        </div>"""
    st.markdown(f'<div class="tech-grid">{tech_html}</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center; color:#1a3050; font-size:0.78rem;
              font-family:'JetBrains Mono',monospace; margin-top:3rem;
              letter-spacing:0.1em">
      SONICCLEAR · DSP + AI SPEECH ENHANCEMENT · ACADEMIC PROJECT
    </p>""", unsafe_allow_html=True)