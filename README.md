# 🎙️ Speech Enhancement System

DSP + AI-based background noise reduction, voice amplification, and echo cancellation — with a Streamlit web dashboard.

---

## 📁 Folder Structure

```
speech_enhancement_project/
│
├── app.py                 ← Main Streamlit dashboard
├── dsp_processing.py      ← All DSP + AI processing functions
├── about_page.py          ← About / project description tab
├── utils.py               ← Plotting utilities (waveform, spectrogram, pie chart)
├── speech_enhancement.m   ← Equivalent MATLAB implementation
├── requirements.txt       ← Python dependencies
│
├── audio_files/           ← (optional) store test audio files here
└── results/               ← Enhanced output files are saved here
```

---

## ⚙️ Setup & Installation

### 1 — Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `librosa` requires `ffmpeg` to load `.mp3` files.
> Install it via:
> - macOS: `brew install ffmpeg`
> - Ubuntu/Debian: `sudo apt install ffmpeg`
> - Windows: download from https://ffmpeg.org/download.html and add to PATH

---

## ▶️ Running the Dashboard

```bash
streamlit run app.py
```

LIVE URL : **https://sonic-real-time-speech-enhancement-system.streamlit.app/**

---

## 🔬 Processing Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1. Load Audio | `load_audio()` | Load .wav/.mp3, resample to 16 kHz mono |
| 2. Pre-processing | `preprocess_audio()` | Peak-normalise to [-1, 1] |
| 3. VAD | `voice_activity_detection()` | Energy-based speech/noise frame detection |
| 4. FFT Analysis | `noise_estimation()` | Estimate noise profile from STFT |
| 5. Bandpass Filter | `bandpass_filter()` | Butterworth 300–3400 Hz speech band |
| 6. Echo Cancellation | `echo_cancellation()` | Subtract delayed echo estimate |
| 7. AI Enhancement | `ai_speech_enhancement()` | Spectral subtraction + Wiener smoothing |
| 8. Amplification | `amplify_weak_voice()` | Auto-gain to target RMS |

---

## 📊 Dashboard Features

- **About Tab** — Project description, pipeline, and tech stack
- **Upload Tab**
  - Upload .wav or .mp3
  - View speech % / noise % / SNR / duration metrics
  - Original vs enhanced audio players
  - Download enhanced .wav
  - Noise vs speech pie chart (Plotly)
  - FFT spectrum overlay (Plotly)
  - Waveform comparison (Matplotlib)
  - Mel spectrogram comparison (Matplotlib + librosa)

---

## 🧮 MATLAB Script

Run `speech_enhancement.m` in MATLAB R2020a or later (Signal Processing Toolbox required):

1. Place your noisy audio file in the same directory
2. Edit `INPUT_FILE` at the top of the script
3. Run — generates 4 figures + `enhanced_output.wav`

---

## 📦 Key Libraries

| Library | Purpose |
|---------|---------|
| `streamlit` | Web dashboard UI |
| `librosa` | Audio loading, STFT, mel spectrogram |
| `scipy.signal` | Butterworth filter, filtfilt |
| `numpy` | Array math, FFT |
| `soundfile` | Write enhanced WAV files |
| `matplotlib` | Waveform & spectrogram plots |
| `plotly` | Interactive FFT & pie charts |
