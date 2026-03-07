% =========================================================================
% speech_enhancement.m
% =========================================================================
% Complete MATLAB Speech Enhancement Pipeline
%
% Pipeline:
%   Audio Input → Pre-processing → VAD → FFT Analysis → Noise Detection
%   → Bandpass Filtering → Voice Amplification → Echo Cancellation
%   → Spectral Subtraction → Spectrogram → SNR Evaluation → Output
%
% Usage:
%   1. Place your noisy audio file in the same folder and update INPUT_FILE.
%   2. Run the script.  Plots and enhanced_output.wav will be generated.
%
% Dependencies: Signal Processing Toolbox (butter, filtfilt, spectrogram)
% =========================================================================

clear; clc; close all;

%% ── CONFIGURATION ────────────────────────────────────────────────────────
INPUT_FILE    = 'noisy_speech.wav';   % <<< Change to your input file
OUTPUT_FILE   = 'enhanced_output.wav';
TARGET_SR     = 16000;   % Resample to 16 kHz
BP_LOW        = 300;     % Bandpass lower cut-off  (Hz)
BP_HIGH       = 3400;    % Bandpass upper cut-off  (Hz)
ECHO_DELAY_MS = 100;     % Estimated echo delay    (ms)
ECHO_ATTEN    = 0.4;     % Echo attenuation factor
OVER_SUBTRACT = 1.5;     % Spectral subtraction over-subtraction factor
SPEC_FLOOR    = 0.002;   % Spectral floor (fraction of magnitude)
TARGET_RMS    = 0.1;     % Target RMS after amplification

%% =========================================================================
%% STEP 1 — AUDIO INPUT
%% =========================================================================
fprintf('[1/10] Loading audio: %s\n', INPUT_FILE);
[audio_raw, sr_orig] = audioread(INPUT_FILE);

% Convert stereo → mono
if size(audio_raw, 2) > 1
    audio_raw = mean(audio_raw, 2);
end

% Resample to target sample rate
if sr_orig ~= TARGET_SR
    audio_raw = resample(audio_raw, TARGET_SR, sr_orig);
end
sr = TARGET_SR;
audio_raw = audio_raw(:);   % ensure column vector

%% =========================================================================
%% STEP 2 — PRE-PROCESSING  (normalisation)
%% =========================================================================
fprintf('[2/10] Pre-processing...\n');
audio = audio_raw / (max(abs(audio_raw)) + eps);
audio_original = audio;   % keep a copy for comparison

%% =========================================================================
%% STEP 3 — VOICE ACTIVITY DETECTION  (VAD)
%% =========================================================================
fprintf('[3/10] Voice Activity Detection...\n');
frame_len  = round(sr * 0.030);   % 30 ms frames
hop_len    = frame_len;
n_frames   = floor(length(audio) / hop_len);

rms_db = zeros(n_frames, 1);
for k = 1:n_frames
    idx   = (k-1)*hop_len + 1 : min(k*hop_len, length(audio));
    frame = audio(idx);
    rms_db(k) = 20 * log10(rms(frame) + eps);
end

% Adaptive threshold: 30th-percentile quiet level + 10 dB
quiet_thr = prctile(rms_db, 30) + 10;
vad_labels = rms_db > quiet_thr;   % 1 = speech, 0 = noise

speech_pct = 100 * mean(vad_labels);
noise_pct  = 100 - speech_pct;
fprintf('   Speech: %.1f %%   |   Noise: %.1f %%\n', speech_pct, noise_pct);

%% =========================================================================
%% STEP 4 — FFT FREQUENCY ANALYSIS
%% =========================================================================
fprintf('[4/10] FFT Analysis...\n');
N    = length(audio);
freq = (0:floor(N/2)) * sr / N;
mag  = abs(fft(audio));
mag  = mag(1:floor(N/2)+1) / N;
mag_db = 20 * log10(mag + eps);

%% =========================================================================
%% STEP 5 — NOISE ESTIMATION  (STFT-based)
%% =========================================================================
fprintf('[5/10] Noise estimation...\n');
nfft     = 1024;
hop_stft = 256;
win      = hann(nfft);

% Compute STFT
[S, F, T] = spectrogram(audio, win, nfft - hop_stft, nfft, sr);
magnitude = abs(S);
phase_mtx = angle(S);

% Noise profile: mean magnitude of quietest 20 % frames
frame_energy  = mean(magnitude, 1);
noise_thr_val = prctile(frame_energy, 20);
noise_frames  = magnitude(:, frame_energy <= noise_thr_val);
if isempty(noise_frames)
    noise_profile = mean(magnitude, 2);
else
    noise_profile = mean(noise_frames, 2);
end

%% =========================================================================
%% STEP 6 — BANDPASS FILTER  (300 – 3400 Hz)
%% =========================================================================
fprintf('[6/10] Bandpass filtering...\n');
nyq = sr / 2;
[b_bp, a_bp] = butter(4, [BP_LOW BP_HIGH] / nyq, 'bandpass');
audio = filtfilt(b_bp, a_bp, audio);

%% =========================================================================
%% STEP 7 — ECHO CANCELLATION
%% =========================================================================
fprintf('[7/10] Echo cancellation...\n');
delay_samp = round(sr * ECHO_DELAY_MS / 1000);
echo_sig   = zeros(size(audio));
echo_sig(delay_samp+1:end) = audio(1:end-delay_samp) * ECHO_ATTEN;
audio = audio - echo_sig;
audio = audio / (max(abs(audio)) + eps);

%% =========================================================================
%% STEP 8 — SPECTRAL SUBTRACTION  (AI-style enhancement)
%% =========================================================================
fprintf('[8/10] Spectral subtraction (AI enhancement)...\n');
[S2, ~, ~] = spectrogram(audio, win, nfft - hop_stft, nfft, sr);
mag2   = abs(S2);
phase2 = angle(S2);

% Re-estimate noise profile on filtered signal
frame_en2 = mean(mag2, 1);
noise_thr2 = prctile(frame_en2, 20);
noise_fr2  = mag2(:, frame_en2 <= noise_thr2);
if isempty(noise_fr2)
    np2 = mean(mag2, 2);
else
    np2 = mean(noise_fr2, 2);
end

subtracted = mag2 - OVER_SUBTRACT * repmat(np2, 1, size(mag2, 2));
subtracted = max(subtracted, SPEC_FLOOR * mag2);

gain = subtracted ./ (mag2 + eps);

% Smooth gain over time (5-frame moving average)
gain_smooth = movmean(gain, 5, 2);

S_enhanced = gain_smooth .* mag2 .* exp(1j * phase2);

% Reconstruct via overlap-add
audio_enh = real(istft_ola(S_enhanced, nfft, hop_stft, length(audio)));
audio_enh = audio_enh / (max(abs(audio_enh)) + eps);

%% =========================================================================
%% STEP 8b — VOICE AMPLIFICATION
%% =========================================================================
fprintf('[8b] Voice amplification...\n');
cur_rms = rms(audio_enh);
gain_amp = min(TARGET_RMS / (cur_rms + eps), 10.0);
audio_enh = tanh(audio_enh * gain_amp);   % apply + soft-clip

%% =========================================================================
%% STEP 9 — VISUALISATIONS
%% =========================================================================
fprintf('[9/10] Generating plots...\n');

% ── Figure 1: Waveform Comparison ─────────────────────────────────────────
t_orig = (0:length(audio_original)-1) / sr;
t_enh  = (0:length(audio_enh)-1)      / sr;

figure('Name','Waveform Comparison','Color','#0E1117','Position',[100 600 900 400]);
subplot(2,1,1);
  plot(t_orig, audio_original, 'Color', '#4A90D9', 'LineWidth', 0.6);
  set(gca,'Color','#0E1117','XColor','#C0C8D8','YColor','#C0C8D8', ...
      'GridColor','#2A2D35'); grid on;
  title('Original Audio','Color','#E0E0E0','FontSize',11);
  xlabel('Time (s)','Color','#C0C8D8'); ylabel('Amplitude','Color','#C0C8D8');

subplot(2,1,2);
  plot(t_enh, audio_enh, 'Color', '#00C9A7', 'LineWidth', 0.6);
  set(gca,'Color','#0E1117','XColor','#C0C8D8','YColor','#C0C8D8', ...
      'GridColor','#2A2D35'); grid on;
  title('Enhanced Audio','Color','#E0E0E0','FontSize',11);
  xlabel('Time (s)','Color','#C0C8D8'); ylabel('Amplitude','Color','#C0C8D8');

% ── Figure 2: FFT Frequency Spectrum ──────────────────────────────────────
N2      = length(audio_enh);
freq2   = (0:floor(N2/2)) * sr / N2;
mag_enh = abs(fft(audio_enh));
mag_enh = mag_enh(1:floor(N2/2)+1) / N2;
mag_enh_db = 20 * log10(mag_enh + eps);

figure('Name','FFT Spectrum','Color','#0E1117','Position',[100 150 900 350]);
hold on;
  plot(freq, mag_db, 'Color', '#4A90D9', 'LineWidth', 1.2, ...
       'DisplayName','Original');
  plot(freq2, mag_enh_db, 'Color', '#00C9A7', 'LineWidth', 1.5, ...
       'DisplayName','Enhanced');
hold off;
set(gca,'Color','#0E1117','XColor','#C0C8D8','YColor','#C0C8D8', ...
    'GridColor','#2A2D35'); grid on;
legend('Location','northeast','TextColor','#E0E0E0','Color','#131720');
title('FFT Frequency Spectrum — Original vs Enhanced', ...
      'Color','#E0E0E0','FontSize',12);
xlabel('Frequency (Hz)','Color','#C0C8D8');
ylabel('Magnitude (dB)','Color','#C0C8D8');
xlim([0, sr/2]);

% ── Figure 3: Noise vs Speech Pie Chart ───────────────────────────────────
figure('Name','Speech vs Noise','Color','#0E1117','Position',[1050 600 420 380]);
pie_vals = [speech_pct, noise_pct];
pie_lbls = {sprintf('Speech\n%.1f%%', speech_pct), ...
            sprintf('Noise\n%.1f%%',  noise_pct)};
h = pie(pie_vals, pie_lbls);
colormap([0 0.788 0.655; 1 0.420 0.420]);
for k = 2:2:length(h)
    h(k).Color      = '#E0E0E0';
    h(k).FontSize   = 11;
    h(k).FontWeight = 'bold';
end
title('Speech vs Noise Distribution','Color','#E0E0E0','FontSize',12);
set(gca,'Color','#0E1117');

% ── Figure 4: Spectrograms ─────────────────────────────────────────────────
figure('Name','Spectrograms','Color','#0E1117','Position',[1050 150 900 420]);

subplot(1,2,1);
  spectrogram(audio_original, win, nfft - hop_stft, nfft, sr, 'yaxis');
  title('Original — Spectrogram','Color','#E0E0E0','FontSize',10);
  set(gca,'Color','#0E1117','XColor','#C0C8D8','YColor','#C0C8D8');
  colormap(gca, 'parula');

subplot(1,2,2);
  spectrogram(audio_enh, win, nfft - hop_stft, nfft, sr, 'yaxis');
  title('Enhanced — Spectrogram','Color','#E0E0E0','FontSize',10);
  set(gca,'Color','#0E1117','XColor','#C0C8D8','YColor','#C0C8D8');
  colormap(gca, 'parula');

%% =========================================================================
%% STEP 10 — SNR EVALUATION & OUTPUT
%% =========================================================================
fprintf('[10/10] SNR evaluation & writing output...\n');

% Compute SNR improvement (approximation)
min_len   = min(length(audio_original), length(audio_enh));
sig_power = mean(audio_original(1:min_len).^2);
noise_pow = mean((audio_original(1:min_len) - audio_enh(1:min_len)).^2) + eps;
snr_db    = 10 * log10(sig_power / noise_pow);

fprintf('\n========================================\n');
fprintf('  RESULTS\n');
fprintf('========================================\n');
fprintf('  Speech frames : %.1f %%\n', speech_pct);
fprintf('  Noise  frames : %.1f %%\n', noise_pct);
fprintf('  Estimated SNR : %.2f dB\n', snr_db);
fprintf('  Output file   : %s\n', OUTPUT_FILE);
fprintf('========================================\n\n');

audiowrite(OUTPUT_FILE, audio_enh, sr);
fprintf('Done!  Enhanced audio saved to: %s\n', OUTPUT_FILE);


%% =========================================================================
%% LOCAL FUNCTION — Overlap-Add ISTFT
%% =========================================================================
function x = istft_ola(S, nfft, hop, orig_len)
%ISTFT_OLA  Overlap-add inverse STFT.
%   S       : complex spectrum matrix (nfft/2+1 × n_frames)
%   nfft    : FFT size
%   hop     : hop length
%   orig_len: expected output length (for trimming)

    n_frames = size(S, 2);
    win      = hann(nfft);

    % Mirror the one-sided spectrum back to two-sided
    S_full = [S; conj(S(end-1:-1:2, :))];

    out_len = (n_frames - 1) * hop + nfft;
    x       = zeros(out_len, 1);
    win_sum = zeros(out_len, 1);

    for k = 1:n_frames
        frame   = real(ifft(S_full(:, k)));
        idx     = (k-1)*hop + 1 : (k-1)*hop + nfft;
        x(idx)       = x(idx)       + win .* frame;
        win_sum(idx) = win_sum(idx) + win .^ 2;
    end

    % Normalize by window sum
    x = x ./ (win_sum + eps);

    % Trim to original length
    if orig_len > 0
        x = x(1:min(orig_len, length(x)));
    end
end
