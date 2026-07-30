"""
rppg/hrv.py — Heart Rate, HRV (RMSSD), and Sugaya-Style Arousal/Stress Classification.

RESEARCH REFERENCES & THEORETICAL MOTIVATION:
1. Sugaya et al.
   "Emotion estimation from EEG and HRV indices with Machine Learning."
   - Established that Heart Rate Variability (HRV) metrics (specifically RMSSD) strongly reflect
     autonomic nervous system (ANS) activation and physiological stress/arousal.
   - Lower HRV (RMSSD) corresponds to higher sympathetic arousal / physiological stress.
   - Higher HRV (RMSSD) corresponds to parasympathetic vagal dominance / calm state.

2. Camera-Based PRV Emotion Recognition (2023).
   "Dimensional emotion recognition from camera-based pulse rate variability (PRV) features."
   - Validates that remote camera-extracted pulse rate variability (PRV) serves as a valid 1D proxy
     for the physiological arousal dimension in dimensional emotion models (Russell Circumplex).
"""

import numpy as np
from scipy import signal as sp_signal


class HRSmoother:
    """
    Smoothing and outlier rejection filter for rPPG Heart Rate estimates.
    """
    def __init__(self, history_size=5, alpha=0.3):
        self.history_size = history_size
        self.alpha = alpha
        self.history = []
        self.last_smoothed = None

    def update(self, raw_bpm):
        if raw_bpm is None or not (50.0 <= raw_bpm <= 140.0):
            return self.last_smoothed

        self.history.append(raw_bpm)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        median_bpm = float(np.median(self.history))

        if self.last_smoothed is None:
            self.last_smoothed = median_bpm
        else:
            self.last_smoothed = self.alpha * median_bpm + (1 - self.alpha) * self.last_smoothed

        return float(self.last_smoothed)


class ArousalSmoother:
    """
    Exponential moving average (EMA) filter for physiological arousal score.
    Prevents abrupt drop-offs to 0.0 or freeze.
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.last_arousal = 0.50

    def update(self, raw_arousal):
        if raw_arousal is None:
            return self.last_arousal
        self.last_arousal = self.alpha * raw_arousal + (1 - self.alpha) * self.last_arousal
        return float(np.clip(self.last_arousal, 0.05, 0.95))


def estimate_heart_rate_fft(filtered_signal, fps, min_bpm=50.0, max_bpm=120.0):
    """
    Estimate heart rate in BPM using FFT spectral power peak detection
    with harmonic suppression to prevent 2x second-harmonic doubling artifacts.
    Reference: Poh et al. (2012), pyVHR.
    """
    sig = np.array(filtered_signal, dtype=np.float64)
    n = len(sig)
    if n < 45 or fps <= 0:
        return None

    fft_vals = np.fft.rfft(sig * np.hanning(n))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_power = np.abs(fft_vals) ** 2

    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0

    valid_idx = np.where((fft_freqs >= min_hz) & (fft_freqs <= max_hz))[0]
    if len(valid_idx) == 0:
        return None

    valid_freqs = fft_freqs[valid_idx]
    valid_power = fft_power[valid_idx]

    peak_sub_idx = np.argmax(valid_power)
    dominant_freq = valid_freqs[peak_sub_idx]
    peak_power = valid_power[peak_sub_idx]

    # Harmonic suppression: check if dominant peak is a 2nd harmonic of a true fundamental HR
    half_freq = dominant_freq / 2.0
    if half_freq >= (48.0 / 60.0):
        # Look for sub-harmonic peak in range [half_freq - 0.15Hz, half_freq + 0.15Hz]
        sub_mask = np.abs(valid_freqs - half_freq) <= 0.15
        if np.any(sub_mask):
            sub_max_power = np.max(valid_power[sub_mask])
            # If fundamental has at least 20% power of harmonic peak, choose fundamental
            if sub_max_power >= 0.20 * peak_power:
                sub_peak_idx = np.argmax(valid_power[sub_mask])
                dominant_freq = valid_freqs[sub_mask][sub_peak_idx]

    bpm = dominant_freq * 60.0
    return float(bpm)



def find_pulse_peaks(filtered_signal, fps, min_bpm=55.0, max_bpm=130.0):
    """
    Locate peak indices in filtered signal corresponding to individual heartbeats.
    """
    sig = np.array(filtered_signal, dtype=np.float64)
    if len(sig) < 15 or fps <= 0:
        return np.array([], dtype=int)

    min_dist_samples = int((60.0 / max_bpm) * fps)
    min_dist_samples = max(1, min_dist_samples)

    peaks, _ = sp_signal.find_peaks(
        sig,
        distance=min_dist_samples,
        prominence=np.std(sig) * 0.15 if np.std(sig) > 0 else None
    )
    return peaks


def compute_rmssd(peak_indices, fps):
    """
    Compute Root Mean Square of Successive Differences (RMSSD) of inter-beat intervals (IBI).
    Formula: RMSSD = sqrt( mean( (RR_i+1 - RR_i)^2 ) )
    """
    if len(peak_indices) < 3 or fps <= 0:
        return None

    rrs_ms = np.diff(peak_indices) / fps * 1000.0
    valid_rrs = rrs_ms[(rrs_ms >= 350.0) & (rrs_ms <= 1500.0)]
    if len(valid_rrs) < 2:
        return None

    successive_diffs = np.diff(valid_rrs)
    rmssd = np.sqrt(np.mean(np.square(successive_diffs)))
    return float(rmssd)


def map_hrv_to_arousal(rmssd, baseline_min_rmssd=10.0, baseline_max_rmssd=100.0):
    """
    Map RMSSD HRV metric to a continuous Arousal / Physiological Stress Score in [0.05, 0.95].
    
    Theoretical Framing (Sugaya Lab & PRV Emotion Papers):
      - Lower RMSSD (high stress / sympathetic tone) -> Arousal Score closer to 1.0
      - Higher RMSSD (calm / parasympathetic vagal tone) -> Arousal Score closer to 0.0
    """
    if rmssd is None or rmssd <= 0:
        return 0.50

    clamped_rmssd = max(baseline_min_rmssd, min(baseline_max_rmssd, rmssd))
    normalized_calm = (clamped_rmssd - baseline_min_rmssd) / (baseline_max_rmssd - baseline_min_rmssd)
    arousal_score = 1.0 - normalized_calm

    return float(np.clip(arousal_score, 0.05, 0.95))


def classify_stress(rmssd, calm_thresh=50.0, stress_thresh=25.0):
    """
    Classify physiological stress state into 3 discrete states based on HRV RMSSD.
    Inspired by Sugaya Lab HRV feature classification.
    
    Args:
        rmssd (float): Computed RMSSD in ms.
        calm_thresh (float): RMSSD threshold above which state is CALM (default: 50.0 ms).
        stress_thresh (float): RMSSD threshold below which state is STRESSED (default: 25.0 ms).
        
    Returns:
        tuple (str, str): (Stress Label, Arousal Category) e.g. ("CALM", "LOW AROUSAL")
    """
    if rmssd is None:
        return "NORMAL", "MODERATE AROUSAL"

    if rmssd >= calm_thresh:
        return "CALM", "LOW AROUSAL"
    elif rmssd < stress_thresh:
        return "STRESSED", "HIGH AROUSAL"
    else:
        return "NORMAL", "MODERATE AROUSAL"
