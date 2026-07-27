"""
rppg/hrv.py — Heart rate and HRV/arousal estimation module.

Implements FR-3: Heart Rate and HRV/Arousal Estimation.
- Rolling window FFT peak detection for Heart Rate (BPM).
- Peak-to-peak inter-beat interval (IBI) analysis & RMSSD-style HRV metric.
- HRV -> Arousal/Stress score mapping with configurable thresholds.

NOTE: Thresholds used in map_hrv_to_arousal are initial guesses and MUST be calibrated
from real pilot test session data.
"""

import numpy as np
from scipy import signal as sp_signal


def estimate_heart_rate_fft(filtered_signal, fps, min_bpm=42.0, max_bpm=180.0):
    """
    Estimate heart rate in BPM using FFT spectral power peak detection.
    
    Args:
        filtered_signal (array-like): Bandpass-filtered signal window.
        fps (float): Sampling frame rate in FPS.
        min_bpm (float): Minimum valid heart rate (default 42 BPM = 0.7 Hz).
        max_bpm (float): Maximum valid heart rate (default 180 BPM = 3.0 Hz).
        
    Returns:
        float: Estimated heart rate in BPM, or None if signal is insufficient.
    """
    sig = np.array(filtered_signal, dtype=np.float64)
    n = len(sig)
    if n < 30 or fps <= 0:
        return None

    # Compute FFT
    fft_vals = np.fft.rfft(sig * np.hanning(n))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_power = np.abs(fft_vals) ** 2

    # Filter frequencies to [min_bpm/60, max_bpm/60]
    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0

    valid_idx = np.where((fft_freqs >= min_hz) & (fft_freqs <= max_hz))[0]
    if len(valid_idx) == 0:
        return None

    peak_sub_idx = np.argmax(fft_power[valid_idx])
    dominant_freq = fft_freqs[valid_idx[peak_sub_idx]]
    bpm = dominant_freq * 60.0
    return float(bpm)


def find_pulse_peaks(filtered_signal, fps, min_bpm=42.0, max_bpm=180.0):
    """
    Locate peak indices in filtered signal corresponding to individual heartbeats.
    
    Args:
        filtered_signal (array-like): Bandpass-filtered signal window.
        fps (float): Sampling frame rate in FPS.
        
    Returns:
        ndarray: Indices of detected systolic peaks.
    """
    sig = np.array(filtered_signal, dtype=np.float64)
    if len(sig) < 15 or fps <= 0:
        return np.array([], dtype=int)

    # Minimum distance between peaks in samples (based on max_bpm)
    min_dist_samples = int((60.0 / max_bpm) * fps)
    min_dist_samples = max(1, min_dist_samples)

    peaks, _ = sp_signal.find_peaks(
        sig,
        distance=min_dist_samples,
        prominence=np.std(sig) * 0.3 if np.std(sig) > 0 else None
    )
    return peaks


def compute_rmssd(peak_indices, fps):
    """
    Compute Root Mean Square of Successive Differences (RMSSD) of inter-beat intervals.
    
    Args:
        peak_indices (array-like): Indices of detected heartbeat peaks.
        fps (float): Sampling rate in FPS.
        
    Returns:
        float: RMSSD in milliseconds (ms), or None if insufficient peaks.
    """
    if len(peak_indices) < 3 or fps <= 0:
        return None

    # Convert peak frame differences to milliseconds
    rrs_ms = np.diff(peak_indices) / fps * 1000.0
    
    # Filter physiologically unrealistic RR intervals (e.g. outside 300ms–1400ms)
    valid_rrs = rrs_ms[(rrs_ms >= 300.0) & (rrs_ms <= 1400.0)]
    if len(valid_rrs) < 2:
        return None

    successive_diffs = np.diff(valid_rrs)
    rmssd = np.sqrt(np.mean(np.square(successive_diffs)))
    return float(rmssd)


def map_hrv_to_arousal(rmssd, baseline_min_rmssd=15.0, baseline_max_rmssd=80.0):
    """
    Map RMSSD HRV metric to a continuous Arousal / Physiological Stress Score [0.0, 1.0].
    
    Physiological Framing (Sugaya's lab alignment):
      - Lower HRV (RMSSD) -> Higher arousal / physiological stress.
      - Higher HRV (RMSSD) -> Lower arousal / high vagal tone.

    NOTE: The default baseline thresholds (15ms - 80ms) are an INITIAL GUESS
    and NEED CALIBRATION from real pilot data.

    Args:
        rmssd (float): RMSSD in milliseconds.
        baseline_min_rmssd (float): RMSSD threshold corresponding to max arousal (default: 15.0 ms).
        baseline_max_rmssd (float): RMSSD threshold corresponding to min arousal (default: 80.0 ms).
        
    Returns:
        float: Arousal score in range [0.0, 1.0], where 1.0 = High Stress/Arousal.
    """
    if rmssd is None:
        return 0.5  # Neutral default when uncalibrated/unavailable

    # Clamp RMSSD to expected baseline range
    clamped_rmssd = max(baseline_min_rmssd, min(baseline_max_rmssd, rmssd))

    # Invert scale: lower RMSSD -> higher arousal
    normalized_calm = (clamped_rmssd - baseline_min_rmssd) / (baseline_max_rmssd - baseline_min_rmssd)
    arousal_score = 1.0 - normalized_calm

    return float(np.clip(arousal_score, 0.0, 1.0))
