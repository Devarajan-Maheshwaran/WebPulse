"""
audio/valence.py — Pitch/Energy feature extraction and voice valence estimation.

Implements FR-4: Audio Capture and Valence Estimation.
Uses librosa to extract fundamental frequency (F0 pitch) and RMS energy over speech segments.
Maps pitch variability and energy patterns to a continuous valence score [-1.0, 1.0].
"""

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def extract_audio_features(audio_data, sr=22050):
    """
    Extract pitch (F0) and RMS energy features from speech audio array.
    
    Args:
        audio_data (ndarray 1D): Audio samples array.
        sr (int): Sampling rate.
        
    Returns:
        dict: Extracted acoustic features (f0_mean, f0_std, rms_mean, rms_std).
    """
    if not HAS_LIBROSA or len(audio_data) < int(sr * 0.1):
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "has_speech": False
        }

    y = np.array(audio_data, dtype=np.float32)

    # RMS Energy Calculation
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # Speech presence check (minimum threshold for RMS energy)
    if rms_mean < 0.005:
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "has_speech": False
        }

    # Pitch (F0) Extraction using pYIN or spectral centroid fallback
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=65.0,   # ~C2
            fmax=1046.0, # ~C6
            sr=sr
        )
        valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    except Exception:
        valid_f0 = np.array([])

    # Fallback to spectral centroid standard deviation if pYIN finds no voiced frame
    if len(valid_f0) > 0:
        f0_mean = float(np.mean(valid_f0))
        f0_std = float(np.std(valid_f0))
    else:
        # Spectral centroid as proxy for pitch variation in unvoiced/complex signals
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        f0_mean = float(np.mean(spec_cent))
        f0_std = float(np.std(spec_cent))

    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "has_speech": True
    }


def estimate_voice_valence(features, baseline_f0_std=15.0, baseline_rms=0.08):
    """
    Map pitch and energy acoustic features to a Valence Score in range [-1.0, 1.0].
    
    Heuristic Mapping (FR-4):
      - Higher pitch variability (f0_std) + moderate/high energy -> Positive valence (+0.2 to +1.0)
      - Flat pitch (low f0_std) + low energy -> Negative valence (-1.0 to -0.2)
      - Silence / no detected speech -> Neutral 0.0
      
    Args:
        features (dict): Feature dictionary from extract_audio_features.
        baseline_f0_std (float): Reference standard deviation of pitch dynamics (Hz).
        baseline_rms (float): Reference mean RMS energy.
        
    Returns:
        float: Valence score between -1.0 (Very Negative) and +1.0 (Very Positive).
    """
    if not features.get("has_speech", False):
        return 0.0  # Neutral when silent

    f0_std = features["f0_std"]
    rms_mean = features["rms_mean"]

    # Compute dynamic ratios relative to baselines
    pitch_dynamic_score = (f0_std - baseline_f0_std) / max(1.0, baseline_f0_std)
    energy_score = (rms_mean - baseline_rms) / max(0.01, baseline_rms)

    raw_valence = 0.5 * pitch_dynamic_score + 0.5 * energy_score

    # Clip score to [-1.0, 1.0]
    valence_score = float(np.clip(raw_valence, -1.0, 1.0))
    return valence_score
