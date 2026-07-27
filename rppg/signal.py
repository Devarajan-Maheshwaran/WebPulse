"""
rppg/signal.py — rPPG signal extraction, detrending, and Butterworth filtering.

Implements FR-2: Pulse Signal Extraction (rPPG).
- Green-channel mean intensity extraction from ROI (or POS method).
- Signal detrending to eliminate slow movement / illumination drift.
- Butterworth bandpass filter (0.7–3.0 Hz, ~42–180 BPM).
"""

import numpy as np
from scipy import signal as sp_signal


def extract_roi_green_channel(roi_crop):
    """
    Extract the average spatial intensity of the green channel from an ROI crop.
    
    Args:
        roi_crop (ndarray): BGR image patch of the ROI.
        
    Returns:
        float: Mean intensity of the green channel (Channel index 1 in BGR).
    """
    if roi_crop is None or roi_crop.size == 0:
        return None
    # OpenCV BGR -> index 1 is Green
    green_mean = np.mean(roi_crop[:, :, 1])
    return float(green_mean)


def extract_roi_pos_signal(roi_crop):
    """
    Extract Plane-Orthogonal-to-Skin (POS) color channel combination from ROI.
    Reference: Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 2017.
    
    Args:
        roi_crop (ndarray): BGR image patch of the ROI.
        
    Returns:
        tuple (float, float, float): Means of (R, G, B) channels.
    """
    if roi_crop is None or roi_crop.size == 0:
        return None
    b_mean = np.mean(roi_crop[:, :, 0])
    g_mean = np.mean(roi_crop[:, :, 1])
    r_mean = np.mean(roi_crop[:, :, 2])
    return r_mean, g_mean, b_mean


def detrend_signal(raw_signal):
    """
    Detrend time-series raw signal to remove low-frequency baseline drifts.
    
    Args:
        raw_signal (array-like): Raw time-series values.
        
    Returns:
        ndarray: Linear detrended signal.
    """
    sig = np.array(raw_signal, dtype=np.float64)
    if len(sig) < 3:
        return sig - np.mean(sig) if len(sig) > 0 else sig
    return sp_signal.detrend(sig)


def butterworth_bandpass_filter(data, fps, lowcut=0.7, highcut=3.0, order=3):
    """
    Apply a 3rd-order Butterworth bandpass filter (0.7–3.0 Hz) to isolate pulse.
    0.7 Hz = ~42 BPM, 3.0 Hz = ~180 BPM.
    
    Args:
        data (array-like): Detrended temporal signal window.
        fps (float): Sampling rate (frames per second).
        lowcut (float): Lower frequency cutoff in Hz (default 0.7 Hz).
        highcut (float): Upper frequency cutoff in Hz (default 3.0 Hz).
        order (int): Butterworth filter order (default 3).
        
    Returns:
        ndarray: Zero-phase bandpass-filtered signal.
    """
    sig = np.array(data, dtype=np.float64)
    n_samples = len(sig)
    
    # Needs sufficient points to filter
    if n_samples < 15 or fps <= 0:
        return sig - np.mean(sig)

    nyquist = 0.5 * fps
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Clamp bounds for filter stability
    low = max(0.01, min(low, 0.95))
    high = max(low + 0.05, min(high, 0.99))

    try:
        b, a = sp_signal.butter(order, [low, high], btype='bandpass')
        # Use filtfilt for zero phase distortion if enough samples exist
        padlen = min(3 * max(len(a), len(b)), n_samples - 1)
        if padlen > 0:
            filtered = sp_signal.filtfilt(b, a, sig, padlen=padlen)
        else:
            filtered = sp_signal.lfilter(b, a, sig)
        return filtered
    except Exception:
        # Fallback if filter calculation fails
        return sig - np.mean(sig)
