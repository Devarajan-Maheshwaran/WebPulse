"""
rppg/signal.py — Classical rPPG Signal Extraction, Detrending, and Butterworth Filtering.

RESEARCH REFERENCES & THEORETICAL MOTIVATION:
1. Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008).
   "Remote plethysmographic imaging using ambient light." Optics Express, 16(26), 21434-21445.
   - Demonstrated that hemoglobin peak absorption occurs in the green optical channel (500-600 nm),
     making mean green-channel intensity from facial ROI the primary classical rPPG signal.

2. Poh, M. Z., McDuff, D. J., & Picard, R. W. (2012).
   "Advancements in telecommunication and clinical monitoring through webcam-based heart rate measurement."
   IEEE Transactions on Biomedical Engineering, 59(4), 1049-1056.
   - Established detrending & 0.7-3.0 Hz (42-180 BPM) bandpass filtering for classical webcam HR/HRV extraction.
"""

import numpy as np
from scipy import signal as sp_signal


def extract_roi_green_channel(roi_crop):
    """
    Extract average spatial green-channel intensity from ROI crop.
    Reference: Verkruysse et al. (2008).
    
    Args:
        roi_crop (ndarray): BGR image patch of the ROI.
        
    Returns:
        float: Mean intensity of the green channel (Index 1 in BGR).
    """
    if roi_crop is None or roi_crop.size == 0:
        return None
    return float(np.mean(roi_crop[:, :, 1]))


def extract_roi_pos_signal(roi_crop):
    """
    Extract Plane-Orthogonal-to-Skin (POS) color channel combination from ROI.
    Reference: Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 2017.
    """
    if roi_crop is None or roi_crop.size == 0:
        return None
    b_mean = np.mean(roi_crop[:, :, 0])
    g_mean = np.mean(roi_crop[:, :, 1])
    r_mean = np.mean(roi_crop[:, :, 2])
    return r_mean, g_mean, b_mean


def detrend_signal(raw_signal):
    """
    Detrend raw time-series signal to eliminate low-frequency movement & ambient light drift.
    Reference: Poh et al. (2012).
    """
    sig = np.array(raw_signal, dtype=np.float64)
    if len(sig) < 3:
        return sig - np.mean(sig) if len(sig) > 0 else sig
    return sp_signal.detrend(sig)


def butterworth_bandpass_filter(data, fps, lowcut=0.7, highcut=3.0, order=3):
    """
    3rd-order Butterworth bandpass filter in 0.7–3.0 Hz (~42–180 BPM) range.
    Reference: Poh et al. (2012).
    
    Args:
        data (array-like): Detrended signal.
        fps (float): Frame rate.
        lowcut (float): Lower frequency cut-off (0.7 Hz = 42 BPM).
        highcut (float): Upper frequency cut-off (3.0 Hz = 180 BPM).
        order (int): Filter order (default 3).
        
    Returns:
        ndarray: Zero-phase bandpass-filtered signal.
    """
    sig = np.array(data, dtype=np.float64)
    n_samples = len(sig)
    
    if n_samples < 15 or fps <= 0:
        return sig - np.mean(sig)

    nyquist = 0.5 * fps
    low = lowcut / nyquist
    high = highcut / nyquist
    
    low = max(0.01, min(low, 0.95))
    high = max(low + 0.05, min(high, 0.99))

    try:
        b, a = sp_signal.butter(order, [low, high], btype='bandpass')
        padlen = min(3 * max(len(a), len(b)), n_samples - 1)
        if padlen > 0:
            filtered = sp_signal.filtfilt(b, a, sig, padlen=padlen)
        else:
            filtered = sp_signal.lfilter(b, a, sig)
        return filtered
    except Exception:
        return sig - np.mean(sig)
