"""
rppg/enhancement.py — Low-Light Preprocessing & Signal Quality Gating (Phase D).

Implements:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for low-light ROI enhancement.
  2. Mean illuminance & brightness checks for low-light detection.
  3. Spectral Signal-to-Noise Ratio (SNR) & Signal Quality Index (SQI) gating.
"""

import cv2
import numpy as np


class ROIEnhancer:
    """
    Lightweight CPU contrast enhancement & lighting quality inspector.
    """

    def __init__(self, clip_limit=2.2, tile_grid_size=(8, 8), min_brightness=32.0,
                 gamma_threshold=75.0):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.min_brightness = min_brightness
        self.gamma_threshold = gamma_threshold
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance(self, roi_bgr):
        """
        Applies CLAHE to the Luminance (L) channel in LAB color space.

        Args:
            roi_bgr (ndarray): BGR image patch.

        Returns:
            ndarray: Contrast-enhanced BGR image patch.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return roi_bgr

        gray_mean = float(np.mean(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)))
        if gray_mean < self.gamma_threshold:
            gamma = 1.0 + min(0.55, (self.gamma_threshold - gray_mean) / 140.0)
            lookup = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
            roi_bgr = cv2.LUT(roi_bgr, lookup)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self.clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def check_brightness(self, roi_bgr):
        """
        Checks mean luminance of cropped ROI.

        Returns:
            tuple: (is_low_light: bool, mean_brightness: float)
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return True, 0.0

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        is_low_light = mean_b < self.min_brightness
        return is_low_light, mean_b


def compute_bvp_snr(bvp_signal, fps=30, min_hr_hz=0.8, max_hr_hz=2.2):
    """
    Computes Signal-to-Noise Ratio (SNR) of predicted BVP signal in heart rate band.
    Reference: De Haan & Jeanne (2013), pyVHR.

    Returns:
        tuple: (snr_db: float, is_quality_sufficient: bool)
    """
    sig = np.array(bvp_signal, dtype=np.float64)
    n = len(sig)
    if n < int(fps * 3):
        return 0.0, True  # Calibrating

    # Detrend & Hanning window
    sig_detrended = sig - np.mean(sig)
    fft_vals = np.fft.rfft(sig_detrended * np.hanning(n))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_power = np.abs(fft_vals) ** 2

    # Restrict analysis to physiological pulse band [0.8 Hz, 2.2 Hz] (~48-132 BPM)
    cardiac_mask = (fft_freqs >= min_hr_hz) & (fft_freqs <= max_hr_hz)
    if not np.any(cardiac_mask) or np.std(sig) < 1e-6:
        return -10.0, False

    cardiac_power = fft_power[cardiac_mask]
    cardiac_freqs = fft_freqs[cardiac_mask]

    peak_idx = np.argmax(cardiac_power)
    peak_freq = cardiac_freqs[peak_idx]

    # Peak band (+/- 0.15 Hz around peak HR frequency)
    peak_mask = np.abs(cardiac_freqs - peak_freq) <= 0.15
    signal_power = np.sum(cardiac_power[peak_mask])
    noise_power = np.sum(cardiac_power[~peak_mask]) + 1e-6

    snr_ratio = signal_power / noise_power
    snr_db = float(10.0 * np.log10(snr_ratio))

    is_sufficient = (snr_db >= -5.0) and (np.std(sig) > 1e-5)
    return snr_db, is_sufficient
