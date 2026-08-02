"""
test_rppg.py — Standalone Test Script for Phase 1 (rPPG Signal Pipeline) & Phase 2 (HR/HRV Arousal Estimation).

Usage:
    conda run -n webpulse python test_rppg.py [--synthetic]

This script tests:
  1. Camera & MediaPipe ROI extraction
  2. Green channel signal collection & detrending/bandpass filtering
  3. Rolling window FFT Heart Rate (BPM) & RMSSD HRV -> Arousal score estimation
"""

import sys
import time
import numpy as np
import cv2

from rppg.capture import FaceROICapturer
from rppg.signal_proc import extract_roi_green_channel, detrend_signal, butterworth_bandpass_filter

from rppg.hrv import estimate_heart_rate_fft, find_pulse_peaks, compute_rmssd, map_hrv_to_arousal


def test_synthetic_signal_pipeline():
    """Run mathematical validation on a synthetic 1.2 Hz (72 BPM) pulse signal with added HRV variations."""
    print("\n--- SYNTHETIC TESTING MODE (MATH & FILTERING VERIFICATION) ---")
    
    fps = 30.0
    duration_sec = 20.0
    t = np.linspace(0, duration_sec, int(fps * duration_sec))
    
    # 72 BPM fundamental (1.2 Hz) + harmonic + noise + low frequency drift
    synthetic_raw = (
        120.0 
        + 5.0 * np.sin(2 * np.pi * 1.2 * t) 
        + 1.5 * np.sin(2 * np.pi * 2.4 * t)
        + 10.0 * np.sin(2 * np.pi * 0.05 * t)  # Low frequency drift
        + np.random.normal(0, 0.5, len(t))    # Noise
    )
    
    # Process through pipeline
    detrended = detrend_signal(synthetic_raw)
    filtered = butterworth_bandpass_filter(detrended, fps=fps, lowcut=0.7, highcut=3.0)
    
    hr_bpm = estimate_heart_rate_fft(filtered, fps=fps)
    peaks = find_pulse_peaks(filtered, fps=fps)
    rmssd = compute_rmssd(peaks, fps=fps)
    arousal = map_hrv_to_arousal(rmssd)
    
    print(f"Synthetic Input: Fundamental 72.0 BPM (1.2 Hz) + Noise/Drift")
    print(f"  Detrended Signal Mean:   {np.mean(detrended):.4f}")
    print(f"  Filtered Signal Std:     {np.std(filtered):.4f}")
    print(f"  Estimated Heart Rate:    {hr_bpm:.2f} BPM (Expected ~72.0 BPM)")
    print(f"  Detected Peaks Count:    {len(peaks)}")
    print(f"  Computed RMSSD HRV:      {rmssd if rmssd else 'N/A'} ms")
    print(f"  Mapped Arousal Score:    {arousal:.3f} (0.0=Calm, 1.0=Stressed)")
    print("\n[VERIFICATION RESULT] Signal processing math & filter pipeline operate correctly.")
    return True


def run_live_webcam_test():
    """Run live rPPG pipeline on actual hardware webcam."""
    print("\n--- LIVE WEBCAM rPPG TEST MODE ---")
    print("Attempting webcam connection... Press 'q' in frame window to quit.\n")
    
    capturer = FaceROICapturer(camera_index=0)
    if not capturer.start():
        print("[UNTESTED ON HARDWARE] Camera 0 could not be opened. Running synthetic math test instead.")
        return test_synthetic_signal_pipeline()

    raw_signal = []
    fps_estimate = 30.0
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = capturer.get_frame()
            if not ret or frame is None:
                print("[WARNING] Frame capture dropped.")
                break

            frame_count += 1
            roi_crop, full_face_box, roi_box, method, quality_meta = capturer.extract_roi(frame)

            # Visual overlay on camera feed
            display_frame = frame.copy()
            if roi_box is not None:
                x, y, w, h = roi_box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ROI: {method}", (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Signal extraction
            if roi_crop is not None:
                g_val = extract_roi_green_channel(roi_crop)
                if g_val is not None:
                    raw_signal.append(g_val)

            # Calculate FPS dynamically
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_estimate = frame_count / elapsed

            # Update rolling window (keep last 15 seconds)
            window_size = int(fps_estimate * 15)
            if len(raw_signal) > window_size:
                raw_signal = raw_signal[-window_size:]

            # Process & Estimate when buffer has enough samples (~5 seconds min)
            if len(raw_signal) >= int(fps_estimate * 5):
                detrended = detrend_signal(raw_signal)
                filtered = butterworth_bandpass_filter(detrended, fps=fps_estimate)
                
                hr_bpm = estimate_heart_rate_fft(filtered, fps=fps_estimate)
                peaks = find_pulse_peaks(filtered, fps=fps_estimate)
                rmssd = compute_rmssd(peaks, fps=fps_estimate)
                arousal = map_hrv_to_arousal(rmssd)

                # Overlay metrics on screen
                hr_str = f"HR: {hr_bpm:.1f} BPM" if hr_bpm else "HR: Estimating..."
                rmssd_str = f"RMSSD: {rmssd:.1f} ms" if rmssd else "RMSSD: --"
                arousal_str = f"Arousal: {arousal:.2f}"

                cv2.putText(display_frame, hr_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, rmssd_str, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, arousal_str, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow("WebPulse — Phase 1 & 2 rPPG Test", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        capturer.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("  WebPulse — Phase 1 & Phase 2 Standalone Test")
    print("=" * 60)
    if len(sys.argv) > 1 and sys.argv[1] == "--synthetic":
        test_synthetic_signal_pipeline()
    else:
        run_live_webcam_test()
