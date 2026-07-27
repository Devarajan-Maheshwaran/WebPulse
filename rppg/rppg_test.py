"""
rppg/rppg_test.py — Standalone Verification Script for Classical rPPG + HR/HRV (Component 1).

RESEARCH CITATIONS & MOTIVATION:
- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). "Remote plethysmographic imaging using ambient light."
- Poh, M. Z., McDuff, D. J., & Picard, R. W. (2012). "Advancements in telecommunication and clinical monitoring."

Implementation Details:
- Captures real webcam video feed.
- Tracks face ROI via MediaPipe Face Mesh / Haar fallback.
- Extracts mean green-channel intensity per frame.
- Detrends raw signal & applies 3rd-order Butterworth bandpass filter (0.7–3.0 Hz, 42–180 BPM).
- Estimates Heart Rate (BPM) via FFT peak detection.
- Computes HRV (RMSSD in ms) from peak inter-beat intervals (IBI).
- NO synthetic/mock data used in live test mode.

Usage:
    python rppg/rppg_test.py
"""

import sys
import os
import time
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rppg.capture import FaceROICapturer
from rppg.signal import extract_roi_green_channel, detrend_signal, butterworth_bandpass_filter
from rppg.hrv import estimate_heart_rate_fft, find_pulse_peaks, compute_rmssd, map_hrv_to_arousal, classify_stress, HRSmoother


def run_rppg_verification_test():
    print("=" * 70)
    print("  WebPulse — Component 1: Classical rPPG & HR/HRV Verification Test")
    print("  Reference: Poh et al. (2012) & Verkruysse et al. (2008)")
    print("=" * 70)
    print("Press 'q' in video window to end test.\n")

    capturer = FaceROICapturer(camera_index=0)
    if not capturer.start():
        print("[UNTESTED ON HARDWARE] Webcam unavailable in this environment.")
        print("Module status: UNVERIFIED (Requires camera access to verify live signals).\n")
        return False

    hr_smoother = HRSmoother(history_size=5, alpha=0.3)
    raw_g_signal = []
    fps_estimate = 30.0
    start_time = time.time()
    frame_count = 0

    cv2.namedWindow("Component 1 — rPPG Verification Test", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = capturer.get_frame()
            if not ret or frame is None:
                print("[WARNING] Frame capture dropped.")
                break

            frame_count += 1
            roi_crop, full_face_box, roi_box, method = capturer.extract_roi(frame)
            display_frame = frame.copy()

            face_detected = full_face_box is not None
            if face_detected:
                fx, fy, fw, fh = full_face_box
                cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Face ROI ({method})", (fx, max(20, fy - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if roi_box is not None:
                    rx, ry, rw, rh = roi_box
                    cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)

                g_val = extract_roi_green_channel(roi_crop)
                if g_val is not None:
                    raw_g_signal.append(g_val)
            else:
                cv2.putText(display_frame, "[SEARCHING FOR FACE]", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_estimate = frame_count / elapsed

            window_size = int(fps_estimate * 15)
            if len(raw_g_signal) > window_size:
                raw_g_signal = raw_g_signal[-window_size:]

            # Process when buffer has >= 3 seconds of samples
            if len(raw_g_signal) >= int(fps_estimate * 3):
                detrended = detrend_signal(raw_g_signal)
                filtered = butterworth_bandpass_filter(detrended, fps=fps_estimate)
                
                raw_hr_bpm = estimate_heart_rate_fft(filtered, fps=fps_estimate)
                smoothed_hr = hr_smoother.update(raw_hr_bpm)
                
                peaks = find_pulse_peaks(filtered, fps=fps_estimate)
                rmssd = compute_rmssd(peaks, fps=fps_estimate)
                arousal = map_hrv_to_arousal(rmssd)
                stress_label, arousal_cat = classify_stress(rmssd)

                # Real-time console output
                if frame_count % int(fps_estimate) == 0:
                    hr_txt = f"{smoothed_hr:.1f} BPM" if smoothed_hr else "Calibrating..."
                    rmssd_txt = f"{rmssd:.1f} ms" if rmssd else "--"
                    print(f"[Real-Time rPPG] HR: {hr_txt:<14} | HRV(RMSSD): {rmssd_txt:<8} | Arousal: {arousal:.2f} ({stress_label})")

                # Display metrics overlay on frame
                hr_disp = f"Heart Rate: {smoothed_hr:.1f} BPM" if smoothed_hr else "Heart Rate: Calibrating..."
                rmssd_disp = f"HRV (RMSSD): {rmssd:.1f} ms" if rmssd else "HRV (RMSSD): --"
                arousal_disp = f"Arousal Score: {arousal:.2f} ({stress_label})"

                cv2.putText(display_frame, hr_disp, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, rmssd_disp, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.putText(display_frame, arousal_disp, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.imshow("Component 1 — rPPG Verification Test", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        capturer.stop()
        cv2.destroyAllWindows()
        print("\n[VERIFICATION COMPLETE] Component 1 rPPG live test finished.")
        return True


if __name__ == "__main__":
    run_rppg_verification_test()
