"""
rppg/rppg_server.py — Process 1: Video Capture & rPPG Producer Server.

Runs as Process 1:
  - Captures webcam frames & detects face ROI (MediaPipe / Haar Cascade).
  - Extracts green-channel signal, detrends, bandpass filters, and estimates:
      - heart_rate (BPM)
      - hrv (RMSSD in ms)
      - arousal_score [0.0, 1.0]
  - Hosts a TCP Socket Server on localhost:5001 broadcasting rPPG metrics as JSON lines.
"""

import sys
import os
import time
import json
import socket
import threading
import numpy as np
import cv2

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rppg.capture import FaceROICapturer
from rppg.signal import extract_roi_green_channel, detrend_signal, butterworth_bandpass_filter
from rppg.hrv import estimate_heart_rate_fft, find_pulse_peaks, compute_rmssd, map_hrv_to_arousal, HRSmoother


class RPPGTCPServer:
    """Simple multi-client TCP Socket Server for broadcasting rPPG data."""

    def __init__(self, host="127.0.0.1", port=5001):
        self.host = host
        self.port = port
        self.clients = []
        self.lock = threading.Lock()
        self.running = False
        self.server_socket = None

    def start(self):
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"[rPPG Server] Listening for TCP connections on {self.host}:{self.port}...")

        accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        accept_thread.start()

    def _accept_loop(self):
        while self.running:
            try:
                client_sock, addr = self.server_socket.accept()
                with self.lock:
                    self.clients.append(client_sock)
                print(f"[rPPG Server] Client connected from {addr}")
            except Exception:
                break

    def broadcast(self, payload_dict):
        message = (json.dumps(payload_dict) + "\n").encode("utf-8")
        with self.lock:
            disconnected = []
            for client in self.clients:
                try:
                    client.sendall(message)
                except Exception:
                    disconnected.append(client)
            for client in disconnected:
                self.clients.remove(client)

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        with self.lock:
            for client in self.clients:
                try:
                    client.close()
                except Exception:
                    pass
            self.clients.clear()


def run_rppg_server():
    print("=" * 70)
    print("  Process 1: WebPulse rPPG Producer Server (localhost:5001)")
    print("=" * 70)
    print("Press 'q' in video window to stop server.\n")

    server = RPPGTCPServer(host="127.0.0.1", port=5001)
    server.start()

    video_capturer = FaceROICapturer(camera_index=0)
    if not video_capturer.start():
        print("[ERROR] Webcam unavailable. Exiting rPPG server.")
        server.stop()
        return

    hr_smoother = HRSmoother(history_size=5, alpha=0.3)
    raw_g_signal = []
    fps_estimate = 30.0
    start_time = time.time()
    last_broadcast_time = 0
    frame_count = 0

    cv2.namedWindow("webpulse - rPPG Video Producer", cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty("webpulse - rPPG Video Producer", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except Exception:
        pass

    try:
        while True:
            ret, frame = video_capturer.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            roi_crop, full_face_box, roi_box, method = video_capturer.extract_roi(frame)
            display_frame = frame.copy()

            face_detected = full_face_box is not None
            if face_detected:
                fx, fy, fw, fh = full_face_box
                cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Face ({method})", (fx, max(20, fy - 10)),
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

            current_hr = None
            current_rmssd = None
            current_arousal = 0.5

            if len(raw_g_signal) >= int(fps_estimate * 3):
                detrended = detrend_signal(raw_g_signal)
                filtered = butterworth_bandpass_filter(detrended, fps=fps_estimate)
                
                raw_hr_bpm = estimate_heart_rate_fft(filtered, fps=fps_estimate)
                current_hr = hr_smoother.update(raw_hr_bpm)
                
                peaks = find_pulse_peaks(filtered, fps=fps_estimate)
                current_rmssd = compute_rmssd(peaks, fps=fps_estimate)
                current_arousal = map_hrv_to_arousal(current_rmssd)

            # Broadcast metrics over TCP every 1 second
            now = time.time()
            if now - last_broadcast_time >= 1.0:
                last_broadcast_time = now
                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    "heart_rate": current_hr,
                    "hrv": current_rmssd,
                    "arousal": current_arousal,
                    "face_detected": face_detected
                }
                server.broadcast(payload)
                
                hr_str = f"{current_hr:.1f} BPM" if current_hr else "Calibrating..."
                print(f"[rPPG Server] Sent -> HR: {hr_str} | Arousal: {current_arousal:.2f} | Face: {face_detected}")

            # Draw Overlay Info
            hr_disp = f"HR: {current_hr:.1f} BPM" if current_hr else "HR: Calibrating..."
            cv2.putText(display_frame, hr_disp, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Arousal: {current_arousal:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.imshow("webpulse - rPPG Video Producer", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        video_capturer.stop()
        cv2.destroyAllWindows()
        server.stop()
        print("\n[rPPG Server] Stopped.")


if __name__ == "__main__":
    run_rppg_server()
