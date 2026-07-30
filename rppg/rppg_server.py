"""
rppg/rppg_server.py — Process 1: Video Capture, rPPG Producer Server & Live HUD Interface.

Runs as Process 1:
  - Captures webcam frames & detects face ROI (MediaPipe / Haar Cascade).
  - Extracts green-channel signal, detrends, bandpass filters, and estimates:
      - heart_rate (BPM)
      - hrv (RMSSD in ms)
      - arousal_score [0.05, 0.95]
  - Hosts a TCP Socket Server on localhost:5001 broadcasting rPPG metrics as JSON lines.
  - Receives live HUD update payloads from Process 3 (app.py) over TCP socket.
  - Renders complete WebPulse visual overlay interface using rppg.hud.
"""

import sys
import os
import time
import json
import queue
import socket
import threading
import numpy as np
import cv2

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rppg.capture import FaceROICapturer
from rppg.signal_proc import extract_roi_green_channel, detrend_signal, butterworth_bandpass_filter
from rppg.hrv import estimate_heart_rate_fft, find_pulse_peaks, compute_rmssd, map_hrv_to_arousal, HRSmoother
from rppg.hud import draw_overlay_hud
from rppg.deep_engine import DeepRPPGEngine


class StateBrokerPublisher:
    """Non-blocking publisher so IPC can never stall camera inference."""

    def __init__(self):
        self.queue = queue.Queue(maxsize=2)
        self.running = False
        self.worker = None

    def start(self):
        self.running = True
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def publish(self, state):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put_nowait(dict(state))

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            try:
                state = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                with socket.create_connection(("127.0.0.1", 5003), timeout=0.25) as client:
                    client.sendall((json.dumps({"type": "publish", "state": state}) + "\n").encode("utf-8"))
            except OSError:
                # Broker may start after the vision process; keep camera running.
                pass


def draw_capture_guides(frame, face_box, rois):
    """Render only the camera guidance; it never alters the ROI crops or model input."""
    if face_box is None:
        return frame

    overlay = frame.copy()
    fx, fy, fw, fh = face_box
    guide_color = (185, 185, 185)
    corner = max(12, min(fw, fh) // 7)

    # A face guide uses four quiet corner marks instead of a distracting full box.
    for x, y, dx, dy in (
        (fx, fy, 1, 1), (fx + fw, fy, -1, 1),
        (fx, fy + fh, 1, -1), (fx + fw, fy + fh, -1, -1),
    ):
        cv2.line(overlay, (x, y), (x + dx * corner, y), guide_color, 1, cv2.LINE_AA)
        cv2.line(overlay, (x, y), (x, y + dy * corner), guide_color, 1, cv2.LINE_AA)

    for roi_data in (rois or {}).values():
        if roi_data.get("box") is None:
            continue
        rx, ry, rw, rh = roi_data["box"]
        usable = roi_data.get("status") == "USABLE"
        color = (205, 185, 65) if usable else (90, 95, 185)
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), color, -1)
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), color, 1, cv2.LINE_AA)

    # Keep the fill barely visible so it communicates model regions without obscuring the face.
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.putText(frame, "rPPG regions", (fx, max(18, fy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, guide_color, 1, cv2.LINE_AA)
    return frame


class RPPGTCPServer:
    """Bi-directional TCP Socket Server for broadcasting rPPG metrics & receiving HUD updates."""

    def __init__(self, host="127.0.0.1", port=5001, on_hud_update=None):
        self.host = host
        self.port = port
        self.on_hud_update = on_hud_update
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

                recv_thread = threading.Thread(target=self._client_recv_loop, args=(client_sock,), daemon=True)
                recv_thread.start()
            except Exception:
                break

    def _client_recv_loop(self, client_sock):
        buffer = ""
        while self.running:
            try:
                data = client_sock.recv(1024).decode("utf-8")
                if not data:
                    break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        try:
                            payload = json.loads(line.strip())
                            if payload.get("type") == "hud_update" and self.on_hud_update:
                                self.on_hud_update(payload)
                        except Exception:
                            pass
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
    print("  Process 1: WebPulse rPPG Producer & Visual Interface Server (localhost:5001)")
    print("=" * 70)
    print("Press 'q' in video window to stop server.\n")

    hud_state = {
        "heart_rate": None,
        "rmssd": None,
        "arousal": 0.50,
        "valence": 0.0,
        "stress_label": "NORMAL",
        "emotion_label": "calm-positive",
        "emotion_desc": "Low arousal & positive valence",
        "transcript": "",
        "llm_response": "",
        "face_detected": False,
        "execution_provider": "Unknown"
    }
    hud_lock = threading.Lock()

    def update_hud_from_client(payload):
        with hud_lock:
            if "valence" in payload:
                hud_state["valence"] = payload["valence"]
            if "transcript" in payload:
                hud_state["transcript"] = payload["transcript"]
            if "emotion_label" in payload:
                hud_state["emotion_label"] = payload["emotion_label"]
            if "emotion_desc" in payload:
                hud_state["emotion_desc"] = payload["emotion_desc"]
            if "stress_label" in payload:
                hud_state["stress_label"] = payload["stress_label"]
            if "llm_response" in payload:
                hud_state["llm_response"] = payload["llm_response"]

    server = RPPGTCPServer(host="127.0.0.1", port=5001, on_hud_update=update_hud_from_client)
    server.start()
    broker_publisher = StateBrokerPublisher()
    broker_publisher.start()

    video_capturer = FaceROICapturer(camera_index=0)
    if not video_capturer.start():
        print("[ERROR] Webcam unavailable. Exiting rPPG server.")
        server.stop()
        broker_publisher.stop()
        return

    backend_mode = os.getenv("RPPG_BACKEND", "deep").lower()
    print(f"[rPPG Server] Active Backend Configuration Mode: '{backend_mode.upper()}'")

    deep_engine = None
    if backend_mode != "classical":
        try:
            deep_engine = DeepRPPGEngine(onnx_path="weights/efficientphys.onnx", frame_depth=10, img_size=72)
            print("[rPPG Server] Deep rPPG Engine loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Could not load Deep rPPG Engine ({e}). Falling back to Classical Green-Channel.")
            deep_engine = None

    print("\n" + "=" * 65)
    print("  [DEBUG STEP 1: STARTUP PROOF SUMMARY]")
    print(f"    1. Active Detector:           {video_capturer.detector_name}")
    print(f"    2. Deep rPPG Backend Mode:    {backend_mode.upper()}")
    print(f"    3. EfficientPhys Model Found: {os.path.exists('weights/efficientphys.onnx')}")
    print(f"    4. ONNX Active Session:       {deep_engine.session is not None if deep_engine else False}")
    print(f"    5. Execution Provider:        {deep_engine.active_provider if deep_engine else 'N/A'}")
    print(f"    6. Fallbacks Active:          {'NONE' if video_capturer.detector_name == 'MEDIAPIPE' and deep_engine and deep_engine.backend_type == 'efficientphys_onnx' else 'HAAR/CLASSICAL ACTIVE'}")
    print("=" * 65 + "\n")

    hr_smoother = HRSmoother(history_size=5, alpha=0.3)


    raw_g_signal = []
    fps_estimate = 30.0
    start_time = time.time()
    last_broadcast_time = 0
    frame_count = 0

    window_title = "webpulse - Multimodal Emotion Companion"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_title, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except Exception:
        pass

    bvp_history = []

    try:
        while True:
            ret, frame = video_capturer.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            rois, full_face_box, method = video_capturer.extract_multi_roi(frame)
            
            # Default quality meta if no face
            quality_meta = {"is_low_light": True, "mean_brightness": 0.0}
            if rois is not None:
                brightness = [v["mean_b"] for v in rois.values() if v["status"] == "USABLE"]
                if brightness:
                    mean_brightness = float(np.mean(brightness))
                    quality_meta = {"is_low_light": mean_brightness < 32.0,
                                    "mean_brightness": mean_brightness}

            display_frame = frame.copy()

            face_detected = full_face_box is not None
            current_hr = None
            current_rmssd = None
            raw_arousal = 0.50
            backend_used = "classical"
            quality_status = "GOOD"
            stress_label = "NORMAL"
            deep_res = None

            if face_detected:
                display_frame = draw_capture_guides(display_frame, full_face_box, rois)

                if deep_engine is not None:
                    # Run EfficientPhys ONNX + low-light preprocessing with all ROIs.
                    deep_res = deep_engine.process_frame(rois, quality_meta=quality_meta)
                    current_hr = deep_res["heart_rate"]
                    current_rmssd = deep_res["rmssd"]
                    raw_arousal = deep_res["arousal_score"]
                    stress_label = deep_res.get("stress_state", "NORMAL")
                    backend_used = deep_res["backend"]
                    hud_state["execution_provider"] = deep_res.get("execution_provider", "Unknown")
                    quality_status = deep_res.get("quality_status", "GOOD")
                    bvp_history.append(deep_res.get("predicted_bvp", 0.0))
                else:
                    # Classical Green-Channel Fallback (uses forehead if available)
                    roi_crop = None
                    if rois is not None and rois["forehead"]["status"] == "USABLE":
                        roi_crop = rois["forehead"]["crop"]
                    g_val = extract_roi_green_channel(roi_crop)
                    if g_val is not None:
                        raw_g_signal.append(g_val)
                        bvp_history.append(g_val)
                    if quality_meta.get("is_low_light"):
                        quality_status = "POOR_LIGHTING"

            if len(bvp_history) > 120:
                bvp_history.pop(0)

            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_estimate = frame_count / elapsed

            if deep_engine is None and len(raw_g_signal) >= int(fps_estimate * 3):
                detrended = detrend_signal(raw_g_signal)
                filtered = butterworth_bandpass_filter(detrended, fps=fps_estimate)
                
                raw_hr_bpm = estimate_heart_rate_fft(filtered, fps=fps_estimate)
                current_hr = hr_smoother.update(raw_hr_bpm)
                
                peaks = find_pulse_peaks(filtered, fps=fps_estimate)
                current_rmssd = compute_rmssd(peaks, fps=fps_estimate)
                raw_arousal = map_hrv_to_arousal(current_rmssd)

            with hud_lock:
                hud_state["heart_rate"] = current_hr
                hud_state["rmssd"] = current_rmssd
                hud_state["arousal"] = raw_arousal
                hud_state["stress_label"] = stress_label
                hud_state["face_detected"] = face_detected
                hud_state["quality_status"] = quality_status
                hud_state["backend_used"] = backend_used
                current_hud_snapshot = hud_state.copy()

            # Broadcast metrics over TCP every 1 second
            now = time.time()
            if now - last_broadcast_time >= 1.0:
                last_broadcast_time = now
                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    "heart_rate": current_hr,
                    "hrv": current_rmssd,
                    "arousal": raw_arousal,
                    "stress_state": stress_label,
                    "face_detected": face_detected,
                    "backend_used": backend_used,
                    "quality_status": quality_status,
                    "execution_provider": hud_state.get("execution_provider", "Unknown"),
                    "snr_db": deep_res.get("snr_db") if deep_res else None,
                }
                server.broadcast(payload)
                broker_publisher.publish(payload)

                hr_str = f"{current_hr:.1f} BPM" if current_hr else "Calibrating..."
                print(f"[rPPG Server ({backend_used})] Sent -> HR: {hr_str} | Arousal: {raw_arousal:.2f} | Stress: {stress_label} | Quality: {quality_status}")

            # Draw FULL WEBPULSE MODERN HUD OVERLAY INTERFACE
            display_frame = draw_overlay_hud(display_frame, current_hud_snapshot, bvp_history=bvp_history)



            cv2.imshow(window_title, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        video_capturer.stop()
        cv2.destroyAllWindows()
        server.stop()
        broker_publisher.stop()
        print("\n[rPPG Server] Stopped.")


if __name__ == "__main__":
    run_rppg_server()
