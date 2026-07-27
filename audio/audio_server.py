"""
audio/audio_server.py — Process 2: Audio & Valence Producer Server.

Runs as Process 2:
  - Captures microphone audio in rolling 4-second segments.
  - Extracts pitch (F0) & energy (RMS) features via librosa to compute voice valence_score.
  - Runs OpenAI Whisper ASR transcription to convert audio to transcript string.
  - Hosts a TCP Socket Server on localhost:5002 broadcasting Audio/ASR payload as JSON lines.
"""

import sys
import os
import time
import json
import socket
import threading

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio.capture import AudioCapturer
from audio.valence import extract_audio_features, estimate_voice_valence
from llm.transcribe import SpeechTranscriber


class AudioTCPServer:
    """Simple multi-client TCP Socket Server for broadcasting Audio/ASR data."""

    def __init__(self, host="127.0.0.1", port=5002):
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
        print(f"[Audio Server] Listening for TCP connections on {self.host}:{self.port}...")

        accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        accept_thread.start()

    def _accept_loop(self):
        while self.running:
            try:
                client_sock, addr = self.server_socket.accept()
                with self.lock:
                    self.clients.append(client_sock)
                print(f"[Audio Server] Client connected from {addr}")
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


def run_audio_server():
    print("=" * 70)
    print("  Process 2: WebPulse Audio & Valence Producer Server (localhost:5002)")
    print("=" * 70)
    print("Press Ctrl+C in terminal to stop server.\n")

    server = AudioTCPServer(host="127.0.0.1", port=5002)
    server.start()

    audio_capturer = AudioCapturer(sample_rate=22050, buffer_duration_sec=4.0)
    if not audio_capturer.start():
        print("[ERROR] Microphone unavailable. Exiting Audio server.")
        server.stop()
        return

    transcriber = SpeechTranscriber(model_name="tiny")
    print("[Audio Server] Pre-loading Whisper ASR model...")
    transcriber.load_model()
    print("[Audio Server] Model loaded. Starting audio capture loop...\n")

    try:
        while True:
            time.sleep(4.0)  # Segment every 4 seconds
            audio_samples, sr = audio_capturer.get_audio_segment()

            if len(audio_samples) < int(sr * 0.5):
                continue

            # Extract valence
            audio_feats = extract_audio_features(audio_samples, sr=sr)
            valence_score = estimate_voice_valence(audio_feats)

            # Transcribe speech
            transcript = transcriber.transcribe(audio_samples, sr=sr)

            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "valence": float(valence_score),
                "transcript": transcript or ""
            }

            server.broadcast(payload)
            print(f"[Audio Server] Sent -> Valence: {valence_score:+.2f} | Transcript: '{transcript or '(Silence)'}'")

    except KeyboardInterrupt:
        print("\n[Audio Server] Keyboard interrupt received.")
    finally:
        audio_capturer.stop()
        server.stop()
        print("[Audio Server] Stopped.")


if __name__ == "__main__":
    run_audio_server()
