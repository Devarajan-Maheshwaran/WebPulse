"""Process 2: low-latency microphone PCM and voice-valence producer.

The microphone is opened only here. Raw 16 kHz, signed 16-bit PCM chunks are
broadcast to the Gemini Live consumer on port 5002, while periodic local voice
valence remains available as background context. Whisper is intentionally not
on this live path because its windowed transcription adds avoidable latency.
"""

import base64
import json
import queue
import socket
import threading
import time

import numpy as np

from audio.capture import AudioCapturer
from audio.valence import estimate_voice_valence, extract_audio_features


class AudioTCPServer:
    """Broadcast newline-delimited PCM and voice-state events to local clients."""

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
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"[Audio Server] Streaming PCM on {self.host}:{self.port}")

    def _accept_loop(self):
        while self.running:
            try:
                client, address = self.server_socket.accept()
                with self.lock:
                    self.clients.append(client)
                print(f"[Audio Server] Client connected from {address}")
            except OSError:
                break

    def broadcast(self, payload):
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with self.lock:
            disconnected = []
            for client in self.clients:
                try:
                    client.sendall(data)
                except OSError:
                    disconnected.append(client)
            for client in disconnected:
                self.clients.remove(client)
                client.close()

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        with self.lock:
            for client in self.clients:
                client.close()
            self.clients.clear()


def run_audio_server():
    print("[Audio Server] Process 2 — Live microphone producer. Press Ctrl+C to stop.")
    pcm_queue = queue.Queue(maxsize=50)
    server = AudioTCPServer()

    def enqueue_pcm(indata):
        pcm = indata.astype("<i2", copy=False).tobytes()
        try:
            pcm_queue.put_nowait(pcm)
        except queue.Full:
            try:
                pcm_queue.get_nowait()
            except queue.Empty:
                pass
            pcm_queue.put_nowait(pcm)

    capturer = AudioCapturer(sample_rate=16000, buffer_duration_sec=4.0, on_chunk=enqueue_pcm)
    server.start()
    if not capturer.start():
        server.stop()
        return

    last_valence_at = 0.0
    try:
        while True:
            try:
                pcm = pcm_queue.get(timeout=0.05)
                server.broadcast({
                    "type": "audio_pcm",
                    "mime_type": "audio/pcm;rate=16000",
                    "data": base64.b64encode(pcm).decode("ascii"),
                })
            except queue.Empty:
                pass

            if time.monotonic() - last_valence_at >= 2.0:
                last_valence_at = time.monotonic()
                samples, sample_rate = capturer.get_audio_segment()
                if len(samples) >= sample_rate:
                    features = extract_audio_features(samples, sr=sample_rate)
                    valence = float(estimate_voice_valence(features)) if features.get("has_speech") else 0.0
                    server.broadcast({"type": "audio_state", "valence": valence})
    except KeyboardInterrupt:
        print("\n[Audio Server] Stopped.")
    finally:
        capturer.stop()
        server.stop()


if __name__ == "__main__":
    run_audio_server()
