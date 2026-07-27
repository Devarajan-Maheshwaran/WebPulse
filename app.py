"""
app.py — Process 3: Fusion + LLM + Session Logging Consumer.

Runs as Process 3 (Main Entry Point):
  - Connects as a TCP Socket Client to:
      - Process 1: rPPG Producer Server (localhost:5001)
      - Process 2: Audio Producer Server (localhost:5002)
  - Receives live arousal_score, valence_score, and speech transcripts.
  - Performs Multimodal Emotion Fusion (Russell's Circumplex Model).
  - Triggers Emotion-Aware LLM Response Generation (Gemini/OpenAI/Anthropic) + TTS.
  - Logs complete session entries into sessions/ directory (CSV & JSON).
"""

import sys
import time
import json
import socket
import argparse
import threading

from fusion.emotion import fuse_emotions
from llm.prompt import LLMResponseGenerator
from llm.tts import TTSEngine
from logging_.session_logger import SessionLogger


class TCPClientSubscriber:
    """Client subscriber that connects to a TCP producer server and reads JSON lines."""

    def __init__(self, host, port, callback, name="Subscriber"):
        self.host = host
        self.port = port
        self.callback = callback
        self.name = name
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        while self.running:
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
                print(f"[{self.name}] Connected to server at {self.host}:{self.port}")
                
                buffer = ""
                while self.running:
                    data = sock.recv(1024).decode("utf-8")
                    if not data:
                        break
                    buffer += data
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            try:
                                payload = json.loads(line.strip())
                                self.callback(payload)
                            except Exception as pe:
                                print(f"[{self.name}] JSON parse error: {pe}")
            except Exception:
                # Retry connection after 2 seconds
                time.sleep(2.0)
            finally:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

    def stop(self):
        self.running = False


def run_live_fusion_consumer(subject_id="subject_pilot", enable_tts=True):
    print("\n" + "=" * 70)
    print("  webpulse — Process 3: Fusion + LLM Companion & Logging Consumer")
    print("=" * 70)
    print("Waiting for rPPG Server (port 5001) and Audio Server (port 5002)...")
    print("Press Ctrl+C to stop.\n")

    logger = SessionLogger(output_dir="sessions")
    session_id = logger.start_session(subject_id=subject_id)
    print(f"Started Session: {session_id}")

    llm_gen = LLMResponseGenerator()
    tts = TTSEngine(rate=150) if enable_tts else None
    print(f"Using LLM Provider: {llm_gen.provider.upper()} (Model: {llm_gen.model})\n")

    latest_rppg = {
        'heart_rate': None,
        'hrv': None,
        'arousal': 0.5,
        'face_detected': False
    }

    latest_audio = {
        'valence': 0.0,
        'transcript': ""
    }

    state_lock = threading.Lock()
    is_llm_busy = False

    def on_rppg_data(data):
        with state_lock:
            latest_rppg['heart_rate'] = data.get('heart_rate')
            latest_rppg['hrv'] = data.get('hrv')
            latest_rppg['arousal'] = data.get('arousal', 0.5)
            latest_rppg['face_detected'] = data.get('face_detected', False)

    def on_audio_data(data):
        nonlocal is_llm_busy
        with state_lock:
            latest_audio['valence'] = data.get('valence', 0.0)
            transcript = data.get('transcript', '')
            if transcript:
                latest_audio['transcript'] = transcript

        # Trigger Fusion & LLM on new audio packet if not busy
        if not is_llm_busy:
            is_llm_busy = True
            threading.Thread(target=process_fusion_and_llm, daemon=True).start()

    def process_fusion_and_llm():
        nonlocal is_llm_busy
        try:
            with state_lock:
                arousal = latest_rppg['arousal']
                valence = latest_audio['valence']
                transcript = latest_audio['transcript']
                hr = latest_rppg['heart_rate']
                hrv = latest_rppg['hrv']

            fused_emotion = fuse_emotions(arousal, valence)

            print("\n" + "=" * 60)
            print(f"[FUSION EVENT] Triggered Multimodal Emotion Analysis")
            print(f"  Physiological Arousal: {arousal:.2f} | Voice Valence: {valence:+.2f}")
            print(f"  Fused Emotion State:   '{fused_emotion['label']}'")
            print(f"  User Transcript:       '{transcript or '(No speech transcribed)'}'")

            response_text = llm_gen.generate_response(fused_emotion, transcript)
            print(f"[LLM RESPONSE] \"{response_text}\"")
            print("=" * 60 + "\n")

            if enable_tts and tts is not None and response_text and not response_text.startswith("["):
                tts.speak(response_text)

            logger.log_entry(
                heart_rate=hr,
                rmssd=hrv,
                arousal_score=arousal,
                valence_score=valence,
                emotion_label=fused_emotion['label'],
                transcript=transcript,
                llm_response=response_text
            )
        finally:
            is_llm_busy = False

    rppg_client = TCPClientSubscriber("127.0.0.1", 5001, on_rppg_data, name="rPPG Client")
    audio_client = TCPClientSubscriber("127.0.0.1", 5002, on_audio_data, name="Audio Client")

    rppg_client.start()
    audio_client.start()

    try:
        while True:
            time.sleep(1.0)
            with state_lock:
                hr_val = f"{latest_rppg['heart_rate']:.1f} BPM" if latest_rppg['heart_rate'] else "Calibrating"
                print(f"[Consumer Status] HR: {hr_val} | Arousal: {latest_rppg['arousal']:.2f} | Valence: {latest_audio['valence']:+.2f}")

    except KeyboardInterrupt:
        print("\n[Process 3] Stopping Consumer...")
    finally:
        rppg_client.stop()
        audio_client.stop()
        summary = logger.stop_session()
        if summary:
            print("\nSession Exported:")
            print(f"  JSON: {summary['json_path']}")
            print(f"  CSV:  {summary['csv_path']}")


def main():
    parser = argparse.ArgumentParser(description="webpulse Process 3 — Fusion + LLM + Logging Consumer")
    parser.add_argument("--subject", type=str, default="subject_pilot", help="Test subject identifier")
    parser.add_argument("--no-tts", action="store_true", help="Disable spoken TTS output")
    args = parser.parse_args()

    run_live_fusion_consumer(subject_id=args.subject, enable_tts=not args.no_tts)


if __name__ == "__main__":
    main()
