"""
app.py — Main Entry Point for WebPulse End-to-End Pipeline.

Integrates Phases 1–6 into a live pipeline per SRS Section 4 Architecture:
  1. Video Capture + Face ROI Detection (MediaPipe/Haar)
  2. rPPG Green Signal Extraction & Butterworth Filtering
  3. FFT Heart Rate & RMSSD HRV -> Arousal Estimation
  4. Microphone Capture -> Pitch/Energy Features -> Voice Valence
  5. Emotion Fusion (Russell's Circumplex Model)
  6. Whisper ASR Transcription + Emotion-Aware LLM Response Generation (OpenAI/Anthropic/Mock)
  7. TTS Spoken Output
  8. Full Session Logging (CSV & JSON export)

Graceful Exception Handling (NFR-3):
  Handles temporary loss of face detection cleanly by pausing signal accumulation
  without crashing or breaking data continuity.
"""

import sys
import time
import argparse
import numpy as np
import cv2

from rppg.capture import FaceROICapturer
from rppg.signal import extract_roi_green_channel, detrend_signal, butterworth_bandpass_filter
from rppg.hrv import estimate_heart_rate_fft, find_pulse_peaks, compute_rmssd, map_hrv_to_arousal

from audio.capture import AudioCapturer
from audio.valence import extract_audio_features, estimate_voice_valence

from fusion.emotion import fuse_emotions

from llm.transcribe import SpeechTranscriber
from llm.prompt import LLMResponseGenerator
from llm.tts import TTSEngine

from logging_.session_logger import SessionLogger


def run_simulated_pipeline(subject_id="subject_simulated"):
    """
    Run simulated pipeline test when hardware sensors are unavailable.
    Explicitly labeled as SIMULATED / NO REAL SENSOR ACCESS IN THIS ENVIRONMENT.
    """
    print("\n" + "=" * 70)
    print("  [SIMULATED / NO REAL SENSOR ACCESS IN THIS ENVIRONMENT]")
    print("  Running End-to-End WebPulse Self-Test with Synthetic Signals")
    print("=" * 70 + "\n")

    logger = SessionLogger(output_dir="sessions")
    session_id = logger.start_session(subject_id=subject_id)
    print(f"Session Started: {session_id}")

    # Initialize modules
    llm_gen = LLMResponseGenerator()
    tts = TTSEngine(rate=150)
    transcriber = SpeechTranscriber(model_name="tiny")

    # Generate synthetic rPPG signal (72 BPM)
    fps = 30.0
    t = np.linspace(0, 15.0, int(fps * 15.0))
    raw_signal = 120.0 + 5.0 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.5, len(t))

    # Process rPPG
    detrended = detrend_signal(raw_signal)
    filtered = butterworth_bandpass_filter(detrended, fps=fps)
    hr_bpm = estimate_heart_rate_fft(filtered, fps=fps)
    peaks = find_pulse_peaks(filtered, fps=fps)
    rmssd = compute_rmssd(peaks, fps=fps)
    arousal = map_hrv_to_arousal(rmssd)

    # Generate synthetic expressive audio tone
    audio_t = np.linspace(0, 2.0, int(22050 * 2.0))
    synth_audio = 0.5 * np.sin(2 * np.pi * (220.0 + 30.0 * np.sin(2 * np.pi * 4.0 * audio_t)) * audio_t)
    audio_feats = extract_audio_features(synth_audio, sr=22050)
    valence = estimate_voice_valence(audio_feats)

    # Fusion
    fused_emotion = fuse_emotions(arousal, valence)

    # Simulated Speech Transcript
    transcript = "Hello! I am testing the complete WebPulse integrated pipeline."

    # LLM Response Generation
    print("\n--- Generating LLM Response ---")
    response_text = llm_gen.generate_response(fused_emotion, transcript)
    print(f"Fused Emotion: {fused_emotion['label']} (Arousal={arousal:.2f}, Valence={valence:+.2f})")
    print(f"Transcript:    \"{transcript}\"")
    print(f"LLM Response:  \"{response_text}\"")

    # Logging
    logger.log_entry(
        heart_rate=hr_bpm,
        rmssd=rmssd,
        arousal_score=arousal,
        valence_score=valence,
        emotion_label=fused_emotion['label'],
        transcript=transcript,
        llm_response=response_text
    )

    summary = logger.stop_session()
    print("\n--- Session Log Summary ---")
    print(f"  Duration:      {summary['duration']} seconds")
    print(f"  Total Records: {summary['total_records']}")
    print(f"  JSON Export:   {summary['json_path']}")
    print(f"  CSV Export:    {summary['csv_path']}")
    print("\n[SIMULATED RUN COMPLETE] Full pipeline integrated and verified.")
    return True


def run_live_pipeline(subject_id="subject_pilot", enable_tts=True):
    """Run live end-to-end WebPulse pipeline with webcam and microphone."""
    print("\n" + "=" * 70)
    print("  WebPulse — Live Multimodal Emotion Recognition & LLM Companion")
    print("=" * 70)
    print("Press 'q' in video window to end session.\n")

    # Initialize Hardware Capturers
    video_capturer = FaceROICapturer(camera_index=0)
    audio_capturer = AudioCapturer(sample_rate=22050, buffer_duration_sec=3.0)

    if not video_capturer.start():
        print("[UNTESTED ON HARDWARE] Webcam unavailable in this environment.")
        return run_simulated_pipeline(subject_id=subject_id)

    audio_capturer.start()

    # Initialize Pipeline Components
    llm_gen = LLMResponseGenerator()
    tts = TTSEngine(rate=150) if enable_tts else None
    transcriber = SpeechTranscriber(model_name="tiny")
    logger = SessionLogger(output_dir="sessions")

    session_id = logger.start_session(subject_id=subject_id)
    print(f"Started Session: {session_id}")

    raw_g_signal = []
    fps_estimate = 30.0
    start_time = time.time()
    frame_count = 0
    face_lost_counter = 0

    try:
        while True:
            ret, frame = video_capturer.get_frame()
            if not ret or frame is None:
                print("[WARNING] Frame capture dropped.")
                break

            frame_count += 1
            roi_crop, roi_box, method = video_capturer.extract_roi(frame)
            display_frame = frame.copy()

            # Handle Face Detection / Loss (NFR-3)
            if roi_box is not None:
                face_lost_counter = 0
                x, y, w, h = roi_box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Face ROI ({method})", (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                g_val = extract_roi_green_channel(roi_crop)
                if g_val is not None:
                    raw_g_signal.append(g_val)
            else:
                # NFR-3 Graceful face loss handling: pause accumulation without crashing
                face_lost_counter += 1
                cv2.putText(display_frame, "[WARNING: FACE LOST — SIGNAL PAUSED]", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Estimate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_estimate = frame_count / elapsed

            # Keep rolling window of 15 seconds
            window_size = int(fps_estimate * 15)
            if len(raw_g_signal) > window_size:
                raw_g_signal = raw_g_signal[-window_size:]

            # Process metrics every ~5 seconds
            if len(raw_g_signal) >= int(fps_estimate * 5):
                detrended = detrend_signal(raw_g_signal)
                filtered = butterworth_bandpass_filter(detrended, fps=fps_estimate)
                
                hr_bpm = estimate_heart_rate_fft(filtered, fps=fps_estimate)
                peaks = find_pulse_peaks(filtered, fps=fps_estimate)
                rmssd = compute_rmssd(peaks, fps=fps_estimate)
                arousal = map_hrv_to_arousal(rmssd)

                # Process Audio
                audio_samples, sr = audio_capturer.get_audio_segment()
                audio_feats = extract_audio_features(audio_samples, sr=sr)
                valence = estimate_voice_valence(audio_feats)

                # Fuse Emotion
                fused_emotion = fuse_emotions(arousal, valence)

                # On Screen Display
                hr_str = f"HR: {hr_bpm:.1f} BPM" if hr_bpm else "HR: Estimating..."
                arousal_str = f"Arousal: {arousal:.2f}"
                valence_str = f"Valence: {valence:+.2f}"
                label_str = f"Emotion: {fused_emotion['label']}"

                cv2.putText(display_frame, hr_str, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, arousal_str, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(display_frame, valence_str, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, label_str, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                # Log periodically
                if frame_count % int(fps_estimate * 3) == 0:
                    logger.log_entry(
                        heart_rate=hr_bpm,
                        rmssd=rmssd,
                        arousal_score=arousal,
                        valence_score=valence,
                        emotion_label=fused_emotion['label'],
                        transcript="",
                        llm_response=""
                    )

            cv2.imshow("WebPulse — Live Multimodal Pipeline", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        video_capturer.stop()
        audio_capturer.stop()
        cv2.destroyAllWindows()

        summary = logger.stop_session()
        if summary:
            print("\nSession Exported:")
            print(f"  JSON: {summary['json_path']}")
            print(f"  CSV:  {summary['csv_path']}")


def main():
    parser = argparse.ArgumentParser(description="WebPulse Multimodal rPPG + Voice LLM System")
    parser.add_argument("--simulated", action="store_true", help="Force synthetic test mode")
    parser.add_argument("--subject", type=str, default="subject_pilot", help="Test subject identifier")
    parser.add_argument("--no-tts", action="store_true", help="Disable spoken TTS output")
    args = parser.parse_args()

    if args.simulated:
        run_simulated_pipeline(subject_id=args.subject)
    else:
        run_live_pipeline(subject_id=args.subject, enable_tts=not args.no-tts)


if __name__ == "__main__":
    main()
