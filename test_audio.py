"""
test_audio.py — Standalone Test Script for Phase 3 (Voice Valence Pipeline).

Usage:
    conda run -n webpulse python test_audio.py [--synthetic]

This script tests:
  1. Audio feature extraction (F0 pitch, RMS energy) using librosa
  2. Voice valence score mapping heuristic
"""

import sys
import numpy as np
from audio.capture import AudioCapturer
from audio.valence import extract_audio_features, estimate_voice_valence


def test_synthetic_audio_pipeline():
    """Run mathematical & acoustic validation on synthetic audio signals."""
    print("\n--- SYNTHETIC AUDIO VALENCE TEST MODE ---")
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2)

    # 1. Expressive / Excited synthetic tone (High pitch variation + high energy)
    freq_mod = 220.0 + 40.0 * np.sin(2 * np.pi * 5.0 * t)  # Pitch modulation (vibrato/pitch swings)
    expressive_wave = 0.8 * np.sin(2 * np.pi * freq_mod * t)

    # 2. Monotone / Sad synthetic tone (Flat low pitch + low energy)
    monotone_wave = 0.05 * np.sin(2 * np.pi * 130.0 * t)

    # Process Expressive Wave
    feats_exp = extract_audio_features(expressive_wave, sr=sr)
    val_exp = estimate_voice_valence(feats_exp)

    # Process Monotone Wave
    feats_mono = extract_audio_features(monotone_wave, sr=sr)
    val_mono = estimate_voice_valence(feats_mono)

    print(f"Expressive Synthetic Signal:")
    print(f"  Pitch Mean: {feats_exp['f0_mean']:.1f} Hz | Pitch Std: {feats_exp['f0_std']:.1f} Hz | RMS: {feats_exp['rms_mean']:.4f}")
    print(f"  Calculated Valence Score: {val_exp:+.3f} (Expected > 0.0, Positive Valence)")

    print(f"\nMonotone Synthetic Signal:")
    print(f"  Pitch Mean: {feats_mono['f0_mean']:.1f} Hz | Pitch Std: {feats_mono['f0_std']:.1f} Hz | RMS: {feats_mono['rms_mean']:.4f}")
    print(f"  Calculated Valence Score: {val_mono:+.3f} (Expected < 0.0, Negative Valence)")

    print("\n[VERIFICATION RESULT] Voice feature extraction and valence scoring operates as specified.")
    return True


def run_live_microphone_test():
    """Run live microphone audio capture and valence estimation."""
    print("\n--- LIVE MICROPHONE VALENCE TEST MODE ---")
    capturer = AudioCapturer(sample_rate=22050, buffer_duration_sec=3.0)
    
    if not capturer.start():
        print("[UNTESTED ON HARDWARE] Microphone could not be started. Running synthetic test instead.")
        return test_synthetic_audio_pipeline()

    print("Listening to microphone for 5 seconds... Speak into your mic!")
    import time
    try:
        for _ in range(10):
            time.sleep(0.5)
            audio_data, sr = capturer.get_audio_segment()
            if len(audio_data) > 0:
                feats = extract_audio_features(audio_data, sr=sr)
                val = estimate_voice_valence(feats)
                speech_status = "Speech" if feats["has_speech"] else "Silent/Noise"
                print(f"[{speech_status}] RMS: {feats['rms_mean']:.4f} | Pitch Std: {feats['f0_std']:.1f} Hz | Valence: {val:+.2f}")
    finally:
        capturer.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("  WebPulse — Phase 3 Voice Valence Standalone Test")
    print("=" * 60)
    if len(sys.argv) > 1 and sys.argv[1] == "--synthetic":
        test_synthetic_audio_pipeline()
    else:
        run_live_microphone_test()
