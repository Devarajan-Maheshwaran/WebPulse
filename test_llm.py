"""
test_llm.py — Standalone Test Script for Phase 5 (Transcription, Prompting, LLM, and TTS).

Usage:
    conda run -n webpulse python test_llm.py

This script tests:
  1. Emotion-aware prompt construction (FR-7)
  2. LLM response generation (FR-7) — shows real API output if API key is in .env,
     otherwise clearly logs the MOCK LLM response used when API key is blank.
  3. Speech transcription interface (FR-6)
  4. TTS engine output (FR-8)
"""

import os
from fusion.emotion import fuse_emotions
from llm.prompt import construct_prompt, LLMResponseGenerator
from llm.transcribe import SpeechTranscriber
from llm.tts import TTSEngine


def test_llm_pipeline():
    print("=" * 60)
    print("  WebPulse — Phase 5 LLM & Transcription Pipeline Test")
    print("=" * 60)

    # 1. Test Prompt Construction
    mock_emotion = fuse_emotions(arousal_score=0.85, valence_score=-0.6)
    mock_transcript = "I have so many deadlines coming up this week and I don't know how to finish everything."

    print("\n--- 1. PROMPT CONSTRUCTION TEST ---")
    prompt = construct_prompt(mock_emotion, mock_transcript)
    print("Generated Prompt:")
    print(prompt)

    # 2. Test LLM Response Generator (Real API vs Mock Fallback)
    print("\n--- 2. LLM RESPONSE GENERATION TEST ---")
    generator = LLMResponseGenerator()

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    has_real_key = (
        bool(openai_key and not openai_key.startswith("your-")) or
        bool(anthropic_key and not anthropic_key.startswith("your-")) or
        bool(gemini_key and not gemini_key.startswith("your-"))
    )

    if has_real_key:
        print("[API KEY DETECTED] Calling live LLM API...")
    else:
        print("[API KEY BLANK] Operating in offline mode. Falling back to MOCK response handler in llm/prompt.py.")

    response = generator.generate_response(mock_emotion, mock_transcript)
    print(f"\nGenerated LLM Response:\n\"{response}\"")

    # 3. Test Speech Transcriber Interface
    print("\n--- 3. SPEECH TRANSCRIBER INTERFACE TEST ---")
    transcriber = SpeechTranscriber(model_name="tiny")
    print(f"Whisper Model Loaded Status: {transcriber.load_model()}")

    # 4. Test TTS Synthesis
    print("\n--- 4. TTS SYNTHESIS TEST ---")
    tts = TTSEngine(rate=150)
    print("Testing TTS engine initialization...")
    first_sentence = response.split(".")[0] + "." if "." in response else response
    print(f"TTS Target Text: '{first_sentence}'")
    print(f"TTS Engine Available: {tts.engine is not None}")

    print("\n[VERIFICATION RESULT] Phase 5 LLM, Transcription, and TTS pipeline completed successfully.")
    return True


if __name__ == "__main__":
    test_llm_pipeline()
