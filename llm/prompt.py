"""
llm/prompt.py — Legacy request/response LLM prompt construction.

Implements FR-7: LLM-Driven Response Generation using google.genai SDK.
Constructs an emotion-aware system prompt combining Russell circumplex label + transcript.
Retained only for offline compatibility. The active runtime uses Gemini Live in
live_emotion_agent.py and does not invoke this module.
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Keep prompt construction usable for offline/static checks before setup.
    pass

SYSTEM_PROMPT = (
    "You are WebPulse, a warm, conversational live companion. Respond in short, natural turns "
    "(one or two complete sentences), acknowledge what the person said, and use body and voice "
    "signals as context rather than a diagnosis. Never claim to know a person's health or mood with certainty."
)


def construct_prompt(emotion_info, transcript):
    """
    Construct the final LLM prompt combining detected emotion label, quality status & live rPPG telemetry.
    """
    label = emotion_info.get("label", "neutral")
    desc = emotion_info.get("description", "")
    arousal = emotion_info.get("arousal", 0.5)
    valence = emotion_info.get("valence", 0.0)
    quality = emotion_info.get("quality_status", "GOOD")
    hr = emotion_info.get("heart_rate")
    rmssd = emotion_info.get("rmssd")
    stress_label = emotion_info.get("stress_label", "NORMAL")

    user_speech = transcript.strip() if transcript and transcript.strip() else "(User is listening / quiet)"
    hr_str = f"{hr:.1f} BPM" if hr else "Calibrating"
    hrv_str = f"{rmssd:.1f} ms" if rmssd else "Calibrating"

    quality_note = ""
    if quality == "POOR_LIGHTING":
        quality_note = "\n[Signal Note: Webcam lighting is dim/low; rely primarily on speech tone and conversational context.]"
    elif quality == "WEAK_SIGNAL":
        quality_note = "\n[Signal Note: Facial rPPG signal quality is weak; respond gently without claiming precise physical state.]"

    state = {
        "heart_rate": hr,
        "hrv_rmssd": rmssd,
        "wesad_stress_label": stress_label,
        "arousal_score": arousal,
        "signal_quality": quality,
        "voice_valence": valence,
        "transcript": user_speech,
    }
    prompt = f"""[System Instruction]
{SYSTEM_PROMPT}

[Grounded live state]
{state}
Plain-language interpretation: {desc or ('calm' if arousal < 0.5 else 'elevated')} ({stress_label.lower()})
Live Heart Rate: {hr_str} | HRV (RMSSD): {hrv_str}
Physiological Arousal: {arousal:.2f} (0.0=Calm, 1.0=Stressed)
Voice Tone Valence: {valence:+.2f} (-1.0=Negative, +1.0=Positive)
Signal quality: {quality}{quality_note}

[User Speech Input]
"{user_speech}"

Respond directly in a warm, responsive live-assistant style. Be cautious and say you may be unsure when signal quality is weak or lighting is poor."""
    return prompt


class LLMResponseGenerator:
    """
    Generates empathetic responses using live LLM APIs (Gemini / OpenAI / Anthropic)
    with a warm offline companion fallback when quota limits or network errors occur.
    """

    def __init__(self, provider=None, model=None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "gemini").lower()
        self.model = model or os.getenv("LLM_MODEL", "gemini-2.5-flash")

    def generate_offline_fallback(self, emotion_info, transcript):
        """Rule-based empathetic fallback when API rate limit or offline mode is reached."""
        label = emotion_info.get("label", "neutral")
        stress = emotion_info.get("stress_label", "NORMAL")
        hr = emotion_info.get("heart_rate")
        user_speech = transcript.strip() if transcript and transcript.strip() else ""

        hr_note = f" (HR ~{int(hr)} BPM)" if hr and 50 <= hr <= 120 else ""

        if stress == "STRESSED" or "aroused-negative" in label:
            if user_speech:
                return f"I hear you saying '{user_speech}'. I notice your physical signals reflect some stress{hr_note} — take a slow, deep breath, I'm right here with you."
            return f"I notice your physiological signals indicate elevated stress right now{hr_note}. Take a gentle pause and let yourself breathe."
        elif "anxious" in label or "excited" in label or "aroused-positive" in label:
            if user_speech:
                return f"I hear the energy in what you shared: '{user_speech}'. I'm listening closely to you."
            return "I'm picking up high energy in your signals right now. I'm right here with you."
        elif "calm-negative" in label or "sad" in label:
            if user_speech:
                return f"Thank you for sharing that with me: '{user_speech}'. Please be gentle with yourself today."
            return "It sounds like things might feel a bit quiet or heavy. I'm right here by your side."
        else:
            if user_speech:
                return f"I hear you saying '{user_speech}'. I'm glad we can share this quiet moment together."
            return "I'm right here with you. Feel free to talk whenever you're ready."


    def generate_response(self, emotion_info, transcript):
        """
        Generate empathetic response string from live LLM.
        """
        prompt = construct_prompt(emotion_info, transcript)

        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

        is_openai_valid = bool(openai_key and not openai_key.startswith("your-"))
        is_anthropic_valid = bool(anthropic_key and not anthropic_key.startswith("your-"))
        is_gemini_valid = bool(gemini_key and not gemini_key.startswith("your-"))

        if self.provider == "gemini":
            if not is_gemini_valid:
                print("[LLM NOTE] GEMINI_API_KEY not set. Using offline companion generator.")
                return self.generate_offline_fallback(emotion_info, transcript)

            # Try google.genai (new SDK)
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                err_str = str(e)
                print(f"[LLM API WARNING] Gemini API call issue ({err_str[:80]}...). Using offline fallback.")
                return self.generate_offline_fallback(emotion_info, transcript)

            # Fallback to legacy google.generativeai if genai import fails
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=gemini_key)
                model = genai_legacy.GenerativeModel(model_name=self.model)
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"[LLM API WARNING] Legacy Gemini API issue. Using offline fallback.")
                return self.generate_offline_fallback(emotion_info, transcript)

        elif self.provider == "openai":
            if not is_openai_valid:
                return self.generate_offline_fallback(emotion_info, transcript)
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=250
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[LLM API WARNING] OpenAI API issue: {e}. Using offline fallback.")
                return self.generate_offline_fallback(emotion_info, transcript)

        elif self.provider == "anthropic":
            if not is_anthropic_valid:
                return self.generate_offline_fallback(emotion_info, transcript)
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                response = client.messages.create(
                    model=self.model if "claude" in self.model else "claude-3-haiku-20240307",
                    max_tokens=250,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            except Exception as e:
                print(f"[LLM API WARNING] Anthropic API issue: {e}. Using offline fallback.")
                return self.generate_offline_fallback(emotion_info, transcript)

        return self.generate_offline_fallback(emotion_info, transcript)
