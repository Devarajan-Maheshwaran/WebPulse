"""
llm/prompt.py — LLM prompt construction and API call integration.

Implements FR-7: LLM-Driven Response Generation using google.genai SDK.
Constructs an emotion-aware system prompt combining Russell circumplex label + transcript.
Supports Gemini / OpenAI / Anthropic API calls reading from .env.
"""

import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are WebPulse, an empathetic, caring, and emotionally-intelligent AI companion. "
    "Your role is to offer warm, supportive conversational responses based on the user's speech "
    "and their estimated physiological emotion state (arousal and valence). "
    "Keep your response concise (2-3 full sentences maximum) and natural. Always finish your complete sentences. "
    "Do not sound clinical, and do not explicitly mention physiological metrics unless asked."
)


def construct_prompt(emotion_info, transcript):
    """
    Construct the final LLM prompt combining detected emotion label & transcript.
    """
    label = emotion_info.get("label", "neutral")
    desc = emotion_info.get("description", "")
    arousal = emotion_info.get("arousal", 0.5)
    valence = emotion_info.get("valence", 0.0)

    user_speech = transcript.strip() if transcript and transcript.strip() else "(User is listening / quiet)"

    prompt = f"""[System Instruction]
{SYSTEM_PROMPT}

[Detected Physiological & Voice Emotion Context]
Emotion Label: {label}
Arousal/Stress Level: {arousal:.2f} (0.0=Calm, 1.0=Stressed)
Voice Valence: {valence:+.2f} (-1.0=Negative, +1.0=Positive)
Context: {desc}

[User Speech Transcript]
"{user_speech}"

Provide a warm, empathetic 2-sentence response directly to the user."""
    return prompt


class LLMResponseGenerator:
    """
    Generates empathetic responses using live LLM APIs (Gemini / OpenAI / Anthropic).
    """

    def __init__(self, provider=None, model=None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "gemini").lower()
        self.model = model or os.getenv("LLM_MODEL", "gemini-2.5-flash")

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
                error_msg = "[LLM ERROR] GEMINI_API_KEY is not configured in .env file."
                print(f"{error_msg}")
                return error_msg

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
                if "429" in err_str or "quota" in err_str.lower():
                    error_msg = "[Gemini Rate Limit Reached — Waiting for quota reset...]"
                else:
                    error_msg = f"[LLM ERROR] Gemini API call failed: {e}"
                print(f"{error_msg}")
                return error_msg

            # Fallback to legacy google.generativeai if genai import fails
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=gemini_key)
                model = genai_legacy.GenerativeModel(model_name=self.model)
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    error_msg = "[Gemini Rate Limit Reached — Waiting for quota reset...]"
                else:
                    error_msg = f"[LLM ERROR] Gemini API call failed: {e}"
                print(f"{error_msg}")
                return error_msg

        elif self.provider == "openai":
            if not is_openai_valid:
                error_msg = "[LLM ERROR] OPENAI_API_KEY is not configured in .env file."
                print(f"{error_msg}")
                return error_msg

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
                error_msg = f"[LLM ERROR] OpenAI API call failed: {e}"
                print(f"{error_msg}")
                return error_msg

        elif self.provider == "anthropic":
            if not is_anthropic_valid:
                error_msg = "[LLM ERROR] ANTHROPIC_API_KEY is not configured in .env file."
                print(f"{error_msg}")
                return error_msg

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
                error_msg = f"[LLM ERROR] Anthropic API call failed: {e}"
                print(f"{error_msg}")
                return error_msg

        else:
            error_msg = f"[LLM ERROR] Unknown provider '{self.provider}'."
            print(f"{error_msg}")
            return error_msg
