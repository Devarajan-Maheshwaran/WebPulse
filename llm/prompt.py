"""
llm/prompt.py — LLM prompt construction and API call integration.

Implements FR-7: LLM-Driven Response Generation.
Constructs an emotion-aware system prompt combining Russell circumplex label + transcript.
Supports OpenAI / Anthropic API calls reading from .env.

NOTE ON MOCK RESPONSES:
If OPENAI_API_KEY or ANTHROPIC_API_KEY is blank or unconfigured in .env,
the system gracefully uses the fallback method `_get_mock_response()` in this file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are WebPulse, an empathetic, caring, and emotionally-intelligent AI companion. "
    "Your role is to offer warm, supportive conversational responses based on the user's speech "
    "and their estimated physiological emotion state (arousal and valence). "
    "Keep your response concise (2-3 sentences maximum) and natural. "
    "Do not sound clinical, and do not explicitly mention physiological metrics unless asked."
)


def construct_prompt(emotion_info, transcript):
    """
    Construct the final LLM prompt combining detected emotion label & transcript.
    
    Args:
        emotion_info (dict): Fused emotion dictionary from fusion.emotion.fuse_emotions.
        transcript (str): Transcribed user speech text.
        
    Returns:
        str: Formatted user prompt for LLM.
    """
    label = emotion_info.get("label", "neutral")
    desc = emotion_info.get("description", "")
    arousal = emotion_info.get("arousal", 0.5)
    valence = emotion_info.get("valence", 0.0)

    user_speech = transcript.strip() if transcript and transcript.strip() else "(User was quiet / no speech detected)"

    prompt = f"""[Detected Physiological & Voice Emotion Context]
Emotion Label: {label}
Arousal/Stress Level: {arousal:.2f} (0.0=Calm, 1.0=Stressed)
Voice Valence: {valence:+.2f} (-1.0=Negative, +1.0=Positive)
Context: {desc}

[User Speech Transcript]
"{user_speech}"

Please respond empathetically to the user in a warm, comforting tone."""
    return prompt


class LLMResponseGenerator:
    """
    Generates empathetic responses using OpenAI / Anthropic API,
    or returns a clearly labeled fallback mock response when API keys are blank.
    """

    def __init__(self, provider=None, model=None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    def generate_response(self, emotion_info, transcript):
        """
        Generate empathetic response string.
        
        Args:
            emotion_info (dict): Emotion dictionary from fusion module.
            transcript (str): Speech transcript.
            
        Returns:
            str: Generated empathetic text response.
        """
        prompt = construct_prompt(emotion_info, transcript)

        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

        # Check if keys are non-empty and non-placeholder
        is_openai_valid = bool(openai_key and not openai_key.startswith("your-"))
        is_anthropic_valid = bool(anthropic_key and not anthropic_key.startswith("your-"))

        if self.provider == "openai" and is_openai_valid:
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
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[ERROR] OpenAI API call failed: {e}")

        elif self.provider == "anthropic" and is_anthropic_valid:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                response = client.messages.create(
                    model=self.model if "claude" in self.model else "claude-3-haiku-20240307",
                    max_tokens=150,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            except Exception as e:
                print(f"[ERROR] Anthropic API call failed: {e}")

        # Fallback to Mock Response when API key is blank/unconfigured
        label = emotion_info.get("label", "neutral")
        return self._get_mock_response(label, transcript)

    def _get_mock_response(self, emotion_label, transcript):
        """
        MOCK RESPONSE GENERATOR (Used during development / when API key is left blank).
        """
        mock_responses = {
            "aroused-positive": (
                "[MOCK LLM RESPONSE - API KEY BLANK] That sounds wonderfully exciting! "
                "I can hear the enthusiasm in your voice, and I'm really happy for you."
            ),
            "aroused-negative": (
                "[MOCK LLM RESPONSE - API KEY BLANK] I hear that you might be feeling stressed or overwhelmed right now. "
                "Take a slow deep breath — I am right here with you."
            ),
            "calm-positive": (
                "[MOCK LLM RESPONSE - API KEY BLANK] You seem very relaxed and content right now. "
                "It is really nice to share a peaceful moment with you."
            ),
            "calm-negative": (
                "[MOCK LLM RESPONSE - API KEY BLANK] You sound a bit tired or down right now. "
                "Please take it easy on yourself today, and know I am here if you want to chat."
            )
        }
        return mock_responses.get(
            emotion_label,
            "[MOCK LLM RESPONSE - API KEY BLANK] I am listening. How are you feeling today?"
        )
