"""WebPulse Live entry point.

The former HTTP LLM and pyttsx3 orchestration has been superseded by the
persistent Gemini Live WebSocket implementation in ``live_emotion_agent.py``.
"""

from live_emotion_agent import main


if __name__ == "__main__":
    main()
