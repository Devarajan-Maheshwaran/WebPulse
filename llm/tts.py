"""
llm/tts.py — Deprecated local Text-to-Speech compatibility module.

Implements FR-8: Response Output (TTS).
The active runtime uses Gemini Live native audio in live_emotion_agent.py;
this module remains for non-live/offline compatibility only.
"""

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


class TTSEngine:
    """
    Offline Text-to-Speech Engine wrapper around pyttsx3.
    """

    def __init__(self, rate=150, volume=0.9):
        self.rate = rate
        self.volume = volume
        self.engine = HAS_PYTTSX3

    def speak(self, text):
        """
        Synthesize text response to spoken audio.
        
        Args:
            text (str): Response text to speak.
            
        Returns:
            bool: True if speech was executed, False otherwise.
        """
        if not text or not text.strip():
            return False

        if HAS_PYTTSX3:
            try:
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                except Exception:
                    pass

                engine = pyttsx3.init()
                engine.setProperty('rate', self.rate)
                engine.setProperty('volume', self.volume)
                engine.say(text)
                engine.runAndWait()
                return True
            except Exception as e:
                print(f"[TTS WARNING] Speech synthesis issue: {e}")
                return False

        else:
            print(f"[TTS DISABLED / DISPLAY ONLY] Output text: '{text}'")
            return False
