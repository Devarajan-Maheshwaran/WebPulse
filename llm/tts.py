"""
llm/tts.py — Text-to-Speech synthesis output module.

Implements FR-8: Response Output (TTS).
Uses pyttsx3 for offline speech synthesis.
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
                engine = pyttsx3.init()
                engine.setProperty('rate', self.rate)
                engine.setProperty('volume', self.volume)
                engine.say(text)
                engine.runAndWait()
                return True
            except Exception as e:
                print(f"[ERROR] TTS speech execution error: {e}")
                return False
        else:
            print(f"[TTS DISABLED / DISPLAY ONLY] Output text: '{text}'")
            return False
