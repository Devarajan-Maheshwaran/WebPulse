"""
llm/transcribe.py — Speech-to-text transcription module.

Implements FR-6: Speech Transcription using OpenAI Whisper.
"""

import tempfile
import numpy as np

try:
    import soundfile as sf
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


class SpeechTranscriber:
    """
    Speech-to-text transcription engine using OpenAI Whisper (local model).
    """

    def __init__(self, model_name="tiny"):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Lazy load Whisper model on first use."""
        if HAS_WHISPER and self.model is None:
            try:
                self.model = whisper.load_model(self.model_name)
                return True
            except Exception as e:
                print(f"[WARNING] Could not load Whisper model '{self.model_name}': {e}")
                return False
        return self.model is not None

    def transcribe(self, audio_data, sr=22050):
        """
        Transcribe a 1D audio sample array into text.
        
        Args:
            audio_data (ndarray 1D float32): Raw audio samples.
            sr (int): Sampling rate.
            
        Returns:
            str: Transcribed text output.
        """
        if audio_data is None or len(audio_data) < int(sr * 0.5):
            return ""

        if not self.load_model():
            return "[ASR Mock: Audio detected, Whisper model not active]"

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                sf.write(tmp_file.name, audio_data, sr)
                result = self.model.transcribe(tmp_file.name, fp16=False)
                return result.get("text", "").strip()
        except Exception as e:
            print(f"[ERROR] Whisper transcription error: {e}")
            return "[ASR Error]"
