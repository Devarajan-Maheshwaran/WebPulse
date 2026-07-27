"""
llm/transcribe.py — Speech-to-text transcription module using OpenAI Whisper.

Implements FR-6: Speech Transcription.
Automatically ensures ffmpeg binary path from imageio-ffmpeg is registered in PATH,
and aliases versioned ffmpeg executable to ffmpeg.exe for Windows compatibility.
"""

import os
import shutil
import tempfile
import numpy as np

# Automatically register ffmpeg executable path and create ffmpeg.exe alias if needed
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_bin_dir = os.path.dirname(ffmpeg_exe)
    target_ffmpeg = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")
    if not os.path.exists(target_ffmpeg) and os.path.exists(ffmpeg_exe):
        try:
            shutil.copyfile(ffmpeg_exe, target_ffmpeg)
        except Exception:
            pass
    if ffmpeg_bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_bin_dir + os.path.pathsep + os.environ.get("PATH", "")
except Exception as e:
    pass

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
            return ""

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            sf.write(tmp_path, audio_data, sr)
            result = self.model.transcribe(tmp_path, fp16=False)
            return result.get("text", "").strip()
        except Exception as e:
            print(f"[ERROR] Whisper transcription error: {e}")
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
