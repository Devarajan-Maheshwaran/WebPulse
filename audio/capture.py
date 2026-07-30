"""
audio/capture.py — Microphone audio capture module.

Implements FR-4 (Audio Capture):
Captures live audio from default microphone using sounddevice.
Buffers audio samples into rolling speech segments for pitch/energy analysis.
"""

import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class AudioCapturer:
    """
    Handles live microphone audio recording using sounddevice.
    """

    def __init__(self, sample_rate=16000, channels=1, buffer_duration_sec=4.0, on_chunk=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = int(sample_rate * buffer_duration_sec)
        self.audio_buffer = np.zeros((0, channels), dtype=np.float32)
        self.stream = None
        self.is_recording = False
        self.on_chunk = on_chunk

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback function for sounddevice InputStream."""
        if status:
            pass  # Suppress overflow warnings in log
        self.audio_buffer = np.vstack((self.audio_buffer, indata))
        # Keep buffer to maximum capacity
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
        if self.on_chunk is not None:
            try:
                self.on_chunk(indata.copy())
            except Exception:
                pass

    def start(self):
        """Start listening to microphone input."""
        if not HAS_SOUNDDEVICE:
            print("[WARNING] sounddevice library not available.")
            return False

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_recording = True
            return True
        except Exception as e:
            print(f"[UNTESTED ON HARDWARE] Microphone access failed: {e}")
            self.is_recording = False
            return False

    def stop(self):
        """Stop listening to microphone input."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False

    def get_audio_segment(self):
        """
        Get current buffered audio segment.
        
        Returns:
            audio_data (ndarray 1D float32): Flattened audio waveform.
            sr (int): Sampling rate.
        """
        if len(self.audio_buffer) == 0:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        return self.audio_buffer.flatten(), self.sample_rate
