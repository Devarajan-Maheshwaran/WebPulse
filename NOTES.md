# WebPulse — Development & Execution Notes

> [!IMPORTANT]
> **SIMULATED / NO REAL SENSOR ACCESS IN THIS ENVIRONMENT**
>
> All automated self-tests and test runs executed within this development environment used **SYNTHETIC / SIMULATED SENSOR DATA** (`--simulated` mode) because physical camera and microphone hardware devices are not attached to this headless execution runner.
>
> **Do NOT mistake any generated test logs or console outputs for real empirical pilot session data.**

---

## Environment & Testing Summary

1. **Hardware Dependencies**:
   - `rppg/capture.py` (webcam video)
   - `audio/capture.py` (microphone audio)
   - Requires real physical hardware testing by the user locally.

2. **Synthetic Verification**:
   - The signal processing math (detrending, 3rd order Butterworth bandpass 0.7–3.0 Hz, FFT heart rate estimation, RMSSD HRV calculation), acoustic pitch/energy feature extraction, Russell circumplex emotion fusion, and session logger export were verified using synthetic math signals.

3. **API Keys & Offline Testing**:
   - `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are left blank in `.env.example` / `.env`.
   - The system includes a mock response fallback in `llm/prompt.py` (`_get_mock_response()`) so the pipeline runs offline cleanly until real keys are provided by the user.

---

## Instructions for Local Testing on Real Hardware

1. **Run Live Multimodal Pipeline**:
   ```bash
   conda run -n webpulse python app.py --subject volunteer_01
   ```

2. **Run Individual Standalone Test Scripts**:
   - rPPG Camera Test: `conda run -n webpulse python test_rppg.py`
   - Audio Microphone Test: `conda run -n webpulse python test_audio.py`
   - Emotion Fusion Unit Tests: `conda run -n webpulse python test_fusion.py`
   - LLM & TTS Test: `conda run -n webpulse python test_llm.py`
   - Session Logger Test: `conda run -n webpulse python test_logging.py`
