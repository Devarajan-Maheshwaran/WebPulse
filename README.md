# WebPulse

**Webcam-based remote-PPG arousal estimation fused with voice valence, driving an LLM's emotionally-appropriate response — a contact-free proof-of-concept complement to wearable-sensor-based emotion-aware companion robot pipelines.**

## What this is

WebPulse is a research prototype that:
1. Estimates a pulse-like signal from a normal laptop webcam (remote photoplethysmography / rPPG) — no wearable, no skin contact.
2. Derives an HRV-based arousal score from that signal.
3. Extracts voice-tone valence (pitch/energy) from microphone audio.
4. Fuses arousal + valence into an emotion label (Russell's Circumplex Model: calm/aroused x positive/negative).
5. Feeds the emotion label + live speech transcript into an LLM to generate an empathetic response.
6. Logs every session (raw signals, HR/HRV, valence, emotion label, transcript, LLM output) to CSV/JSON for review and demonstration.

It does **not** reimplement rPPG signal extraction from scratch — it builds directly on established, open-source rPPG techniques and applies them to a new pipeline (voice fusion + LLM response generation) and a new framing (low-compliance alternative to wearable sensors, especially relevant for elderly users).

## Why

Physiological-signal-based emotion recognition (PPG/EEG wearables driving robot/LLM responses) is a well-established and effective approach, but a known real-world limitation is that many users — especially elderly users — resist wearing biosensors daily due to comfort, cost, or stigma (Noninvasive Patient Monitoring with Ambient Sensors, ASME 2024). Recent work (CAST-Phys, arXiv 2025) shows facial-video-derived physiological signals can substitute for contact sensors in affect recognition. WebPulse is a small, honest exploration of whether this contactless approach can drive the same kind of emotion-aware response pipeline, tested live on real people.

## What's novel here (and what isn't)

**Not novel:** the rPPG signal extraction technique itself (face ROI -> color-channel signal -> bandpass filter -> FFT/peak detection). This is a mature, well-documented method with multiple open-source implementations.

**Novel / the actual contribution:**
- Fusing webcam-derived HRV-based arousal/stress classification (explicitly inspired by Sugaya's lab's EEG+HRV emotion estimation work) with voice-derived valence in one live pipeline. [shibaura.elsevierpure](https://shibaura.elsevierpure.com/en/publications/feature-comparison-of-emotion-estimation-by-eeg-and-heart-rate-va)
- Driving LLM-based empathetic response generation from that fused, contactless emotion estimate.
- Framing and testing it explicitly as a low-compliance complement to wearable-based elderly-care emotion sensing.
- Real pilot sessions with logged data, not just a signal-processing demo.

## How it works (architecture)

```
Webcam --> Face/ROI detection --> Green/POS channel signal --> Bandpass filter (0.7-3.0 Hz)
                                                                        |
                                                                        v
                                                     Heart rate + HRV --> Arousal score
                                                                        |
Microphone --> ASR transcript                                          |
            --> Pitch/energy features --> Valence score                |
                                                                        v
                                          Fusion (Arousal, Valence) --> Emotion label
                                                                        |
                                                                        v
                                 Prompt (label + transcript) --> LLM API --> Response
                                                                        |
                                                                        v
                                                    Text / TTS output + session log
```

---

## Installation & Setup Guide

### 1. Environment Prerequisites
- Python 3.11 (Conda environment recommended)
- OpenCV, MediaPipe, SciPy, NumPy, Librosa, SoundDevice, OpenAI-Whisper, PyTTSSx3

### 2. Create Environment & Install Dependencies
```bash
# Create Conda environment
conda create -n webpulse python=3.11 -y
conda activate webpulse

# Install required dependencies
pip install -r requirements.txt
```

### 3. API Key Configuration
Copy `.env.example` to `.env` and fill in your API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-actual-api-key-here
LLM_MODEL=gpt-4o-mini
```
*Note: If no API key is provided, WebPulse automatically falls back to an offline mock response generator for local testing.*

---

## How to Run

### 1. Run Live Integrated Multimodal App
```bash
conda run -n webpulse python app.py --subject volunteer_01
```
*Press `q` in the camera display window to quit and export session logs.*

### 2. Run Self-Test / Simulated Mode (No Hardware Required)
```bash
conda run -n webpulse python app.py --simulated
```

### 3. Run Standalone Module Test Scripts
- **rPPG Pipeline Test**: `conda run -n webpulse python test_rppg.py`
- **Voice Valence Test**: `conda run -n webpulse python test_audio.py`
- **Emotion Fusion Unit Tests**: `conda run -n webpulse python test_fusion.py`
- **LLM & Transcription Test**: `conda run -n webpulse python test_llm.py`
- **Session Logger Test**: `conda run -n webpulse python test_logging.py`

---

## Session Logs

All pilot sessions export timestamped log files into the `sessions/` directory:
- **`sessions/session_<timestamp>_<subject_id>.json`**: Complete structured record containing metadata, continuous HR/HRV metrics, arousal, valence, emotion quadrant, speech transcript, and LLM text response.
- **`sessions/session_<timestamp>_<subject_id>.csv`**: Tabular CSV export suitable for data analysis and visualization.

---

## Repo Structure

```
WebPulse/
  rppg/             # Face mesh landmark detection, green ROI signal, bandpass filtering, HR & RMSSD HRV
  audio/            # Microphone stream, librosa pitch/energy feature extraction, valence heuristic
  fusion/           # Russell's Circumplex Model (arousal, valence) -> 4-quadrant emotion label
  llm/              # Whisper speech-to-text, prompt builder, OpenAI/Anthropic client, pyttsx3 TTS engine
  logging_/         # Session lifecycle logger (JSON and CSV file export)
  sessions/         # Output directory for exported session log files
  app.py            # Main entry point for live & simulated multimodal application
  test_rppg.py      # Standalone rPPG test script
  test_audio.py     # Standalone audio valence test script
  test_fusion.py    # Unit tests for emotion fusion
  test_llm.py       # Standalone LLM prompt & TTS test script
  test_logging.py   # Standalone session logger test script
  SRS_WebPulse.md   # Full Software Requirements Specification
  README_WebPulse.md
  README.md
```

## Built on top of (references and building blocks)

**Research grounding (2025-2026):**
- CAST-Phys: Contactless Affective States Through Physiological Signals Database, arXiv, 2025
- Dimensional emotion recognition from camera-based PRV, 2023
- A Novel HMD-Mounted Contactless Proxi-rPPG Sensor for Emotion Recognition, ETRI
- Noninvasive Patient Monitoring with Ambient Sensors for Alzheimer's Disease, ASME, 2024
- Exploring Contactless Techniques in Multimodal Emotion Recognition (survey), 2024
- Contactless/sensor-based emotion recognition in older users (review), EAI Endorsed Transactions, 2025
- Exploring the Feasibility of Wearable Sensors for Emotion Recognition in Older Adults, PMC, Dec 2025
- Emotion Detection in Older Adults Using Physiological Signals from Wearables, arXiv, July 2025

**Open-source implementations used/adapted:**
- pyVHR (phuselab) — Python rPPG framework (GREEN/CHROM/POS methods)
- rPPG-Toolbox (ubicomplab) — Deep-learning-based rPPG toolbox, NeurIPS 2023
- webcam-pulse-detector (thearn) — minimal real-time webcam pulse detector
- yarppg (Sam Proell) — MediaPipe Face Mesh + green-channel pulse extraction
- ppg (quinnzipse) — simple webcam PPG heartbeat detector
- vitallens-python (Rouast Labs) — actively maintained rPPG library (2026)
- "Contactless Stress Sensing with Just a Camera" (Medium, Nov 2025) — step-by-step implementation blueprint used as primary reference

**Other components:**
- librosa — pitch/energy extraction for valence
- Whisper (OpenAI) — speech-to-text transcription
- LLM API (Claude/GPT) — empathetic response generation
- pyttsx3 — TTS engine for spoken output

See `SRS_WebPulse.md` for the full requirements specification, detailed references, and phased build plan.

## Status

Proof-of-concept / research prototype. Not a medical device. Tested on a small number of volunteer sessions with informed consent; not validated for clinical or diagnostic use.

## Limitations

- Sensitive to lighting, motion, and skin tone (a known limitation across all rPPG literature).
- Arousal/valence-to-emotion mapping is currently heuristic, not learned from a large dataset.
- No wearable ground-truth comparison yet; adding one is a natural next step.
- Small pilot size; not a controlled study.

## Future work

- Add a wearable (e.g., smartwatch HR/HRV) as a parallel ground-truth channel for validation.
- Expand pilot testing, especially with older adult volunteers, to directly address the elderly-specific validation gap noted in recent reviews.
- Explore swapping the rPPG backend for a more advanced method (pyVHR/rPPG-Toolbox) for higher accuracy.
- Refine the arousal/valence fusion rules using pilot data.
- Experiment with CNN/transformer/SSM-based rPPG backends (e.g., PhysNet via rPPG-Toolbox) as a more advanced replacement for the classical rPPG stage, once the basic pipeline is stable, to show awareness of state-of-the-art methods without overcomplicating the initial proof-of-concept. [github](https://github.com/ubicomplab/rPPG-Toolbox)
