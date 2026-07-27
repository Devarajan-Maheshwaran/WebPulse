# Software Requirements Specification (SRS)
## Project: WebPulse — Webcam-Based Remote-PPG Arousal Estimation Fused with Voice Valence for LLM-Driven Emotional Response Generation

Version: 1.0
Date: 2026-07-27
Author: Devarajan Maheshwaran

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for **WebPulse**, a software system that estimates a user's emotional arousal from a webcam-derived remote photoplethysmography (rPPG) signal, fuses it with voice-tone valence extracted from microphone audio, and drives a Large Language Model (LLM) to generate an emotionally-appropriate conversational response. The system is designed as a low-cost, contact-free complement to physiological-sensor-based emotion-aware robot/agent pipelines, specifically targeting deployment contexts (e.g., elderly care) where wearable biosensor compliance is a known barrier.

### 1.2 Scope
WebPulse is a **proof-of-concept research prototype**, not a medical device. It:
- Captures live video from a standard laptop/webcam and live audio from a standard microphone.
- Extracts a pulse-like waveform from facial skin color variation (remote PPG).
- Derives a Heart-Rate-Variability-based arousal proxy from the pulse waveform.
- Extracts pitch/energy-based valence features from speech.
- Fuses arousal and valence into a 4-quadrant emotion label using Russell's Circumplex Model of Affect.
- Transcribes speech and sends the emotion label + transcript to an LLM API to generate an empathetic response.
- Outputs the response as text and/or synthesized speech.
- Logs every session (raw signals, derived scores, labels, LLM outputs) for later analysis and demonstration.

Out of scope: clinical-grade accuracy, FDA/medical certification, long-term deployment, large-scale dataset collection.

### 1.3 Intended Audience
- The author (developer/researcher).
- Academic reviewers / prospective research supervisors evaluating the proof-of-concept.
- Future collaborators extending the project.

### 1.4 Definitions and Acronyms
- **rPPG** — Remote Photoplethysmography: extraction of a pulse-related signal from video of skin, without contact.
- **HRV** — Heart Rate Variability: variation in time between successive heartbeats, used as an autonomic arousal indicator.
- **RMSSD** — Root Mean Square of Successive Differences, a standard HRV metric.
- **Valence** — The positive/negative quality of an emotion (pleasant vs. unpleasant).
- **Arousal** — The intensity/activation level of an emotion (calm vs. excited).
- **Circumplex Model of Affect (Russell, 1980)** — A 2D model mapping emotions onto Valence (x-axis) and Arousal (y-axis) axes.
- **LLM** — Large Language Model (e.g., GPT, Claude).
- **ROI** — Region of Interest (facial skin area used for signal extraction).

### 1.5 References (existing research and implementations this project builds on)

**Foundational / most recent papers (2025–2026):**
1. CAST-Phys: *Contactless Affective States Through Physiological Signals Database*, arXiv, 2025. Establishes the current state of the art for combining facial-video-derived physiological signals (rPPG, EDA, respiration) with affect recognition; used here as the primary justification and terminology reference for the contactless approach.
2. *Dimensional emotion recognition from camera-based PRV*, PubMed, 2023. Validates that camera-derived Pulse Rate Variability correlates with HRV-based arousal/valence, supporting the technical feasibility of the arousal-extraction step.
3. *A Novel HMD-Mounted Contactless Proxi-rPPG Sensor for Emotion Recognition*, ETRI. Provides a benchmark accuracy figure (~65% on 4-quadrant classification) for realistic performance expectations.
4. *Noninvasive Patient Monitoring with Ambient Sensors... for Individuals Living with Alzheimer's Disease*, ASME, 2024. Provides the compliance/usability justification: wearables and cameras both face adoption barriers in elderly care, motivating low-friction sensing alternatives.
5. *Exploring Contactless Techniques in Multimodal Emotion Recognition*, Oulu repository, 2024. Survey of contactless modalities used as background/related-work material.
6. EAI Endorsed Transactions review on contactless/sensor-based emotion recognition in older users, 2025. Notes the lack of empirical validation of contactless emotion recognition specifically in elderly populations — this is the specific gap WebPulse's testing addresses at small scale.
7. *Exploring the Feasibility of Wearable Sensors for Emotion Recognition in Older Adults*, PMC, Dec 2025. Found weak correlation between facial expression and wearable-sensor-derived emotion in older adults, motivating multimodal fusion (pulse + voice) rather than single-channel sensing.
8. Emotion Detection in Older Adults Using Physiological Signals from Wearables (Empatica E4, Shimmer3 GSR+), arXiv, July 2025. Represents the "baseline" wearable-based approach WebPulse is proposed as a complement to.

**Open-source rPPG implementations used as building blocks (not reimplemented from scratch):**
9. `pyVHR` (phuselab) — Python framework for remote heart-rate/pulse extraction with multiple established algorithms (GREEN, CHROM, POS).
10. `rPPG-Toolbox` (ubicomplab) — Deep-learning-based remote PPG toolbox, NeurIPS 2023, usable as an optional higher-accuracy backend.
11. `webcam-pulse-detector` (thearn) — Minimal, well-documented real-time webcam pulse detector using forehead ROI + FFT.
12. `yarppg` (Sam Proell, blog + code) — MediaPipe Face Mesh-based facial ROI tracking with green-channel pulse extraction, actively documented in 2022–2023.
13. `ppg` (quinnzipse) — Simple webcam-based PPG heartbeat detector, minimal dependencies.
14. Medium walkthrough, "Contactless Stress Sensing with Just a Camera" (Nov 2025) — Step-by-step replication guide: face detection → ROI → detrend → Butterworth bandpass filter (0.7–3.0 Hz) → FFT peak → heart rate. Used as the primary implementation blueprint.
15. `vitallens-python` (Rouast Labs) — Actively maintained (2026) rPPG library supporting G/POS/CHROM methods and HR + respiratory rate extraction; optional drop-in alternative signal-extraction backend.

**Speech/valence and LLM components (existing tools, not novel):**
16. `librosa` — Python audio analysis library used for pitch (F0) and energy (RMS) feature extraction for valence estimation.
17. OpenAI Whisper (or equivalent open-source ASR) — used for speech-to-text transcription.
18. Any hosted LLM API (e.g., Anthropic Claude, OpenAI GPT) — used for empathetic response generation.
19. Any TTS engine (e.g., system TTS, ElevenLabs, pyttsx3) — used for optional spoken output.

---

## 2. Overall Description

### 2.1 Product Perspective
WebPulse is a standalone desktop application (Python-based) that runs locally on a laptop with a webcam and microphone. It is a research prototype intended to demonstrate feasibility, not a production system. It is architecturally analogous to physiological-signal-to-LLM pipelines used in emotion-aware companion robots, but substitutes contact sensors with webcam/microphone-based sensing.

### 2.2 Product Functions (Summary)
1. Capture live video and detect/track a facial ROI.
2. Extract a pulse waveform from the ROI using an established rPPG method.
3. Filter and process the waveform to estimate heart rate and an HRV-based arousal score.
4. Capture live audio, transcribe speech, and extract pitch/energy features for valence.
5. Fuse arousal + valence into one of four emotion quadrants.
6. Construct a prompt containing the emotion label and transcript, send it to an LLM, and receive an empathetic response.
7. Present the response as on-screen text and optionally as synthesized speech.
8. Log all raw and derived data per session for later review and demonstration.

### 2.3 User Characteristics
- Primary user during testing: the developer and a small group of friends/volunteers (5–8 people) acting as test subjects.
- No specialized technical knowledge required to be a test subject; only to run the software.

### 2.4 Constraints
- Standard laptop webcam and microphone (no specialized hardware).
- Requires reasonably stable lighting for reliable rPPG signal extraction.
- Requires internet access for LLM API calls (unless a local LLM is substituted).
- Not validated for clinical or diagnostic use.
- Accuracy is expected to be moderate (consistent with ~60–75% reported in related contactless-emotion literature), not high-precision.

### 2.5 Assumptions and Dependencies
- Test subjects consent to being recorded (video + audio) for short sessions.
- Python 3.x environment with OpenCV, MediaPipe (or dlib), NumPy, SciPy, librosa, and an LLM API client library.
- A quiet, adequately lit room for testing.

---

## 3. Specific Requirements

### 3.1 Functional Requirements

**FR-1: Video Capture and Face/ROI Detection**
- The system shall capture live video frames from the default webcam at a minimum of 20 FPS.
- The system shall detect the face in each frame and define a stable ROI (e.g., forehead or cheek region) using a face-landmark detector (MediaPipe Face Mesh or Haar cascade + fixed offset).

**FR-2: Pulse Signal Extraction (rPPG)**
- The system shall extract the mean pixel intensity of the green channel (or a POS/CHROM-derived signal) from the ROI for each frame, producing a raw time-series signal.
- The system shall detrend the raw signal and apply a Butterworth bandpass filter in the 0.7–3.0 Hz range (approx. 42–180 BPM) to isolate the pulse component.

**FR-3: Heart Rate and HRV/Arousal Estimation**
- The system shall estimate instantaneous heart rate via FFT peak detection or peak-to-peak interval analysis on the filtered signal, over a rolling window (e.g., 10–15 seconds).
- The system shall compute an HRV proxy (e.g., RMSSD-style variability of peak-to-peak intervals) over a rolling window (e.g., 30–60 seconds).
- The system shall map HRV to an arousal/stress score: lower HRV variability → higher arousal/physiological stress; higher HRV variability → lower arousal, using thresholds calibrated during pilot testing. This framing explicitly mirrors the stress/arousal estimation approach used in Sugaya's lab's work comparing emotion estimation from EEG and heart rate variability indices with machine learning. [shibaura.elsevierpure](https://shibaura.elsevierpure.com/en/publications/feature-comparison-of-emotion-estimation-by-eeg-and-heart-rate-va)

**FR-4: Audio Capture and Valence Estimation**
- The system shall capture live audio from the microphone during each session.
- The system shall extract pitch (F0) and energy (RMS) features using librosa over short speech segments.
- The system shall map pitch/energy patterns to a valence score (positive/negative), using simple heuristics or a lightweight pretrained model.

**FR-5: Emotion Fusion**
- The system shall combine the arousal score and valence score into one of four emotion-quadrant labels (e.g., calm-positive, calm-negative, aroused-positive, aroused-negative), based on Russell's Circumplex Model.

**FR-6: Speech Transcription**
- The system shall transcribe captured speech to text using an ASR engine (e.g., Whisper).

**FR-7: LLM-Driven Response Generation**
- The system shall construct a prompt containing the current emotion label and the transcribed text.
- The system shall send this prompt to an LLM API and receive a generated empathetic response.

**FR-8: Response Output**
- The system shall display the LLM-generated response as on-screen text.
- The system shall optionally convert the response to speech using a TTS engine.

**FR-9: Session Logging**
- The system shall log, per session and per time window: raw pulse waveform data (or summary), heart rate trace, HRV/arousal score, valence score, emotion label, transcript, and LLM response, each with a timestamp.
- The system shall support exporting session logs (e.g., to CSV/JSON) for later analysis.

**FR-10: Test Session Support**
- The system shall support running discrete "sessions" (start/stop) so that each test subject's interaction can be logged and reviewed separately.

### 3.2 Non-Functional Requirements

**NFR-1: Usability**
- The system shall require no specialized hardware beyond a standard laptop webcam and microphone.
- The system shall provide a simple start/stop control for sessions.

**NFR-2: Performance**
- The system shall process video and update the displayed heart rate/arousal estimate with a maximum latency of 2–3 seconds after the rolling window fills.
- The system shall generate an LLM response within a few seconds of speech input, dependent on API latency.

**NFR-3: Reliability**
- The system shall handle temporary loss of face detection (e.g., user looks away) gracefully, without crashing, by pausing signal accumulation until the face is redetected.

**NFR-4: Maintainability**
- The system shall be modular: separate components for video capture/rPPG, audio/valence, fusion logic, LLM interfacing, and logging, so that any component (e.g., the rPPG backend) can be swapped for a more advanced implementation (e.g., pyVHR or rPPG-Toolbox) without rewriting the rest of the pipeline.

**NFR-5: Ethical/Privacy Considerations**
- The system shall only record test subjects with informed consent.
- The system shall store session data locally by default and shall not transmit raw video/audio to any third party beyond the necessary LLM/ASR API calls (text/transcript only, not raw video).
- The system shall allow deletion of session recordings on request.

**NFR-6: Portability**
- The system shall run on standard consumer laptops (Windows/macOS/Linux) without specialized hardware.

---

## 4. System Architecture (High-Level)

```
Webcam Video ---> Face/ROI Detection ---> Green/POS Signal Extraction ---> Bandpass Filter
                                                                              |
                                                                              v
                                                          Heart Rate + HRV/Arousal Estimation
                                                                              |
Microphone Audio ---> ASR (transcript) ---> Pitch/Energy Extraction ---> Valence Estimation
                                                                              |
                                                                              v
                                                        Emotion Fusion (Arousal, Valence) -> Label
                                                                              |
                                                                              v
                                                 Prompt Construction (Label + Transcript) -> LLM API
                                                                              |
                                                                              v
                                                     Text Response  ->  (optional) TTS Output
                                                                              |
                                                                              v
                                                         Session Logger (CSV/JSON, timestamped)
```

---

## 5. Build Plan (Phased, Using Existing Implementations)

**Phase 0 — Environment setup**
- Install Python, OpenCV, MediaPipe, NumPy, SciPy, librosa, an ASR library, an LLM API client, and a TTS library.

**Phase 1 — rPPG signal pipeline (reuse existing implementation)**
- Start from a minimal open-source implementation (e.g., `webcam-pulse-detector` or the Medium 2025 walkthrough approach: Haar cascade/MediaPipe ROI -> green channel -> detrend -> Butterworth bandpass -> FFT).
- Validate: confirm the estimated heart rate is plausible (compare briefly against a phone HR app or manual pulse count).

**Phase 2 — HRV/arousal scoring**
- Implement peak-to-peak interval extraction from the filtered waveform.
- Compute a rolling RMSSD-style variability metric.
- Define and calibrate simple thresholds for low/high arousal based on pilot recordings.

**Phase 3 — Voice valence pipeline**
- Use librosa to extract pitch and energy from short audio segments.
- Define simple heuristics (e.g., higher pitch variability + higher energy -> more positive/aroused; flat pitch + low energy -> more negative/calm) or a small pretrained valence classifier if available.

**Phase 4 — Fusion and emotion labeling**
- Combine arousal score and valence score into one of four quadrant labels per Russell's Circumplex Model.

**Phase 5 — Transcription and LLM response**
- Integrate an ASR engine for transcription.
- Construct the LLM prompt using the emotion label and transcript.
- Call the LLM API and display/return the response; optionally synthesize speech via TTS.

**Phase 6 — Logging and session management**
- Implement start/stop session controls.
- Log all raw and derived values with timestamps to CSV/JSON per session.

**Phase 7 — Pilot testing on friends**
- Run 5–8 short sessions with different volunteers, under informed consent.
- Record short screen captures showing the live waveform, scores, label, and LLM response.
- Compile a simple summary table: subject, session duration, average HR, arousal score range, valence score range, final emotion label(s), and sample LLM responses.

**Phase 8 — Write-up**
- Summarize methodology, cite the reference papers listed above, present the session logs/table, and discuss limitations and future work (e.g., integrating a real wearable ground-truth comparison, expanding to more subjects, refining fusion rules).

---

## 6. Limitations and Future Work

- Accuracy of webcam-based HR/HRV extraction is sensitive to lighting, motion, and skin tone; this is a known limitation across all rPPG literature, not unique to this implementation.
- The arousal/valence-to-emotion mapping is heuristic at this stage rather than learned from a large labeled dataset; refining it with more pilot data or a small classifier is future work.
- No ground-truth wearable comparison is included in the initial prototype; adding one (e.g., a consumer smartwatch's HR/HRV reading run in parallel) would strengthen validation and is suggested as an immediate next step.
- The current scope is a proof-of-concept with a small number of test subjects, not a validated clinical or elderly-specific study; a natural extension is a small structured study with older adult volunteers, following the gap identified in the EAI Endorsed Transactions review and the PMC wearable-feasibility paper referenced above.
- Explore replacing the classical green-channel/POS rPPG stage with a CNN-based or state-space-model-based rPPG backend (e.g., PhysNet or rPPG-Toolbox) as a future upgrade, to align with current deep-learning-based rPPG literature, while keeping the initial proof-of-concept focused on simpler, verifiable methods. [github](https://github.com/ubicomplab/rPPG-Toolbox)

---

## 7. Acceptance Criteria for the Proof-of-Concept

The project will be considered a successful proof-of-concept if:
1. The system reliably extracts a plausible heart-rate estimate from webcam video across at least 5 different test subjects.
2. The system produces an arousal score that visibly changes between calm and speaking/active conditions in pilot sessions.
3. The system produces a valence score that visibly differs between neutral and emotionally expressive speech in pilot sessions.
4. The system generates coherent, context-appropriate LLM responses that vary based on the fused emotion label.
5. All sessions are fully logged with raw/derived data and are available for review (CSV/JSON + optional screen recordings).
