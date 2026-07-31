# WebPulse

WebPulse is a local, real-time multimodal affective-computing research prototype. It estimates cardiac activity from a webcam using EfficientPhys-based remote photoplethysmography (rPPG), derives heart-rate variability (HRV/PRV) features, classifies physiological stress with a WESAD-trained RandomForest, estimates voice valence, and provides a spoken Gemini Live companion.

The system is intended for HCI research and interaction prototyping. Its camera and audio estimates are uncertain context signals, not medical measurements, diagnosis, or clinical monitoring.

## Research Positioning

The design is informed by:

- Liu et al., *EfficientPhys* and the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox): efficient deep remote-PPG inference and evaluation patterns.
- Schmidt et al., [WESAD](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html): wearable stress and affect labels used to train the local HRV classifier.
- Russell, [A Circumplex Model of Affect](https://doi.org/10.1037/h0077714): arousal–valence representation used by fusion.
- Ikeda, Horie, and Sugaya, [Estimating Emotion with Biological Information for Robot Interaction](https://doi.org/10.1016/j.procs.2017.08.198): conceptual motivation for mapping biological indices into interaction-facing affect.
- Ziaratnia, Laohakangvalvit, Sugaya, and Sripian, [Multimodal Deep Learning for Remote Stress Estimation Using CCT-LSTM](https://openaccess.thecvf.com/content/WACV2024/html/Ziaratnia_Multimodal_Deep_Learning_for_Remote_Stress_Estimation_Using_CCT-LSTM_WACV_2024_paper.html): motivation for explicit modality branches and temporal stabilization.

WebPulse does not claim to reproduce CCT-LSTM, train on UBFC-Phys, or achieve the reported paper results. It is a lightweight engineering implementation using EfficientPhys, WESAD-derived HRV classification, voice features, and temporal fusion.

## System Architecture

The runtime is split into three local processes so camera inference, IPC, and network/audio I/O do not block one another:

```text
Process 1: Camera + rPPG + WESAD
  Webcam -> MediaPipe ROIs -> EfficientPhys x3 -> median BVP
          -> HR + RMSSD/PRV -> WESAD stress/arousal -> localhost:5003

Process 3: State broker
  localhost:5003 -> latest-only in-memory physiological state

Process 2: Gemini Live companion
  Microphone -> PCM WebSocket input
  Broker state -> labeled 0–5 context -> Gemini Live
  Gemini native audio -> low-latency speaker queue
```

### Process 1: camera and physiological pipeline

Entry point: [rppg/rppg_server.py](rppg/rppg_server.py)

1. OpenCV captures the webcam stream.
2. MediaPipe FaceLandmarker detects the face and extracts three named regions:
   - `forehead`
   - `left_cheek`
   - `right_cheek`
3. Each region is checked for usable pixels, occlusion, brightness, and temporal validity.
4. The capture layer passes raw ROI crops to the model layer. Enhancement is applied exactly once immediately before EfficientPhys normalization; this avoids double CLAHE/gamma processing.
5. EfficientPhys is run separately on each ROI's synchronized temporal window. The default window is 10 frames plus the preceding frame required by the exported model's temporal-difference operation.
6. A window is accepted only when all three ROI queues are valid and ready. If one region is occluded, its queue is cleared so stale data cannot be paired with fresh data from another region.
7. The three EfficientPhys BVP outputs are aligned and fused with a per-sample median. Median fusion reduces the effect of a transient glasses reflection, highlight, hair occlusion, or landmark error while retaining forehead and both cheeks.
8. HR is estimated from the fused BVP using FFT within the configured physiological band. Pulse peaks from the same fused BVP produce RMSSD.
9. The same fused HR/HRV result is passed to the WESAD classifier. WESAD is not run independently on only the forehead or on a separate single-cheek stream.
10. A lightweight three-level temporal stabilizer rejects isolated WESAD state changes and accepts sustained changes. The resulting state is `CALM`, `NORMAL`, or `STRESSED`.

Implementation: [rppg/capture.py](rppg/capture.py), [rppg/deep_engine.py](rppg/deep_engine.py), [rppg/hrv.py](rppg/hrv.py).

### EfficientPhys inference

[rppg/deep_engine.py](rppg/deep_engine.py) loads `weights/efficientphys.onnx` with ONNX Runtime. `DmlExecutionProvider` is preferred on Windows, with `CPUExecutionProvider` as fallback. Each raw BGR crop is converted to RGB, resized to 72×72, scaled to `[0, 1]`, and transposed to channel-first format before inference.

The model is a single-ROI temporal model, so “three-region EfficientPhys” means three synchronized model evaluations followed by robust signal fusion—not an unsupported change to the ONNX input shape. This preserves compatibility with the exported model.

The server startup output must identify the active backend and provider. A healthy deep path should report:

```text
EfficientPhys execution: GPU (DirectML)
```

CPU fallback is supported but may reduce frame rate.

### HR, HRV, and WESAD

[rppg/hrv.py](rppg/hrv.py) computes:

- heart rate in BPM from the fused BVP spectrum;
- pulse peaks using physiological distance and prominence constraints;
- RMSSD from successive valid inter-beat intervals;
- arousal from bounded HRV mapping when required.

[fusion/wesad_classifier.py](fusion/wesad_classifier.py) uses the existing trained artifact at `fusion/models/wesad_hrv_classifier.pkl`. Its feature vector is:

```text
[RMSSD in ms, estimated SDNN in ms, mean HR in BPM]
```

The classifier returns a three-level stress label, arousal score, confidence, and source. The current artifact is not retrained during live execution. The same fused physiological window drives both HRV-derived metrics and WESAD, ensuring the model does not receive a different ROI subset than the HR estimator.

### Multimodal fusion

[fusion/emotion.py](fusion/emotion.py) combines:

- body modality: fused rPPG HR/HRV, WESAD stress, and physiological arousal;
- audio modality: voice valence derived from microphone features and speech content;
- dimensional state: one of `calm-positive`, `calm-negative`, `aroused-positive`, or `aroused-negative`.

The temporal stabilizer is intentionally small and CPU-friendly. It is a smoothing and state-consistency mechanism inspired by the temporal role of recurrent stages in multimodal affect models; it is not a CCT-LSTM replacement.

### Process 3: state broker

[state_broker.py](state_broker.py) is a localhost TCP broker on `127.0.0.1:5003`. It stores the latest state in memory and forwards updates to subscribers. No state files or disk polling are used for live IPC. Process 1 publishes approximately once per second; Process 2 consumes the latest available state without waiting for camera inference.

### Process 2: Gemini Live audio companion

Entry point: [live_emotion_agent.py](live_emotion_agent.py), launched through [app.py](app.py).

- Loads `GEMINI_API_KEY` from `.env`.
- Maintains one persistent asynchronous WebSocket session.
- Streams 16 kHz, mono, signed 16-bit PCM microphone chunks through `realtimeInput`.
- Sends physiological context through `clientContent` with `turnComplete: false`, so background state does not independently trigger speech.
- Pushes context at a two-second background cadence and again at speech start/end so the user's current turn is paired with recent state.
- Requests native Gemini audio and writes received 24 kHz PCM frames to a low-latency speaker queue.
- Clears the speaker queue immediately when `serverContent.interrupted` is received.
- Separates `inputTranscription` into the `VOICE` HUD panel and `outputTranscription` into the `COMPANION` panel.

Gemini receives an abstracted labeled context rather than raw physiological values. The payload includes stress, arousal, valence, heart-rate category, HRV stress-load category, signal quality, and bounded 0–5 scores. Raw BPM and RMSSD remain inside the local process for display and computation.

## Running the Project

Use the project’s virtual environment and install [requirements.txt](requirements.txt):

```powershell
pip install -r requirements.txt
```

Create `.env` with at least:

```text
GEMINI_API_KEY=your_key_here
```

Start three PowerShell terminals from the project root, in this order:

```powershell
# Terminal 1: state broker
python state_broker.py
```

```powershell
# Terminal 2: camera/rPPG/WESAD process
python -m rppg.rppg_server
```

```powershell
# Terminal 3: Gemini Live/audio process
python app.py
```

Press `q` in the camera window to stop the vision process. The camera process should report the active ROI/backend state and periodically print HR, arousal, stress, signal quality, and ROI coverage.

Useful environment settings include:

```text
RPPG_BACKEND=deep
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
GEMINI_LIVE_VOICE=Puck
```

Set `RPPG_BACKEND=classical` only for diagnostics; the classical fallback also requires forehead, left-cheek, and right-cheek samples before calculating its fused signal.

## Verification

Static and component checks:

```powershell
python -m py_compile rppg\capture.py rppg\deep_engine.py rppg\rppg_server.py fusion\emotion.py live_emotion_agent.py
python -m pytest -q test_fusion.py
```

Runtime verification should confirm:

- all three ROI names are present and usable during accepted windows;
- `roi_count` is `3` in state messages;
- the backend is `efficientphys_onnx`;
- DirectML is active when available;
- `quality_status` is `GOOD` before interpreting HRV/stress changes;
- WESAD receives the fused HRV result and reports its source/confidence;
- the Gemini process reports broker connection and Live WebSocket connection.

## Limitations

Camera rPPG is sensitive to lighting, motion, skin-region occlusion, camera exposure, glasses reflections, and face tracking quality. RMSSD requires sufficiently clean pulse peaks over a meaningful temporal window; a displayed value during weak signal should not be interpreted as reliable. The WESAD artifact is a research classifier and may require calibration or retraining for a new population, camera, environment, and task.

The project does not claim clinical accuracy, direct CCT-LSTM implementation, UBFC-Phys training, or universal emotion recognition. A research evaluation should compare HR against a reference pulse sensor, report ROI coverage and signal-quality exclusions, and evaluate stress classification using subject-independent validation.
