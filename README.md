# WebPulse

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Inter&size=22&duration=2800&pause=900&color=2EA44F&center=true&vCenter=true&width=760&lines=Real-time+physiology-aware+HCI;EfficientPhys+with+three+facial+ROIs;WESAD+stress+classification+%2B+Gemini+Live" alt="WebPulse animated project summary" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/EfficientPhys-ONNX-7F52FF" alt="EfficientPhys ONNX" />
  <img src="https://img.shields.io/badge/DirectML-GPU-0078D4?logo=amd&logoColor=white" alt="DirectML GPU" />
  <img src="https://img.shields.io/badge/WESAD-HRV%20classifier-2EA44F" alt="WESAD classifier" />
  <img src="https://img.shields.io/badge/Gemini-Live%20audio-4285F4?logo=google&logoColor=white" alt="Gemini Live" />
  <img src="https://img.shields.io/badge/ROI-forehead%20%2B%202%20cheeks-111827" alt="Three facial regions" />
</p>

WebPulse is a local HCI prototype for:

- **Contactless HR/HRV:** estimates pulse and variability from forehead and bilateral-cheek ROIs.
- **Affective inference:** combines EfficientPhys features with WESAD stress classification.
- **Gemini Live context:** converts physiology into bounded 1-to-5 scores before interaction.
- **Local voice loop:** streams microphone audio and plays responses without a separate cloud audio service.

Research prototype only; not a medical device or diagnostic system.

## **Future direction: federated learning**

- **Privacy:** keep face video, rPPG, voice, and biodata on-device instead of uploading raw personal signals.
- **Deployment:** share protected model updates using secure aggregation, encrypted transport, and optional differential privacy.
- **Learning:** improve models across clients without centralizing raw biometric data or face videos.
- **Generalization:** use held-out subjects, drift monitoring, and regularization; FL alone does not prevent overfitting or underfitting.
- **Hardware path:** EfficientPhys fits the current PC; PhysMamba, 3D CNNs, and other [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) models can follow when hardware allows.

## Proof of work

These captures show the running application with live inference, `GOOD` signal quality, `3/3` ROI coverage, HR/HRV output, WESAD state labels, voice transcription, and Gemini responses.

### Calm state and full ROI coverage

![WebPulse calm state with good signal](docs/evidence/webpulse-calm-good-signal.png)

### Normal state and full ROI coverage

![WebPulse normal state with good signal](docs/evidence/webpulse-normal-good-signal.png)

### Live emotion query

![WebPulse live emotion query](docs/evidence/webpulse-live-emotion-query.png)

## Architecture

```mermaid
flowchart LR
    C[Webcam] --> L[Face landmarks]
    L --> R[Forehead + left cheek + right cheek]
    R --> E[EfficientPhys ONNX]
    E --> B[Median BVP fusion]
    B --> P[HR / HRV]
    P --> W[WESAD stress model]
    W --> S[Local state broker]
    S --> M[Ten-entry scored memory]
    M --> G[Gemini Live WebSocket]
    A[Microphone] --> G
    G --> O[Native audio + HUD]
```

The runtime is split into three local processes:

| Process | Command | Role |
|---|---|---|
| Vision | `python -m rppg.rppg_server` | Webcam, landmarks, ROI extraction, EfficientPhys, BVP fusion, HR/HRV, WESAD, HUD server. |
| Broker | `python state_broker.py` | Latest-only in-memory TCP IPC on `127.0.0.1:5003`. |
| Live agent | `python app.py` | Microphone, Gemini Live, scored context, audio playback, transcripts, and interruption handling. |

## Physiology pipeline

1. MediaPipe identifies the forehead, left cheek, and right cheek.
2. Each available region is quality-checked and passed through EfficientPhys.
3. Region-level BVP predictions are median-fused to reduce local noise.
4. The fused pulse produces HR, peak-based RMSSD, arousal, and signal quality.
5. The WESAD HRV classifier produces `CALM`, `NORMAL`, or `STRESSED`.

Partial coverage continues with the available regions and is marked `WEAK_SIGNAL`. With no usable region, the system reports `NO_VALID_ROI` instead of inventing a new physiological estimate.

## Scored context and temporary memory

The broker carries raw local state. The Live agent converts it into model-facing labels:

| Field | Meaning |
|---|---|
| `wesad_stress_score_5` | `1` calm, `3` normal, `5` stressed. |
| `arousal_score_5` | `1` low activation to `5` high activation. |
| `valence_score_5` | `1` negative to `5` positive. |
| `hr_score_5` | Bounded heart-rate category. |
| `hrv_stress_load_5` | Inverse HRV calmness/stress-load score. |
| `signal_quality` / `roi_coverage` | Reliability context for interpretation. |

`live_emotion_agent.py` maintains a fixed `deque(maxlen=10)`. At most one scored state is added per second. When the eleventh state arrives, the oldest state is discarded. At speech start and speech end, the agent reads the newest scored state and sends it as incomplete context, so physiology updates do not create unsolicited model turns.

## Gemini Live behavior

- Persistent bidirectional WebSocket for microphone input and native audio output.
- 16 kHz PCM microphone input and 24 kHz speaker output.
- `clientContent` with `turnComplete: false` for answer-time physiology context.
- User audio is the only generation trigger; idle telemetry is retained locally.
- Speaker playback is filtered from microphone forwarding, while genuine speech can barge in and clear the playback queue.
- The system prompt treats the scores as uncertain research context and avoids medical claims.

## Research basis

The following review connects the research ideas to the implemented MVP:

| Research reference | Review finding | WebPulse implementation and MVP choice |
|---|---|---|
| [CCT-LSTM, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Ziaratnia_Multimodal_Deep_Learning_for_Remote_Stress_Estimation_Using_CCT-LSTM_WACV_2024_paper.html) | Multimodal video and physiological branches benefit from temporal modeling. | Uses EfficientPhys, HRV/WESAD, voice valence, and deterministic fusion. Full CCT-LSTM was deferred because it needs more data, training, and hardware. |
| [CAST-Phys, 2025](https://arxiv.org/html/2507.06080v1) and [Khan et al., 2024](https://link.springer.com/article/10.1007/s00530-024-01302-2) | Contactless multimodal affect sensing remains sensitive to subject, lighting, and modality variation. | Uses explicit ROI quality, signal labels, local voice features, and conservative uncertainty handling rather than claiming generalized emotion recognition. |
| [Zhou et al., 2023](https://www.sciencedirect.com/science/article/abs/pii/S1046202323001433) and [SympCam, 2025](https://www.themoonlight.io/en/review/sympcam-remote-optical-measurement-of-sympathetic-arousal) | Camera-derived pulse/PRV can support dimensional arousal and sympathetic-state estimation. | Fuses BVP-derived HR/HRV into an arousal score; it is treated as an estimate, not a clinical measurement. |
| [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) and [paper](https://arxiv.org/abs/2210.00716) | Reproducible rPPG pipelines need defined preprocessing, model evaluation, and dataset discipline. | The ROI and EfficientPhys ONNX path follows this research direction; the MVP uses a lightweight local runtime for the available PC. |
| [EfficientPhys](https://arxiv.org/abs/2207.04850) | Efficient spatio-temporal rPPG is suitable for resource-constrained inference. | Current deep backend; higher-capacity 3D CNNs and PhysMamba are future hardware-dependent options. |
| [Suzuki et al., 2021](https://iaiai.org/letters/index.php/liir/article/view/330) and [Nishibe et al., 2025](https://iaiai.org/letters/index.php/liir/article/view/330) | HRV features such as RMSSD can support emotion-related physiological modeling. | Uses RMSSD, HR, WESAD classification, and a bounded 1-to-5 bio-state score rather than a larger feature-selection model. |
| [Ikeda, Horie, and Sugaya](https://doi.org/10.1016/j.procs.2017.08.198) and [A Circumplex Model of Affect](https://doi.org/10.1037/h0077714) | Biological signals can be mapped into interaction-facing arousal-valence states. | Provides the interaction framing and affect-fusion representation used by the Live companion. |

WebPulse does not reproduce CCT-LSTM, claim paper-level benchmark performance, or replace subject-independent evaluation with live screenshots.

## Installation and run

Create a Python environment and install the project dependencies:

```powershell
pip install -r requirements.txt
```

Create `.env` with:

```text
GEMINI_API_KEY=your_key_here
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
GEMINI_LIVE_VOICE=Puck
RPPG_BACKEND=deep
```

Open three PowerShell terminals in the project root:

```powershell
# Terminal 1
python state_broker.py

# Terminal 2
python -m rppg.rppg_server

# Terminal 3
python app.py
```

Expected runtime indicators include `EfficientPhys execution: GPU (DirectML)`, `ROIs: 3/3`, broker broadcasts, and `Gemini Live connected`.

## Verification

```powershell
python -m py_compile live_emotion_agent.py rppg\capture.py rppg\deep_engine.py rppg\rppg_server.py rppg\hud.py fusion\emotion.py
python -m unittest test_fusion.py
```

The checked runtime path verifies ONNX metadata, DirectML selection, three-ROI inference, partial-coverage behavior, zero-coverage behavior, scored context bounds, ten-entry memory eviction, and answer-time context serialization.

## Limitations

Lighting, motion, exposure, glasses reflections, landmark errors, skin visibility, and individual physiology affect rPPG quality. RMSSD is meaningful only when the fused pulse has reliable peaks. The WESAD classifier may require recalibration for a new population, task, camera, and environment.

For a research submission, report ROI coverage, signal exclusions, HR error against a reference sensor, latency, and subject-independent WESAD evaluation.

<p align="center">
  <img src="https://img.shields.io/badge/Status-research%20prototype-2EA44F" alt="Research prototype" />
  <img src="https://img.shields.io/badge/Inference-local%20first-111827" alt="Local first inference" />
  <img src="https://img.shields.io/badge/Audio-bidirectional%20streaming-4285F4" alt="Bidirectional streaming" />
</p>
