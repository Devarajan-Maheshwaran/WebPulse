# WebPulse

WebPulse is a research prototype for affect-aware interaction. It uses camera-based rPPG to estimate local HR/HRV, a WESAD-trained stress classifier, voice valence, and a Gemini Live spoken companion. Responses are conditioned on fused body and audio context, not presented as medical assessment or diagnosis.

## System Architecture

- **Deep rPPG:** EfficientPhys from rPPG-Toolbox runs through ONNX Runtime with DirectML GPU preference and CPU fallback. MediaPipe capture supplies forehead and cheek regions.
- **HR/HRV and stress:** The local BVP stream yields heart rate and time-domain PRV/HRV features, including RMSSD. A WESAD-derived RandomForest produces `CALM`, `NORMAL`, or `STRESSED` plus arousal; a small temporal stabilizer rejects isolated label jumps.
- **Multimodal fusion:** The body branch supplies HR/HRV, WESAD state, signal quality, and an interpretable 0–5 bio-state rating. The audio branch supplies voice valence and live speech. Together they form a Russell-style arousal–valence state.
- **LLM companion:** Gemini Live receives labeled background context, streams native audio, and supports barge-in interruption. Raw BPM and RMSSD remain local.

Run the local processes in order:

```powershell
python state_broker.py
python -m rppg.rppg_server
python app.py
```

## Research-Backed Design

### Deep rPPG and remote HR/HRV

- [Liu et al., *rPPG-Toolbox: Deep Remote PPG Toolbox*, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d7d0d548a6317407e02230f15ce75817-Abstract-Datasets_and_Benchmarks.html) informs the EfficientPhys integration and general deep-rPPG evaluation approach.
- [Liu et al., *EfficientPhys*, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.html) motivates the lightweight camera physiological-inference path.

### HRV-based stress and interactive emotion mapping

- [Schmidt et al., *Introducing WESAD*, ICMI 2018](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) provides the stress/affect framing used by the local HRV classifier.
- [Ikeda, Horie, and Sugaya, *Estimating Emotion with Biological Information for Robot Interaction*, 2017](https://doi.org/10.1016/j.procs.2017.08.198) conceptually supports mapping physiological indices to interaction-facing affect.
- [Russell, *A Circumplex Model of Affect*, 1980](https://doi.org/10.1037/h0077714) informs the arousal–valence fusion of body arousal and voice valence.

### Multimodal and temporal design

- [Ziaratnia, Laohakangvalvit, Sugaya, and Sripian, *Multimodal Deep Learning for Remote Stress Estimation Using CCT-LSTM*, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Ziaratnia_Multimodal_Deep_Learning_for_Remote_Stress_Estimation_Using_CCT-LSTM_WACV_2024_paper.html) motivates explicit modality branches and lightweight temporal stabilization. WebPulse does not reproduce CCT-LSTM or train on UBFC-Phys.

## Implementation Notes

- **Backend:** EfficientPhys ONNX, multi-ROI capture, local rPPG/PRV processing, and three-process local IPC.
- **Stress classifier:** The WESAD-derived RandomForest uses RMSSD, estimated SDNN, and mean HR; its live output is temporally stabilized before fusion.
- **Bio-data rating:** `bio_state_score_5` is a monotonic stress/activation summary: `0` is unavailable/unreliable physiology, `1` is very calm, `3` is moderate, and `5` is high activation/stress. Gemini receives this labeled scale and voice valence, never raw HR/RMSSD.

## Limitations and Future Work

WebPulse is inspired by CCT-LSTM multimodal stress estimation, not a direct implementation or UBFC-Phys reproduction. Future work may validate a small per-modality temporal model or additional contactless datasets when this can be done without compromising real-time behavior.
