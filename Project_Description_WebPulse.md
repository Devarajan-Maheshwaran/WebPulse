# Project Description: WebPulse

## One-line pitch
A webcam-only, contact-free system that estimates emotional arousal from a remote-PPG pulse signal, fuses it with voice-tone valence, and drives an LLM to generate an emotionally-appropriate response — proposed as a low-compliance complement to wearable-sensor-based emotion-aware companion robot pipelines.

## Problem
Emotion-aware companion robots for elderly users typically rely on physiological sensors (EEG, PPG wearables) to estimate emotional state before generating a robot/LLM response. This approach is technically strong but has a well-documented real-world limitation: many elderly users resist wearing biosensors daily due to comfort, cost, or stigma, which limits real-world deployment and data continuity (ASME 2024; EAI Endorsed Transactions review, 2025).

## Proposed approach
Instead of replacing or competing with physiological-sensor pipelines, WebPulse explores whether contactless signals — a webcam-derived pulse signal (remote PPG) and voice tone — can approximate the same arousal/valence estimate well enough to drive a similar LLM-based empathetic response, without requiring any wearable device. This directly follows the direction opened by CAST-Phys (arXiv, 2025), which introduced facial-video-derived physiological signals as a substitute for contact sensors in affect recognition, and camera-based PRV research (2023) showing measurable correlation between webcam-derived pulse variability and physiological arousal.

## What was built
1. A face/ROI-based rPPG signal extraction pipeline (built on established open-source techniques, not implemented from scratch).
2. An HRV-based arousal scoring module operating on the extracted pulse waveform.
3. A voice-based valence scoring module using pitch/energy features.
4. A fusion step mapping (arousal, valence) to one of four emotion-quadrant labels (Russell's Circumplex Model).
5. An LLM-based response generator that takes the emotion label and a live speech transcript and produces an empathetic reply, optionally spoken via TTS.
6. A session logger that records raw and derived data per test session for review.

## What was tested
The system was run in short live sessions with several volunteer test subjects (friends), under informed consent. Each session logged the pulse waveform/HR trace, arousal score, valence score, resulting emotion label, and the LLM's generated response, with timestamps. Screen recordings were captured for a subset of sessions to demonstrate the live pipeline end-to-end.

## Why this matters
This is a small, honest proof-of-concept showing that a contactless, zero-compliance-cost sensing approach can plausibly support the same "physiological signal -> emotion -> LLM response" architecture used in current elderly-companion-robot research, without requiring a wearable. It is proposed as a complementary tool, not a replacement, for use cases where wearable adoption is a barrier — a gap explicitly noted in recent literature but, as of the papers reviewed, not yet addressed by combining webcam-based sensing with LLM-driven response generation in one tested pipeline.

## Key references
See README.md and SRS_WebPulse.md for the full list of research papers and open-source implementations this project is grounded in and built on top of.
