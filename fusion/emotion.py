"""
fusion/emotion.py — Emotion Fusion Module (Russell's Circumplex Model + Sugaya Stress Classification).

RESEARCH REFERENCES:
1. Russell, J. A. (1980). "A circumplex model of affect." Journal of Personality and Social Psychology.
   - Maps emotional state across two orthogonal axes: Arousal (physiological activation) and Valence (emotional tone).
2. Sugaya et al. & PRV Emotion Papers:
   - Integrates discrete stress classification (CALM, NORMAL, STRESSED) derived from HRV RMSSD metrics.
"""

import collections

import numpy as np

from fusion.wesad_classifier import WESADHRVClassifier

_wesad_classifier_instance = WESADHRVClassifier()


class StressStateSmoother:
    """Small temporal stabilizer for live three-level WESAD state.

    This is a deliberately lightweight analogue of the temporal stage in
    CCT-LSTM: a one-sample classifier jump cannot change the displayed state,
    while sustained changes are admitted after two of the latest three
    inferences. It adds no model, I/O, or appreciable latency.
    """

    _LEVELS = ("CALM", "NORMAL", "STRESSED")

    def __init__(self, window=3, arousal_alpha=0.35):
        self._history = collections.deque(maxlen=window)
        self._stable_state = None
        self._arousal = None
        self._arousal_alpha = arousal_alpha

    def update(self, stress_state, arousal_score):
        candidate = str(stress_state or "NORMAL").upper()
        if candidate not in self._LEVELS:
            candidate = "NORMAL"
        self._history.append(candidate)

        if self._stable_state is None:
            self._stable_state = candidate
        elif candidate != self._stable_state:
            required = min(2, len(self._history))
            if sum(item == candidate for item in self._history) >= required:
                self._stable_state = candidate

        value = float(np.clip(arousal_score if arousal_score is not None else 0.5, 0.05, 0.95))
        if self._arousal is None:
            self._arousal = value
        else:
            self._arousal = self._arousal_alpha * value + (1.0 - self._arousal_alpha) * self._arousal
        return self._stable_state, float(self._arousal)


class EmotionFuser:
    """
    Lightweight two-modality fusion: the body branch contributes rPPG/WESAD
    arousal and stress, while the audio branch contributes voice valence.
    Existing HR/arousal smoothing provides temporal stabilization inspired by,
    but not a reimplementation of, CCT-LSTM-style temporal modelling.
    """

    QUADRANT_DESCRIPTIONS = {
        "aroused-positive": "High arousal & positive valence (e.g., excited, enthusiastic, joyful)",
        "aroused-negative": "High arousal & negative valence (e.g., stressed, anxious, frustrated, angry)",
        "calm-positive": "Low arousal & positive valence (e.g., relaxed, serene, content, peaceful)",
        "calm-negative": "Low arousal & negative valence (e.g., sad, depressed, fatigued, bored)"
    }

    def __init__(self, arousal_threshold=0.5, valence_threshold=0.0):
        self.arousal_threshold = arousal_threshold
        self.valence_threshold = valence_threshold
        self.wesad_classifier = _wesad_classifier_instance

    def fuse(self, arousal_score, valence_score, hrv_rmssd=None, heart_rate=70.0,
             stress_label=None, classifier_confidence=None, classifier_source=None):
        """
        Fuse body arousal and audio valence into a Russell-style emotion quadrant.
        """
        if stress_label is None:
            wesad_pred = self.wesad_classifier.predict(hrv_rmssd, heart_rate=heart_rate)
            # Override arousal score if WESAD classifier is loaded and hrv_rmssd is available.
            if hrv_rmssd is not None and wesad_pred.get("source") in ["wesad_rf", "wesad_reference_fallback"]:
                a = wesad_pred["arousal_score"]
            else:
                a = float(max(0.0, min(1.0, arousal_score if arousal_score is not None else 0.5)))
            resolved_stress = wesad_pred.get("stress_state", "NORMAL")
            confidence = wesad_pred.get("confidence", 0.85)
            source = wesad_pred.get("source", "wesad")
        else:
            # Process 1 has already classified and temporally stabilized this
            # body state. Reusing it prevents Process 2 from producing a
            # conflicting WESAD label for the same physiological window.
            a = float(max(0.0, min(1.0, arousal_score if arousal_score is not None else 0.5)))
            resolved_stress = str(stress_label).upper()
            confidence = 0.85 if classifier_confidence is None else classifier_confidence
            source = classifier_source or "process_1_wesad"

        v = float(max(-1.0, min(1.0, valence_score if valence_score is not None else 0.0)))

        is_high_arousal = a >= self.arousal_threshold
        is_positive_valence = v >= self.valence_threshold

        if is_high_arousal and is_positive_valence:
            label = "aroused-positive"
        elif is_high_arousal and not is_positive_valence:
            label = "aroused-negative"
        elif not is_high_arousal and is_positive_valence:
            label = "calm-positive"
        else:
            label = "calm-negative"

        return {
            "arousal": round(a, 3),
            "valence": round(v, 3),
            "label": label,
            "stress_label": resolved_stress,
            "description": self.QUADRANT_DESCRIPTIONS[label],
            "classifier_confidence": round(float(confidence), 2),
            "classifier_source": source,
        }


def fuse_emotions(arousal_score, valence_score, hrv_rmssd=None, heart_rate=70.0,
                  arousal_threshold=0.5, valence_threshold=0.0,
                  stress_label=None, classifier_confidence=None, classifier_source=None):
    """Convenience helper function for emotion fusion."""
    fuser = EmotionFuser(arousal_threshold=arousal_threshold, valence_threshold=valence_threshold)
    return fuser.fuse(
        arousal_score, valence_score, hrv_rmssd=hrv_rmssd, heart_rate=heart_rate,
        stress_label=stress_label, classifier_confidence=classifier_confidence,
        classifier_source=classifier_source,
    )
