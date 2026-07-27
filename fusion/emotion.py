"""
fusion/emotion.py — Emotion Fusion Module (Russell's Circumplex Model + Sugaya Stress Classification).

RESEARCH REFERENCES:
1. Russell, J. A. (1980). "A circumplex model of affect." Journal of Personality and Social Psychology.
   - Maps emotional state across two orthogonal axes: Arousal (physiological activation) and Valence (emotional tone).
2. Sugaya et al. & PRV Emotion Papers:
   - Integrates discrete stress classification (CALM, NORMAL, STRESSED) derived from HRV RMSSD metrics.
"""

from rppg.hrv import classify_stress


class EmotionFuser:
    """
    Fuses physiological arousal and voice valence per Russell's Circumplex Model of Affect.
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

    def fuse(self, arousal_score, valence_score, hrv_rmssd=None):
        """
        Map continuous (arousal, valence) pair to emotion quadrant & stress label.
        
        Args:
            arousal_score (float): Physiological arousal score [0.0, 1.0].
            valence_score (float): Voice tone valence score [-1.0, 1.0].
            hrv_rmssd (float, optional): RMSSD HRV metric in ms for Sugaya stress classification.
            
        Returns:
            dict: {
                "arousal": float,
                "valence": float,
                "label": str,
                "stress_label": str,
                "description": str
            }
        """
        a = float(max(0.0, min(1.0, arousal_score if arousal_score is not None else 0.5)))
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

        stress_label, _ = classify_stress(hrv_rmssd)

        return {
            "arousal": round(a, 3),
            "valence": round(v, 3),
            "label": label,
            "stress_label": stress_label,
            "description": self.QUADRANT_DESCRIPTIONS[label]
        }


def fuse_emotions(arousal_score, valence_score, hrv_rmssd=None, arousal_threshold=0.5, valence_threshold=0.0):
    """Convenience helper function for emotion fusion."""
    fuser = EmotionFuser(arousal_threshold=arousal_threshold, valence_threshold=valence_threshold)
    return fuser.fuse(arousal_score, valence_score, hrv_rmssd=hrv_rmssd)
