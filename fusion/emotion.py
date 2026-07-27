"""
fusion/emotion.py — Emotion Fusion Module (Russell's Circumplex Model).

Implements FR-5: Emotion Fusion.
Combines (arousal, valence) scores into one of four emotion-quadrant labels:
  - Quadrant 1: High Arousal, Positive Valence -> "aroused-positive" (Excited / Happy)
  - Quadrant 2: High Arousal, Negative Valence -> "aroused-negative" (Stressed / Angry / Anxious)
  - Quadrant 3: Low Arousal, Positive Valence  -> "calm-positive"    (Relaxed / Content)
  - Quadrant 4: Low Arousal, Negative Valence  -> "calm-negative"    (Sad / Bored / Depressed)
"""


class EmotionFuser:
    """
    Fuses arousal and valence scores per Russell's Circumplex Model of Affect.
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

    def fuse(self, arousal_score, valence_score):
        """
        Map continuous (arousal, valence) pair to emotion label & metadata.
        
        Args:
            arousal_score (float): Physiological arousal/stress score [0.0, 1.0].
            valence_score (float): Voice tone valence score [-1.0, 1.0].
            
        Returns:
            dict: {
                "arousal": float,
                "valence": float,
                "label": str,
                "description": str
            }
        """
        # Clamp inputs to valid ranges
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

        return {
            "arousal": round(a, 3),
            "valence": round(v, 3),
            "label": label,
            "description": self.QUADRANT_DESCRIPTIONS[label]
        }


def fuse_emotions(arousal_score, valence_score, arousal_threshold=0.5, valence_threshold=0.0):
    """Convenience helper function for emotion fusion."""
    fuser = EmotionFuser(arousal_threshold=arousal_threshold, valence_threshold=valence_threshold)
    return fuser.fuse(arousal_score, valence_score)
