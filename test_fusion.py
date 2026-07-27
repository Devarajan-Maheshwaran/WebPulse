"""
test_fusion.py — Unit Tests for Component 4 (Russell Circumplex Emotion Fusion & Stress Labels).

RESEARCH REFERENCES:
- Russell, J. A. (1980). "A circumplex model of affect."
- Sugaya et al. HRV+EEG stress classification.
"""

import unittest
from fusion.emotion import EmotionFuser, fuse_emotions


class TestEmotionFusion(unittest.TestCase):

    def setUp(self):
        self.fuser = EmotionFuser()

    def test_quadrant_1_aroused_positive(self):
        """High arousal (0.85) + Positive valence (+0.6) -> aroused-positive."""
        res = self.fuser.fuse(arousal_score=0.85, valence_score=0.6)
        self.assertEqual(res["label"], "aroused-positive")

    def test_quadrant_2_aroused_negative(self):
        """High arousal (0.85) + Negative valence (-0.7) -> aroused-negative."""
        res = self.fuser.fuse(arousal_score=0.85, valence_score=-0.7)
        self.assertEqual(res["label"], "aroused-negative")

    def test_quadrant_3_calm_positive(self):
        """Low arousal (0.20) + Positive valence (+0.5) -> calm-positive."""
        res = self.fuser.fuse(arousal_score=0.20, valence_score=0.5)
        self.assertEqual(res["label"], "calm-positive")

    def test_quadrant_4_calm_negative(self):
        """Low arousal (0.15) + Negative valence (-0.4) -> calm-negative."""
        res = self.fuser.fuse(arousal_score=0.15, valence_score=-0.4)
        self.assertEqual(res["label"], "calm-negative")

    def test_stress_labels_integration(self):
        """Verify discrete stress label outputs for high vs low HRV RMSSD values."""
        calm_res = fuse_emotions(arousal_score=0.2, valence_score=0.5, hrv_rmssd=70.0)
        self.assertEqual(calm_res["stress_label"], "CALM")

        stress_res = fuse_emotions(arousal_score=0.8, valence_score=-0.5, hrv_rmssd=15.0)
        self.assertEqual(stress_res["stress_label"], "STRESSED")


if __name__ == "__main__":
    unittest.main()
