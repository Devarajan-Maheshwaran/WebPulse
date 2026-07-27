"""
test_fusion.py — Unit tests for Phase 4 (Emotion Fusion Module).

Usage:
    conda run -n webpulse python test_fusion.py

[DISCLAIMER] These are unit tests executed with synthetic (arousal, valence) numerical pairs
to verify mapping logic according to Russell's Circumplex Model. They do NOT serve as
evidence that the full hardware pipeline works on real empirical data.
"""

import unittest
from fusion.emotion import fuse_emotions, EmotionFuser


class TestEmotionFusion(unittest.TestCase):
    """Unit tests for Russell's Circumplex Model quadrant mapping logic."""

    def setUp(self):
        self.fuser = EmotionFuser(arousal_threshold=0.5, valence_threshold=0.0)

    def test_quadrant_1_aroused_positive(self):
        """Test High Arousal (0.8) and Positive Valence (+0.6) -> 'aroused-positive'"""
        res = self.fuser.fuse(arousal_score=0.8, valence_score=0.6)
        self.assertEqual(res["label"], "aroused-positive")
        self.assertIn("excited", res["description"].lower())

    def test_quadrant_2_aroused_negative(self):
        """Test High Arousal (0.9) and Negative Valence (-0.7) -> 'aroused-negative'"""
        res = self.fuser.fuse(arousal_score=0.9, valence_score=-0.7)
        self.assertEqual(res["label"], "aroused-negative")
        self.assertIn("stressed", res["description"].lower())

    def test_quadrant_3_calm_positive(self):
        """Test Low Arousal (0.2) and Positive Valence (+0.4) -> 'calm-positive'"""
        res = self.fuser.fuse(arousal_score=0.2, valence_score=0.4)
        self.assertEqual(res["label"], "calm-positive")
        self.assertIn("relaxed", res["description"].lower())

    def test_quadrant_4_calm_negative(self):
        """Test Low Arousal (0.1) and Negative Valence (-0.5) -> 'calm-negative'"""
        res = self.fuser.fuse(arousal_score=0.1, valence_score=-0.5)
        self.assertEqual(res["label"], "calm-negative")
        self.assertIn("sad", res["description"].lower())

    def test_boundary_conditions(self):
        """Test exact threshold boundary cases."""
        # Exact threshold -> high arousal, positive valence
        res_bound = self.fuser.fuse(arousal_score=0.5, valence_score=0.0)
        self.assertEqual(res_bound["label"], "aroused-positive")

        # Extreme clamping checks
        res_clamp = self.fuser.fuse(arousal_score=1.5, valence_score=-2.0)
        self.assertEqual(res_clamp["arousal"], 1.0)
        self.assertEqual(res_clamp["valence"], -1.0)
        self.assertEqual(res_clamp["label"], "aroused-negative")


if __name__ == "__main__":
    print("=" * 60)
    print("  WebPulse — Phase 4 Emotion Fusion Unit Tests (Synthetic Pairs)")
    print("=" * 60)
    unittest.main()
