"""
test_arousal.py — Unit Tests for Component 3 (Camera-Based PRV/HRV Mapping to Arousal & Stress).

RESEARCH MOTIVATION & REFERENCE:
- Camera-Based PRV Emotion Recognition (2023):
  "Dimensional emotion recognition from camera-based pulse rate variability (PRV) features."
- Validates that camera-derived HRV/PRV metrics serve as a 1D physiological proxy for affective arousal/stress.
"""

import unittest
from rppg.hrv import map_hrv_to_arousal, classify_stress


class TestArousalAndStressMapping(unittest.TestCase):
    """
    Unit tests for Component 3 HRV -> Arousal mapping & Component 2 Sugaya stress classification.
    """

    def test_high_hrv_maps_to_low_arousal_and_calm(self):
        """High HRV variability (e.g. 75 ms RMSSD) -> Low arousal score & CALM state."""
        rmssd = 75.0
        arousal = map_hrv_to_arousal(rmssd, baseline_min_rmssd=15.0, baseline_max_rmssd=80.0)
        stress_label, arousal_cat = classify_stress(rmssd, calm_thresh=50.0, stress_thresh=25.0)

        self.assertLess(arousal, 0.3, f"Expected low arousal (<0.3) for high RMSSD 75ms, got {arousal}")
        self.assertEqual(stress_label, "CALM")
        self.assertEqual(arousal_cat, "LOW AROUSAL")

    def test_moderate_hrv_maps_to_moderate_arousal_and_normal(self):
        """Moderate HRV variability (e.g. 35 ms RMSSD) -> Moderate arousal score & NORMAL state."""
        rmssd = 35.0
        arousal = map_hrv_to_arousal(rmssd, baseline_min_rmssd=15.0, baseline_max_rmssd=80.0)
        stress_label, arousal_cat = classify_stress(rmssd, calm_thresh=50.0, stress_thresh=25.0)

        self.assertTrue(0.3 <= arousal <= 0.8, f"Expected moderate arousal [0.3, 0.8] for RMSSD 35ms, got {arousal}")
        self.assertEqual(stress_label, "NORMAL")
        self.assertEqual(arousal_cat, "MODERATE AROUSAL")

    def test_low_hrv_maps_to_high_arousal_and_stressed(self):
        """Low HRV variability (e.g. 12 ms RMSSD) -> High arousal score & STRESSED state."""
        rmssd = 12.0
        arousal = map_hrv_to_arousal(rmssd, baseline_min_rmssd=15.0, baseline_max_rmssd=80.0)
        stress_label, arousal_cat = classify_stress(rmssd, calm_thresh=50.0, stress_thresh=25.0)

        self.assertGreaterEqual(arousal, 0.8, f"Expected high arousal (>=0.8) for low RMSSD 12ms, got {arousal}")
        self.assertEqual(stress_label, "STRESSED")
        self.assertEqual(arousal_cat, "HIGH AROUSAL")

    def test_none_hrv_returns_default_neutral(self):
        """Uncertain/None HRV returns default neutral arousal 0.5 & NORMAL state."""
        rmssd = None
        arousal = map_hrv_to_arousal(rmssd)
        stress_label, arousal_cat = classify_stress(rmssd)

        self.assertEqual(arousal, 0.5)
        self.assertEqual(stress_label, "NORMAL")


if __name__ == "__main__":
    unittest.main()
