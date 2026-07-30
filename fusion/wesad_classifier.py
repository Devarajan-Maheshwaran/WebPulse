"""
fusion/wesad_classifier.py — Runtime WESAD-Trained HRV Stress/Arousal Classifier.

Replaces heuristic HRV thresholds with a trained lightweight Scikit-Learn Random Forest model
trained on WESAD (Wearable Stress and Affect Detection) dataset BVP/HRV signals.

Outputs:
  - stress_label: 'CALM', 'NORMAL', or 'STRESSED'
  - arousal_score: Continuous float in [0.05, 0.95]
  - confidence: Classifier probability score
"""

import sys
import os
import time
import pickle
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class WESADHRVClassifier:
    """Runtime wrapper for WESAD-trained HRV stress and arousal classifier."""

    def __init__(self, model_path="fusion/models/wesad_hrv_classifier.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_loaded = False

        self.load_model()

    def load_model(self):
        """Loads trained WESAD model pickle file if present."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data.get("model")
                    self.scaler = data.get("scaler")
                    self.is_loaded = True
                print(f"[WESAD Classifier SUCCESS] Loaded trained HRV model from {self.model_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load WESAD model ({e}). Using physiological fallback.")
                self.is_loaded = False
        else:
            print(f"[WESAD Classifier] No trained model found at {self.model_path}. Using WESAD reference fallback.")
            self.is_loaded = False

    def predict(self, rmssd, heart_rate=70.0, sdnn=None):
        """
        Predicts stress label and continuous arousal score from live HRV metrics.

        Args:
            rmssd (float): Root mean square of successive differences (ms).
            heart_rate (float): Heart rate in BPM (default 70.0).
            sdnn (float): Standard deviation of NN intervals (default estimated from RMSSD).

        Returns:
            dict: {
                'stress_state': 'CALM' | 'NORMAL' | 'STRESSED',
                'arousal_score': float in [0.05, 0.95],
                'confidence': float,
                'source': 'wesad_rf' | 'wesad_fallback'
            }
        """
        if rmssd is None or np.isnan(rmssd):
            return {
                "stress_state": "NORMAL",
                "arousal_score": 0.50,
                "confidence": 0.50,
                "source": "wesad_fallback"
            }

        if sdnn is None:
            sdnn = rmssd * 1.15  # Empirical SDNN to RMSSD scaling ratio

        # If default 70.0 BPM was passed without explicit HR, adjust HR physiologically based on RMSSD
        if heart_rate == 70.0:
            if rmssd < 24.0:
                heart_rate = 88.0
            elif rmssd > 45.0:
                heart_rate = 66.0

        features = np.array([[rmssd, sdnn, heart_rate]], dtype=np.float32)


        if self.is_loaded and self.model is not None:
            try:
                if self.scaler is not None:
                    scaled_feats = self.scaler.transform(features)
                else:
                    scaled_feats = features

                probs = self.model.predict_proba(scaled_feats)[0]
                pred_class = int(np.argmax(probs))
                conf = float(np.max(probs))

                # Class mapping: 0 -> CALM, 1 -> NORMAL/AMUSED, 2 -> STRESSED
                labels = ["CALM", "NORMAL", "STRESSED"]
                raw_pred_idx = min(pred_class, 2)
                stress_state = labels[raw_pred_idx]

                # Keep the trained model grounded in the physiological safety
                # bounds used by the WESAD reference fallback. A low-HRV sample
                # must not be reported as calm solely because of model drift.
                if rmssd <= 22.0 and stress_state == "CALM":
                    stress_state = "STRESSED"
                elif rmssd >= 45.0 and stress_state == "STRESSED":
                    stress_state = "CALM"


                # Continuous arousal calculation from class probabilities:
                # P(STRESSED) increases arousal, P(CALM) decreases arousal
                p_calm = probs[0] if len(probs) > 0 else 0.33
                p_stressed = probs[2] if len(probs) > 2 else 0.33
                raw_arousal = 0.50 + 0.45 * (p_stressed - p_calm)
                arousal_score = float(np.clip(raw_arousal, 0.05, 0.95))

                if os.getenv("WEBPULSE_DEBUG", "1") == "1" and time.time() % 4 < 0.1:
                    print(f"[DEBUG STEP 5: WESAD CLASSIFIER] Input Feature Vector [RMSSD, SDNN, HR]: [{rmssd:.1f}ms, {sdnn:.1f}ms, {heart_rate:.1f}BPM] | Output Class: {stress_state} | Confidence: {conf:.2f} | Arousal: {arousal_score:.2f}")

                return {
                    "stress_state": stress_state,
                    "arousal_score": arousal_score,
                    "confidence": conf,
                    "source": "wesad_rf"
                }
            except Exception as e:
                print(f"[WESAD Classifier Error] Prediction failed ({e}). Falling back.")


        # Physiological Reference Fallback (matching WESAD population stats)
        # Baseline / Calm: RMSSD > 42 ms -> High parasympathetic tone
        # Stress: RMSSD < 22 ms -> Sympathetic dominance
        if rmssd >= 45.0:
            stress_state = "CALM"
            arousal_score = float(np.clip(0.50 - (rmssd - 45.0) / 100.0, 0.05, 0.40))
        elif rmssd <= 22.0:
            stress_state = "STRESSED"
            arousal_score = float(np.clip(0.60 + (22.0 - rmssd) / 50.0, 0.65, 0.95))
        else:
            stress_state = "NORMAL"
            arousal_score = float(np.clip(0.40 + (45.0 - rmssd) / 23.0 * 0.20, 0.40, 0.60))

        return {
            "stress_state": stress_state,
            "arousal_score": arousal_score,
            "confidence": 0.85,
            "source": "wesad_reference_fallback"
        }
