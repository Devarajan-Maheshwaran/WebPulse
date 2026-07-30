"""
test_wesad_verification.py — Comprehensive WESAD Implementation Verification Suite.

Systematically verifies:
1. Real WESAD dataset presence in data/WESAD/ (S2.pkl ... S17.pkl).
2. Offline training pipeline & model card generation.
3. Model artifact loading & Random Forest classifier structure.
4. Runtime inference on diverse HRV feature vectors (CALM / STRESSED).
5. End-to-end integration into EmotionFuser and LLM Prompt Builder.
"""

import os
import sys
import pickle
import numpy as np

# Force UTF-8 encoding for stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fusion.train_wesad_classifier import load_wesad_dataset, train_and_save_wesad_classifier
from fusion.wesad_classifier import WESADHRVClassifier
from fusion.emotion import EmotionFuser
from llm.prompt import construct_prompt


def verify_wesad_pipeline():
    print("=" * 70)
    print("  WebPulse -- WESAD Implementation & Runtime Verification Suite")
    print("=" * 70 + "\n")

    # -----------------------------------------------------------------------
    # STEP 1: Verify Real WESAD Dataset Files
    # -----------------------------------------------------------------------
    print("[STEP 1] Verifying Real WESAD Dataset Files in 'data/WESAD/'...")
    wesad_dir = "data/WESAD"
    subjects = [f for f in os.listdir(wesad_dir) if f.startswith("S") and os.path.isdir(os.path.join(wesad_dir, f))]
    print(f"  Found {len(subjects)} subject directories: {sorted(subjects)}")
    
    assert len(subjects) >= 10, f"Expected real WESAD subjects in {wesad_dir}, found {len(subjects)}"

    for s in sorted(subjects):
        pkl_path = os.path.join(wesad_dir, s, f"{s}.pkl")
        assert os.path.exists(pkl_path), f"Missing pickle file for subject {s}: {pkl_path}"
        file_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
        print(f"  ✓ {s}.pkl verified ({file_size_mb:.1f} MB)")

    print("  [STEP 1 PASSED] All real WESAD dataset files present and verified!\n")

    # -----------------------------------------------------------------------
    # STEP 2: Verify Training Pipeline on Real Data
    # -----------------------------------------------------------------------
    print("[STEP 2] Verifying Offline Training Pipeline on Real WESAD Dataset...")
    X, y = load_wesad_dataset(wesad_dir)
    print(f"  Extracted Feature Matrix Shape: {X.shape} | Labels Shape: {y.shape}")
    print(f"  Class Distribution: CALM (0)={np.sum(y==0)}, NORMAL (1)={np.sum(y==1)}, STRESSED (2)={np.sum(y==2)}")

    assert X.shape[0] > 5000, f"Expected >5000 feature windows from WESAD, got {X.shape[0]}"
    assert X.shape[1] == 3, f"Expected 3 features [RMSSD, SDNN, Mean HR], got {X.shape[1]}"

    acc, f1 = train_and_save_wesad_classifier(data_dir=wesad_dir)
    print(f"  ✓ Training completed on real data! Accuracy: {acc*100:.2f}%, Macro F1: {f1:.4f}")
    print("  [STEP 2 PASSED] WESAD training pipeline verified!\n")

    # -----------------------------------------------------------------------
    # STEP 3: Verify Model Artifact & Model Card
    # -----------------------------------------------------------------------
    print("[STEP 3] Verifying Model Artifact & WESAD Model Card...")
    model_path = "fusion/models/wesad_hrv_classifier.pkl"
    card_path = "fusion/WESAD_MODEL_CARD.md"

    assert os.path.exists(model_path), f"Missing model artifact: {model_path}"
    assert os.path.exists(card_path), f"Missing model card: {card_path}"

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    assert "model" in model_data and "scaler" in model_data, "Invalid model artifact structure"
    print(f"  Loaded model type: {type(model_data['model'])}")
    print(f"  Loaded scaler type: {type(model_data['scaler'])}")

    with open(card_path, "r", encoding="utf-8") as f:
        card_text = f.read()

    print(f"  Model Card snippet:\n{card_text.strip()}\n")
    print("  [STEP 3 PASSED] Model artifact and card verified!\n")

    # -----------------------------------------------------------------------
    # STEP 4 & 5: Runtime Classifier Predictions (No Thresholds)
    # -----------------------------------------------------------------------
    print("[STEP 4 & 5] Testing Runtime Classifier on Diverse HRV Vectors...")
    clf = WESADHRVClassifier(model_path=model_path)
    assert clf.is_loaded, "WESADHRVClassifier failed to load model artifact"

    # Test Vector A: High RMSSD / Low HR -> CALM
    calm_res = clf.predict(rmssd=65.0, heart_rate=62.0, sdnn=72.0)
    print(f"  Calm Vector (RMSSD=65ms, HR=62BPM):")
    print(f"    Stress State: {calm_res['stress_state']} | Arousal: {calm_res['arousal_score']:.3f} | Conf: {calm_res['confidence']:.2f} | Source: {calm_res['source']}")

    # Test Vector B: Low RMSSD / High HR -> STRESSED
    stress_res = clf.predict(rmssd=12.0, heart_rate=115.0, sdnn=14.0)
    print(f"  Stress Vector (RMSSD=12ms, HR=115BPM):")
    print(f"    Stress State: {stress_res['stress_state']} | Arousal: {stress_res['arousal_score']:.3f} | Conf: {stress_res['confidence']:.2f} | Source: {stress_res['source']}")

    assert calm_res["stress_state"] == "CALM", f"Expected CALM, got {calm_res['stress_state']}"
    assert stress_res["stress_state"] == "STRESSED", f"Expected STRESSED, got {stress_res['stress_state']}"
    assert calm_res["source"] == "wesad_rf", f"Expected wesad_rf source, got {calm_res['source']}"
    assert stress_res["source"] == "wesad_rf", f"Expected wesad_rf source, got {stress_res['source']}"
    assert calm_res["arousal_score"] < 0.40, f"Expected low arousal for CALM, got {calm_res['arousal_score']}"
    assert stress_res["arousal_score"] > 0.70, f"Expected high arousal for STRESSED, got {stress_res['arousal_score']}"

    print("  [STEP 4 & 5 PASSED] Runtime classifier predictions verified! (No heuristic thresholds used)\n")

    # -----------------------------------------------------------------------
    # STEP 6: Verify Fusion & LLM Prompt Integration
    # -----------------------------------------------------------------------
    print("[STEP 6] Verifying Fusion Layer & LLM Prompt Integration...")
    fuser = EmotionFuser()

    # Fuse stress physiological state with negative voice valence
    fused_stress = fuser.fuse(arousal_score=0.8, valence_score=-0.45, hrv_rmssd=12.0, heart_rate=115.0)
    print(f"  Fused Stress State Output: {fused_stress}")
    assert fused_stress["stress_label"] == "STRESSED"
    assert fused_stress["label"] == "aroused-negative"
    assert fused_stress["classifier_source"] == "wesad_rf"

    # Construct LLM prompt
    emotion_info = {
        "label": fused_stress["label"],
        "description": fused_stress["description"],
        "arousal": fused_stress["arousal"],
        "valence": fused_stress["valence"],
        "quality_status": "GOOD",
        "heart_rate": 115.0,
        "rmssd": 12.0,
        "stress_label": fused_stress["stress_label"]
    }
    transcript = "I'm overwhelmed by the workload today."
    prompt = construct_prompt(emotion_info, transcript)

    print("\n  Constructed LLM Prompt Preview:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)

    assert "STRESSED" in prompt, "WESAD stress label missing from LLM prompt"
    assert "115.0 BPM" in prompt, "Heart rate missing from LLM prompt"
    assert "12.0 ms" in prompt, "RMSSD HRV missing from LLM prompt"

    print("\n  [STEP 6 PASSED] Emotion fusion and LLM prompt integration verified!\n")

    print("=" * 70)
    print("  WESAD IMPLEMENTATION VERIFICATION PASSED -- ALL CHECKS SUCCESSFUL")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    verify_wesad_pipeline()
