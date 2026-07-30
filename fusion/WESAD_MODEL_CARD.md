# WESAD HRV Stress/Arousal Classifier Model Card

## Model Architecture
- **Classifier**: Scikit-Learn `RandomForestClassifier` (n_estimators=60, max_depth=6)
- **Input Features**: `[RMSSD (ms), SDNN (ms), Mean HR (BPM)]`
- **Output Classes**: `0: CALM`, `1: NORMAL/AMUSED`, `2: STRESSED`
- **Data Source**: WESAD Wearable Dataset

## Evaluation Metrics (Held-Out Test Split)
- **Accuracy**: 72.17%
- **Macro F1 Score**: 0.5921
- **Artifact Path**: `fusion/models/wesad_hrv_classifier.pkl`
