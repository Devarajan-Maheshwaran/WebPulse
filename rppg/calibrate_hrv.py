"""
rppg/calibrate_hrv.py — Calibration Helper for HRV-Based Stress Thresholds.

RESEARCH REFERENCE & MOTIVATION:
Inspired by Sugaya Lab ("Emotion estimation from EEG and HRV indices with ML")
and PRV Emotion Recognition research.
Because individual baseline HRV (RMSSD) varies based on age, fitness, and autonomic tone,
thresholds for CALM, NORMAL, and STRESSED must be empirically calibrated on real session logs.

Usage:
    python rppg/calibrate_hrv.py [--sessions_dir sessions]
"""

import os
import glob
import json
import argparse
import numpy as np


def calibrate_hrv_thresholds(sessions_dir="sessions"):
    """
    Read session log files, extract recorded RMSSD metrics, compute statistics,
    and suggest empirical thresholds for CALM, NORMAL, and STRESSED classification.
    """
    print("=" * 70)
    print("  WebPulse — Empirical HRV/RMSSD Threshold Calibration Helper")
    print("  Inspired by Sugaya Lab HRV Feature Calibration")
    print("=" * 70 + "\n")

    json_files = glob.glob(os.path.join(sessions_dir, "*.json"))
    if not json_files:
        print(f"[WARNING] No session JSON files found in directory '{sessions_dir}'.")
        print("Run a live WebPulse session first to log real physiological data.\n")
        return None

    all_rmssd = []

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                entries = data.get("log_entries", [])
                for entry in entries:
                    rmssd = entry.get("rmssd")
                    if rmssd is not None and isinstance(rmssd, (int, float)) and rmssd > 0:
                        all_rmssd.append(float(rmssd))
        except Exception as e:
            print(f"[WARNING] Could not parse file '{file_path}': {e}")

    if not all_rmssd:
        print("[WARNING] Session log files were found, but no valid RMSSD numeric records exist.")
        print("Ensure face tracking and heart rate estimation run continuously during live sessions.\n")
        return None

    rmssd_arr = np.array(all_rmssd)
    min_val = np.min(rmssd_arr)
    max_val = np.max(rmssd_arr)
    mean_val = np.mean(rmssd_arr)
    std_val = np.std(rmssd_arr)
    p25 = np.percentile(rmssd_arr, 25)
    p50 = np.median(rmssd_arr)
    p75 = np.percentile(rmssd_arr, 75)

    print(f"Parsed {len(json_files)} session file(s) with {len(all_rmssd)} valid RMSSD data points.")
    print("-" * 60)
    print(f"  Min RMSSD:         {min_val:.2f} ms")
    print(f"  Max RMSSD:         {max_val:.2f} ms")
    print(f"  Mean RMSSD:        {mean_val:.2f} ms (std: {std_val:.2f})")
    print(f"  25th Percentile:   {p25:.2f} ms")
    print(f"  Median (50th):     {p50:.2f} ms")
    print(f"  75th Percentile:   {p75:.2f} ms")
    print("-" * 60)

    # Sugaya-style Empirical Threshold Recommendation:
    # 25th percentile -> stress_threshold (below this = STRESSED)
    # 75th percentile -> calm_threshold   (above this = CALM)
    print("\n--- SUGGESTED EMPIRICAL THRESHOLDS ---")
    print(f"  STRESSED Threshold (p25):  < {p25:.1f} ms  (High Arousal)")
    print(f"  NORMAL Range (p25 - p75): {p25:.1f} ms – {p75:.1f} ms  (Moderate Arousal)")
    print(f"  CALM Threshold (p75):      > {p75:.1f} ms  (Low Arousal)")
    print("\n[NOTE] Update `baseline_min_rmssd` and `baseline_max_rmssd` in `rppg/hrv.py` with these values.")
    print("=" * 70 + "\n")

    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "p25": p25,
        "p75": p75
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate HRV/RMSSD stress thresholds from WebPulse session logs.")
    parser.add_argument("--sessions_dir", type=str, default="sessions", help="Directory containing session logs")
    args = parser.parse_args()

    calibrate_hrv_thresholds(sessions_dir=args.sessions_dir)


if __name__ == "__main__":
    main()
