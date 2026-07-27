"""
test_logging.py — Standalone Test Script for Phase 6 (Session Logging Module).

Usage:
    conda run -n webpulse python test_logging.py

This script tests:
  1. Session lifecycle management (start_session, log_entry, stop_session)
  2. JSON and CSV log file creation in sessions/ directory
  3. Verification of logged data fields & formatting
"""

import os
import json
import csv
import time
from logging_.session_logger import SessionLogger


def test_session_logger():
    print("=" * 60)
    print("  WebPulse — Phase 6 Session Logger Standalone Test")
    print("=" * 60)

    logger = SessionLogger(output_dir="sessions")
    
    # 1. Start Session
    session_id = logger.start_session(subject_id="volunteer_test_01")
    print(f"\nStarted Session: {session_id}")
    print(f"Session Active Status: {logger.is_active}")

    # 2. Log Sample Entries
    print("\nLogging sample session entries...")
    logger.log_entry(
        heart_rate=72.5,
        rmssd=35.2,
        arousal_score=0.42,
        valence_score=0.65,
        emotion_label="calm-positive",
        transcript="Hello! I am feeling relaxed today after a nice walk.",
        llm_response="That sounds wonderful. Walking is such a pleasant way to unwind.",
        raw_signal_summary={"g_mean": 124.5, "detrend_std": 2.1}
    )

    time.sleep(0.5)

    logger.log_entry(
        heart_rate=88.0,
        rmssd=18.4,
        arousal_score=0.82,
        valence_score=-0.55,
        emotion_label="aroused-negative",
        transcript="I have a big presentation in ten minutes and I'm really nervous.",
        llm_response="Take a slow deep breath. You are well prepared for this presentation.",
        raw_signal_summary={"g_mean": 126.1, "detrend_std": 4.8}
    )

    # 3. Stop Session & Export
    summary = logger.stop_session()
    print("\nSession Stopped & Data Exported.")
    print(f"  Duration: {summary['duration']} seconds")
    print(f"  Total Records: {summary['total_records']}")
    print(f"  JSON Log File: {summary['json_path']}")
    print(f"  CSV Log File:  {summary['csv_path']}")

    # 4. Verify Files Exist & Are Valid
    assert os.path.exists(summary['json_path']), "JSON file was not created!"
    assert os.path.exists(summary['csv_path']), "CSV file was not created!"

    with open(summary['json_path'], "r", encoding="utf-8") as f:
        json_data = json.load(f)
        assert json_data["session_id"] == session_id
        assert len(json_data["records"]) == 2
        print("\n[VERIFICATION] JSON schema validation PASSED.")

    with open(summary['csv_path'], "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["emotion_label"] == "calm-positive"
        assert reader[1]["emotion_label"] == "aroused-negative"
        print("[VERIFICATION] CSV schema validation PASSED.")

    print("\n[VERIFICATION RESULT] Phase 6 Session Logger module completed successfully.")
    return True


if __name__ == "__main__":
    test_session_logger()
