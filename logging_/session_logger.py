"""
logging_/session_logger.py — Session logging and data export module.

Implements FR-9: Session Logging and FR-10: Test Session Support.
Manages discrete recording sessions, buffers timestamped frame/window records,
and exports full session data to CSV and JSON formats in the sessions/ directory.
"""

import os
import json
import csv
import time
from datetime import datetime


class SessionLogger:
    """
    Manages session lifecycle (start/stop) and exports structured logs per session.
    """

    def __init__(self, output_dir="sessions"):
        self.output_dir = output_dir
        self.current_session_id = None
        self.session_start_time = None
        self.subject_id = None
        self.records = []
        self.is_active = False

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def start_session(self, subject_id="subject_pilot"):
        """
        Start a new timestamped recording session.
        
        Args:
            subject_id (str): Identifier for the test subject/volunteer.
            
        Returns:
            str: Generated unique session ID.
        """
        self.subject_id = subject_id
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = f"session_{timestamp_str}_{subject_id}"
        self.session_start_time = time.time()
        self.records = []
        self.is_active = True
        return self.current_session_id

    def log_entry(self, heart_rate, rmssd, arousal_score, valence_score, emotion_label, transcript, llm_response, raw_signal_summary=None):
        """
        Log a timestamped record entry within the active session.
        
        Args:
            heart_rate (float): Heart rate estimate in BPM.
            rmssd (float): RMSSD metric in ms.
            arousal_score (float): Arousal score [0.0, 1.0].
            valence_score (float): Valence score [-1.0, 1.0].
            emotion_label (str): Fused emotion quadrant label.
            transcript (str): Speech-to-text transcript.
            llm_response (str): Generated LLM response.
            raw_signal_summary (dict): Optional signal summary stats.
            
        Returns:
            bool: True if entry was logged, False if no session active.
        """
        if not self.is_active:
            return False

        entry_time = time.time()
        relative_time_sec = round(entry_time - self.session_start_time, 2)
        iso_timestamp = datetime.fromtimestamp(entry_time).isoformat()

        record = {
            "timestamp_iso": iso_timestamp,
            "elapsed_seconds": relative_time_sec,
            "heart_rate_bpm": round(heart_rate, 2) if heart_rate is not None else None,
            "rmssd_ms": round(rmssd, 2) if rmssd is not None else None,
            "arousal_score": round(arousal_score, 3) if arousal_score is not None else None,
            "valence_score": round(valence_score, 3) if valence_score is not None else None,
            "emotion_label": emotion_label,
            "transcript": transcript or "",
            "llm_response": llm_response or "",
            "raw_signal_summary": raw_signal_summary or {}
        }
        self.records.append(record)
        return True

    def stop_session(self):
        """
        Stop active session and export recorded entries to JSON and CSV files.
        
        Returns:
            dict: Summary dictionary containing filepaths and stats, or None if inactive.
        """
        if not self.is_active or not self.current_session_id:
            return None

        self.is_active = False
        session_duration = round(time.time() - self.session_start_time, 2)

        session_filename_base = os.path.join(self.output_dir, self.current_session_id)
        json_path = f"{session_filename_base}.json"
        csv_path = f"{session_filename_base}.csv"

        # Construct full JSON structure
        session_data = {
            "session_id": self.current_session_id,
            "subject_id": self.subject_id,
            "start_time": datetime.fromtimestamp(self.session_start_time).isoformat(),
            "duration_seconds": session_duration,
            "total_records": len(self.records),
            "records": self.records
        }

        # Export JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        # Export CSV
        fieldnames = [
            "timestamp_iso",
            "elapsed_seconds",
            "heart_rate_bpm",
            "rmssd_ms",
            "arousal_score",
            "valence_score",
            "emotion_label",
            "transcript",
            "llm_response"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                row = {k: rec.get(k, "") for k in fieldnames}
                writer.writerow(row)

        return {
            "session_id": self.current_session_id,
            "json_path": json_path,
            "csv_path": csv_path,
            "duration": session_duration,
            "total_records": len(self.records)
        }
