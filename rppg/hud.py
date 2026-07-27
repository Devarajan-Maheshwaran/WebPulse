"""
rppg/hud.py — Minimal Left-Side Visual HUD Renderer for WebPulse.

Renders a sleek, high-contrast, unobtrusive minimal sidebar on the left side of the video window:
  - Keeps the central video stream & user face 100% clear.
  - Displays all live physiological, audio, emotion, and LLM metrics in clean vertical typography.
"""

import cv2
import numpy as np


def draw_overlay_hud(frame, hud_data):
    """
    Draw a constant minimal HUD sidebar on the left side of the screen.
    
    Args:
        frame (ndarray): OpenCV BGR frame.
        hud_data (dict): Current metrics state.
        
    Returns:
        ndarray: Frame with minimal left-side HUD drawn.
    """
    if frame is None:
        return frame

    h, w, _ = frame.shape
    sidebar_w = min(310, int(w * 0.35))

    # 1. Dark, semi-transparent left sidebar background (88% dark navy opacity)
    sub_roi = frame[:, :sidebar_w]
    dark_panel = np.full_like(sub_roi, (15, 18, 25), dtype=np.uint8)
    cv2.addWeighted(dark_panel, 0.88, sub_roi, 0.12, 0, sub_roi)

    # Sidebar right border line
    cv2.line(frame, (sidebar_w, 0), (sidebar_w, h), (60, 60, 70), 1)

    # Color Palette (BGR)
    CYAN = (255, 230, 0)
    GREEN = (80, 230, 110)
    YELLOW = (0, 215, 255)
    RED = (90, 90, 255)
    WHITE = (245, 245, 245)
    GRAY = (160, 165, 175)

    # Extract Metrics
    hr = hud_data.get("heart_rate")
    rmssd = hud_data.get("rmssd")
    arousal = hud_data.get("arousal", 0.50)
    valence = hud_data.get("valence", 0.0)
    stress_label = hud_data.get("stress_label", "NORMAL")
    emotion_label = hud_data.get("emotion_label", "calm-positive")
    transcript = hud_data.get("transcript", "")
    llm_response = hud_data.get("llm_response", "")
    face_detected = hud_data.get("face_detected", False)

    x_margin = 15
    curr_y = 30

    # A. BRAND TITLE & STATUS
    cv2.putText(frame, "WebPulse", (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    
    status_str = "● FACE" if face_detected else "○ SEARCHING"
    status_col = GREEN if face_detected else YELLOW
    cv2.putText(frame, status_str, (x_margin + 160, curr_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_col, 1)

    curr_y += 15
    cv2.line(frame, (x_margin, curr_y), (sidebar_w - x_margin, curr_y), (50, 55, 65), 1)

    # B. PHYSIOLOGY SECTION
    curr_y += 25
    cv2.putText(frame, "PHYSIOLOGY", (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1)

    curr_y += 24
    hr_str = f"HR:      {hr:.1f} BPM" if hr else "HR:      Calibrating..."
    cv2.putText(frame, hr_str, (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)

    curr_y += 22
    rmssd_str = f"HRV:     {rmssd:.1f} ms" if rmssd else "HRV:     --"
    cv2.putText(frame, rmssd_str, (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)

    curr_y += 22
    cv2.putText(frame, f"Arousal: {arousal:.2f}", (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)

    curr_y += 22
    s_col = GREEN if stress_label == "CALM" else (YELLOW if stress_label == "NORMAL" else RED)
    cv2.putText(frame, f"Stress:  {stress_label}", (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, s_col, 2)

    # C. VOICE & SPEECH SECTION
    curr_y += 30
    cv2.line(frame, (x_margin, curr_y - 12), (sidebar_w - x_margin, curr_y - 12), (50, 55, 65), 1)
    cv2.putText(frame, "VOICE & SPEECH", (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1)

    curr_y += 24
    cv2.putText(frame, f"Valence: {valence:+.2f}", (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)

    curr_y += 22
    disp_speech = transcript if transcript else "(listening...)"
    if len(disp_speech) > 28:
        disp_speech = disp_speech[:25] + "..."
    cv2.putText(frame, f"Speech:  \"{disp_speech}\"", (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, YELLOW, 1)

    # D. FUSED EMOTION SECTION
    curr_y += 30
    cv2.line(frame, (x_margin, curr_y - 12), (sidebar_w - x_margin, curr_y - 12), (50, 55, 65), 1)
    cv2.putText(frame, "EMOTION STATE", (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1)

    curr_y += 25
    cv2.putText(frame, emotion_label.upper(), (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

    # E. LLM COMPANION OUTPUT SECTION
    curr_y += 32
    cv2.line(frame, (x_margin, curr_y - 12), (sidebar_w - x_margin, curr_y - 12), (50, 55, 65), 1)
    cv2.putText(frame, "LLM COMPANION", (x_margin, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1)

    curr_y += 22
    disp_llm = llm_response if llm_response else "Waiting for input..."
    
    # Simple word wrap inside sidebar
    max_char_line = 28
    words = disp_llm.split()
    lines = []
    c_line = ""
    for w_tok in words:
        if len(c_line) + len(w_tok) + 1 <= max_char_line:
            c_line += (" " if c_line else "") + w_tok
        else:
            lines.append(c_line)
            c_line = w_tok
    if c_line:
        lines.append(c_line)

    max_lines = max(1, int((h - curr_y - 10) / 20))
    for i, line_text in enumerate(lines[:max_lines]):
        cv2.putText(frame, line_text, (x_margin, curr_y + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)

    return frame
