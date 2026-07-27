"""
rppg/hud.py — Full HUD (Heads-Up Display) Visual Overlay Renderer for WebPulse.

Renders all real-time multimodal metrics onto the video window interface:
  1. Header Bar: "WebPulse — Multimodal Emotion Companion"
  2. Biometrics Panel: Heart Rate (BPM), HRV (RMSSD), Stress State, Arousal score
  3. Audio & Speech Panel: Voice Valence score, User Speech Transcript
  4. Fused Emotion Panel: Circumplex Quadrant & Description
  5. LLM Response Output Box: Live companion response text
"""

import cv2
import numpy as np


def draw_overlay_hud(frame, hud_data):
    """
    Draw clean, high-contrast, semi-transparent HUD panels on the video frame.
    
    Args:
        frame (ndarray): OpenCV BGR frame.
        hud_data (dict): Current metrics state.
        
    Returns:
        ndarray: Frame with HUD overlay drawn.
    """
    if frame is None:
        return frame

    h, w, _ = frame.shape
    overlay = frame.copy()

    # Colors (BGR)
    BG_DARK = (20, 20, 25)
    CYAN = (255, 230, 0)
    GREEN = (80, 220, 100)
    YELLOW = (0, 215, 255)
    RED = (80, 80, 255)
    WHITE = (245, 245, 245)
    GRAY = (170, 170, 170)
    ACCENT_PURPLE = (230, 120, 180)

    # 1. TOP HEADER BANNER
    header_height = 40
    cv2.rectangle(overlay, (0, 0), (w, header_height), BG_DARK, -1)
    cv2.putText(frame, "WebPulse -- Live Multimodal Emotion Companion", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)

    # Status indicator
    face_detected = hud_data.get("face_detected", False)
    status_text = "FACE DETECTED" if face_detected else "SEARCHING FOR FACE"
    status_color = GREEN if face_detected else YELLOW
    cv2.putText(frame, status_text, (w - 220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # Extract metrics
    hr = hud_data.get("heart_rate")
    rmssd = hud_data.get("rmssd")
    arousal = hud_data.get("arousal", 0.5)
    valence = hud_data.get("valence", 0.0)
    stress_label = hud_data.get("stress_label", "NORMAL")
    emotion_label = hud_data.get("emotion_label", "calm-positive")
    emotion_desc = hud_data.get("emotion_desc", "Low arousal & positive valence")
    transcript = hud_data.get("transcript", "")
    llm_response = hud_data.get("llm_response", "")

    # 2. TOP-LEFT PANEL: BIOMETRICS & PHYSIOLOGY
    p1_x, p1_y, p1_w, p1_h = 15, 50, 260, 130
    cv2.rectangle(overlay, (p1_x, p1_y), (p1_x + p1_w, p1_y + p1_h), BG_DARK, -1)
    cv2.rectangle(frame, (p1_x, p1_y), (p1_x + p1_w, p1_y + p1_h), CYAN, 1)

    cv2.putText(frame, "PHYSIOLOGY & BIOMETRICS", (p1_x + 10, p1_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)

    hr_str = f"Heart Rate:  {hr:.1f} BPM" if hr else "Heart Rate:  Calibrating..."
    rmssd_str = f"HRV (RMSSD): {rmssd:.1f} ms" if rmssd else "HRV (RMSSD): Calibrating..."
    arousal_str = f"Arousal:     {arousal:.2f} / 1.00"

    cv2.putText(frame, hr_str, (p1_x + 10, p1_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)
    cv2.putText(frame, rmssd_str, (p1_x + 10, p1_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)
    cv2.putText(frame, arousal_str, (p1_x + 10, p1_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)

    # Stress Badge
    s_color = GREEN if stress_label == "CALM" else (YELLOW if stress_label == "NORMAL" else RED)
    cv2.putText(frame, f"Stress: {stress_label}", (p1_x + 10, p1_y + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.48, s_color, 2)

    # 3. TOP-RIGHT PANEL: AUDIO & VOICE SPEECH
    p2_w, p2_h = 320, 130
    p2_x, p2_y = w - p2_w - 15, 50
    cv2.rectangle(overlay, (p2_x, p2_y), (p2_x + p2_w, p2_y + p2_h), BG_DARK, -1)
    cv2.rectangle(frame, (p2_x, p2_y), (p2_x + p2_w, p2_y + p2_h), ACCENT_PURPLE, 1)

    cv2.putText(frame, "VOICE & SPEECH ANALYTICS", (p2_x + 10, p2_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_PURPLE, 1)

    val_str = f"Voice Valence: {valence:+.2f} [-1 to +1]"
    cv2.putText(frame, val_str, (p2_x + 10, p2_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)

    # Speech transcript
    disp_transcript = transcript if transcript else "(Listening for speech...)"
    if len(disp_transcript) > 36:
        disp_transcript = disp_transcript[:33] + "..."
    cv2.putText(frame, "Speech Transcript:", (p2_x + 10, p2_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)
    cv2.putText(frame, f"\"{disp_transcript}\"", (p2_x + 10, p2_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)

    # 4. BOTTOM-LEFT PANEL: FUSED EMOTION STATE
    p3_x, p3_y, p3_w, p3_h = 15, h - 130, 320, 115
    cv2.rectangle(overlay, (p3_x, p3_y), (p3_x + p3_w, p3_y + p3_h), BG_DARK, -1)
    cv2.rectangle(frame, (p3_x, p3_y), (p3_x + p3_w, p3_y + p3_h), GREEN, 1)

    cv2.putText(frame, "MULTIMODAL FUSED EMOTION", (p3_x + 10, p3_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
    cv2.putText(frame, f"State: {emotion_label.upper()}", (p3_x + 10, p3_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

    # Wrap description
    desc_str = emotion_desc
    if len(desc_str) > 40:
        cv2.putText(frame, desc_str[:38], (p3_x + 10, p3_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
        cv2.putText(frame, desc_str[38:], (p3_x + 10, p3_y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
    else:
        cv2.putText(frame, desc_str, (p3_x + 10, p3_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)

    # 5. BOTTOM-RIGHT PANEL: LLM RESPONSE COMPANION OUTPUT
    p4_w, p4_h = w - p3_w - 45, 115
    p4_x, p4_y = p3_x + p3_w + 15, h - 130
    cv2.rectangle(overlay, (p4_x, p4_y), (p4_x + p4_w, p4_y + p4_h), BG_DARK, -1)
    cv2.rectangle(frame, (p4_x, p4_y), (p4_x + p4_w, p4_y + p4_h), YELLOW, 1)

    cv2.putText(frame, "WEBPULSE LLM COMPANION OUTPUT", (p4_x + 10, p4_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)

    disp_llm = llm_response if llm_response else "Initializing LLM companion response..."
    # Line wrapping for LLM output (up to 3 lines)
    words = disp_llm.split()
    lines = []
    curr_line = ""
    max_chars = max(20, int(p4_w / 9))

    for word in words:
        if len(curr_line) + len(word) + 1 <= max_chars:
            curr_line += (" " if curr_line else "") + word
        else:
            lines.append(curr_line)
            curr_line = word
    if curr_line:
        lines.append(curr_line)

    for i, l_str in enumerate(lines[:3]):
        y_pos = p4_y + 45 + (i * 22)
        cv2.putText(frame, l_str, (p4_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    # Blend transparent overlays (0.75 frame + 0.25 overlay)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return frame
