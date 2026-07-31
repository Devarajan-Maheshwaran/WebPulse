"""Minimal live overlay for the WebPulse camera view."""

import cv2
import numpy as np


def _text(frame, value, x, y, size=0.45, color=(235, 235, 235), weight=1):
    cv2.putText(frame, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, weight, cv2.LINE_AA)


def _wrap(text, width):
    words = (text or "").split()
    lines, line = [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def draw_overlay_hud(frame, hud_data, bvp_history=None):
    """Draw a restrained, readable status rail over the camera frame."""
    if frame is None:
        return frame

    h, w, _ = frame.shape
    rail_w = min(305, max(270, int(w * 0.27)))
    panel = frame[:, :rail_w].copy()
    cv2.rectangle(panel, (0, 0), (rail_w, h), (18, 20, 24), -1)
    cv2.addWeighted(panel, 0.90, frame[:, :rail_w], 0.10, 0, frame[:, :rail_w])
    cv2.line(frame, (rail_w, 0), (rail_w, h), (80, 86, 94), 1)

    white = (238, 238, 238)
    muted = (155, 160, 166)
    accent = (210, 190, 60)  # BGR: restrained warm yellow
    green = (105, 210, 100)
    amber = (50, 190, 235)
    red = (75, 80, 225)

    hr = hud_data.get("heart_rate")
    rmssd = hud_data.get("rmssd")
    arousal = float(hud_data.get("arousal", 0.5) or 0.5)
    valence = float(hud_data.get("valence", 0.0) or 0.0)
    stress = hud_data.get("stress_label", "NORMAL")
    emotion = hud_data.get("emotion_label", "calm-positive")
    quality = hud_data.get("quality_status", "GOOD")
    roi_count = hud_data.get("roi_count", 0)
    provider = hud_data.get("execution_provider", "Unknown")
    face = hud_data.get("face_detected", False)
    transcript = hud_data.get("transcript", "")
    response = hud_data.get("llm_response", "")

    quality_color = green if quality == "GOOD" else amber if quality == "POOR_LIGHTING" else red
    y = 32
    _text(frame, "WebPulse", 18, y, 0.72, white, 2)
    _text(frame, "LIVE", rail_w - 58, y, 0.38, green if face else amber, 1)
    y += 18
    cv2.line(frame, (18, y), (rail_w - 18, y), (70, 74, 80), 1)

    y += 28
    _text(frame, "SIGNAL", 18, y, 0.34, muted, 1)
    _text(frame, quality.replace("_", " "), 18, y + 23, 0.52, quality_color, 2)
    _text(frame, f"ROI coverage {roi_count}/3", 18, y + 43, 0.34, muted, 1)
    y += 68

    _text(frame, "HEART RATE", 18, y, 0.34, muted, 1)
    _text(frame, f"{hr:.1f} BPM" if hr is not None else "Calibrating", 18, y + 27, 0.62, white, 2)
    _text(frame, "HRV / RMSSD", 158, y, 0.34, muted, 1)
    _text(frame, f"{rmssd:.1f} ms" if rmssd is not None else "--", 158, y + 27, 0.62, white, 2)
    y += 68

    _text(frame, "STATE", 18, y, 0.34, muted, 1)
    _text(frame, stress, 18, y + 24, 0.52, white, 2)
    _text(frame, f"Arousal {arousal:.2f}  |  Voice {valence:+.2f}", 18, y + 46, 0.39, muted, 1)
    y += 70

    _text(frame, "PULSE", 18, y, 0.34, muted, 1)
    y += 10
    chart_x, chart_y = 18, y + 8
    chart_w, chart_h = rail_w - 36, 52
    cv2.rectangle(frame, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (30, 33, 38), -1)
    cv2.line(frame, (chart_x, chart_y + chart_h // 2),
             (chart_x + chart_w, chart_y + chart_h // 2), (60, 64, 70), 1)
    if bvp_history and len(bvp_history) > 4:
        values = np.asarray(bvp_history[-70:], dtype=np.float32)
        values = (values - values.mean()) / (values.std() + 1e-5)
        points = []
        for i, value in enumerate(values):
            px = chart_x + int(i * chart_w / max(1, len(values) - 1))
            py = chart_y + chart_h // 2 - int(np.clip(value, -2.0, 2.0) * 10)
            points.append((px, py))
        cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, accent, 1, cv2.LINE_AA)
    y = chart_y + chart_h + 32

    _text(frame, emotion.replace("-", " / ").upper(), 18, y, 0.46, accent, 1)
    y += 30
    _text(frame, "VOICE", 18, y, 0.34, muted, 1)
    voice = transcript if transcript else "Listening..."
    for line in _wrap(voice, 38)[:2]:
        y += 18
        _text(frame, line, 18, y, 0.39, white, 1)
    y += 27
    _text(frame, "COMPANION", 18, y, 0.34, muted, 1)
    for line in _wrap(response or "Waiting for speech", 38)[:4]:
        y += 18
        _text(frame, line, 18, y, 0.39, white, 1)

    provider_label = "DirectML GPU" if provider == "DmlExecutionProvider" else provider
    _text(frame, f"EfficientPhys  |  {provider_label}", 18, h - 18, 0.33, muted, 1)
    return frame
