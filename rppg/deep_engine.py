"""EfficientPhys-only deep rPPG inference with DirectML-first ONNX Runtime."""

import collections
import os
import time

import cv2
import numpy as np

from fusion.wesad_classifier import WESADHRVClassifier
from fusion.emotion import StressStateSmoother
from rppg.enhancement import ROIEnhancer, compute_bvp_snr
from rppg.hrv import (
    ArousalSmoother,
    HRSmoother,
    compute_rmssd,
    estimate_heart_rate_fft,
    find_pulse_peaks,
    map_hrv_to_arousal,
)


class DeepRPPGEngine:
    """Run the rPPG-Toolbox EfficientPhys ONNX model on each usable ROI."""

    def __init__(self, onnx_path="weights/efficientphys.onnx", frame_depth=10,
                 img_size=72, buffer_seconds=15, fps=30):
        self.onnx_path = onnx_path
        self.frame_depth = frame_depth
        self.img_size = img_size
        self.fps = fps
        self.buffer_size = int(buffer_seconds * fps)
        self.frame_queues = {
            name: collections.deque(maxlen=frame_depth + 1)
            for name in ("forehead", "left_cheek", "right_cheek")
        }
        self.bvp_buffer = collections.deque(maxlen=self.buffer_size)
        self.session = None
        self.input_name = None
        self.output_name = None
        self.active_provider = "Unavailable"
        self.backend_type = "classical_fallback"
        self._last_inference_time = 0.0
        self._min_inference_interval = 0.25
        self.roi_enhancer = ROIEnhancer()
        self.hr_smoother = HRSmoother(alpha=0.2)
        self.arousal_smoother = ArousalSmoother(alpha=0.35)
        self.wesad_classifier = WESADHRVClassifier()
        self.stress_smoother = StressStateSmoother()
        self._init_session()

    def _init_session(self):
        if not os.path.exists(self.onnx_path):
            print(f"[Deep rPPG] EfficientPhys model not found: {self.onnx_path}")
            return
        try:
            import onnxruntime as ort
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
            active = self.session.get_providers()
            self.active_provider = active[0] if active else "Unavailable"
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.backend_type = "efficientphys_onnx"
            print(f"[Deep rPPG] ONNX providers: {active}")
            print("[Deep rPPG] EfficientPhys execution: "
                  f"{'GPU (DirectML)' if self.active_provider == 'DmlExecutionProvider' else 'CPU fallback'}")
        except Exception as exc:
            print(f"[Deep rPPG] EfficientPhys ONNX unavailable: {exc}")

    def preprocess_roi(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        enhanced = self.roi_enhancer.enhance(roi_bgr)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))

    def _infer(self, frames):
        if self.session is None:
            return None
        # EfficientPhys expects T+1 raw frames and computes torch.diff internally.
        model_input = np.asarray(frames, dtype=np.float32)
        try:
            output = self.session.run([self.output_name], {self.input_name: model_input})[0]
            return np.asarray(output).reshape(-1)
        except Exception as exc:
            print(f"[Deep rPPG] EfficientPhys inference failed: {exc}")
            return None

    def process_frame(self, rois_dict, quality_meta=None):
        quality_meta = quality_meta or {"is_low_light": True, "mean_brightness": 0.0}
        if isinstance(rois_dict, np.ndarray):
            rois_dict = {"forehead": {"crop": rois_dict, "status": "USABLE"}}
        rois_dict = rois_dict or {}
        usable = []
        for name, queue in self.frame_queues.items():
            info = rois_dict.get(name, {})
            if info.get("status") == "USABLE":
                frame = self.preprocess_roi(info.get("crop"))
                if frame is not None:
                    queue.append(frame)
                    usable.append(name)

        if not usable:
            return self._result("NO_VALID_ROI", quality_meta, confidence_low=True)

        now = time.time()
        ready = [name for name in usable if len(self.frame_queues[name]) == self.frame_depth + 1]
        if ready and now - self._last_inference_time >= self._min_inference_interval:
            predictions = {}
            for name in ready:
                result = self._infer(list(self.frame_queues[name]))
                if result is not None and result.size:
                    predictions[name] = result
            if predictions:
                # Equal fusion keeps cheeks useful when the forehead is occluded.
                fused = np.mean(list(predictions.values()), axis=0)
                new_count = max(1, int(self.fps * self._min_inference_interval))
                self.bvp_buffer.extend(fused[-new_count:].tolist())
                self._last_inference_time = now

        signal = np.asarray(self.bvp_buffer, dtype=np.float64)
        snr_db, snr_ok = compute_bvp_snr(signal, self.fps)
        low_light = bool(quality_meta.get("is_low_light", False))
        if low_light and quality_meta.get("mean_brightness", 0) < 24:
            quality = "POOR_LIGHTING"
        elif len(signal) >= int(self.fps * 4) and not snr_ok:
            quality = "WEAK_SIGNAL"
        else:
            quality = "GOOD"

        hr = self.hr_smoother.last_smoothed or 70.0
        rmssd = 35.0
        arousal = self.arousal_smoother.last_arousal or 0.5
        stress = "NORMAL"
        if len(signal) >= int(self.fps * 4):
            hr = self.hr_smoother.update(estimate_heart_rate_fft(signal, self.fps))
            rmssd = compute_rmssd(find_pulse_peaks(signal, self.fps), self.fps)
            prediction = self.wesad_classifier.predict(rmssd, heart_rate=hr)
            stress, temporal_arousal = self.stress_smoother.update(
                prediction.get("stress_state", "NORMAL"),
                prediction.get("arousal_score", map_hrv_to_arousal(rmssd)),
            )
            arousal = self.arousal_smoother.update(temporal_arousal)
        return self._result(quality, quality_meta, confidence_low=quality != "GOOD", hr=hr,
                            rmssd=rmssd, arousal=arousal, stress=stress, snr_db=snr_db)

    def _result(self, quality, meta, confidence_low, hr=None, rmssd=None, arousal=None,
                stress="NORMAL", snr_db=0.0):
        return {
            "heart_rate": hr if hr is not None else (self.hr_smoother.last_smoothed or 70.0),
            "rmssd": rmssd if rmssd is not None else 35.0,
            "arousal_score": arousal if arousal is not None else (self.arousal_smoother.last_arousal or 0.5),
            "stress_state": stress,
            "predicted_bvp": float(self.bvp_buffer[-1]) if self.bvp_buffer else 0.0,
            "backend": self.backend_type,
            "execution_provider": self.active_provider,
            "face_detected": quality != "NO_VALID_ROI",
            "quality_status": quality,
            "snr_db": round(float(snr_db), 1),
            "is_low_light": bool(meta.get("is_low_light", False)),
            "mean_brightness": meta.get("mean_brightness", 0.0),
            "confidence_low": confidence_low,
        }
