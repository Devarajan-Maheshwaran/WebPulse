import os
import cv2
import numpy as np

HAS_MEDIAPIPE = False
FaceLandmarker = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    HAS_MEDIAPIPE = True
except Exception as e:
    HAS_MEDIAPIPE = False

from rppg.enhancement import ROIEnhancer


class FaceROICapturer:
    """
    Handles 720p webcam video capture and facial Region of Interest (ROI) extraction
    using MediaPipe Tasks FaceLandmarker with CLAHE low-light enhancement.
    """

    def __init__(self, camera_index=0, width=1280, height=720, fallback_haar=True):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.roi_enhancer = ROIEnhancer(clip_limit=2.2, tile_grid_size=(8, 8), min_brightness=32.0)

        # MediaPipe Tasks initialization
        self.landmarker = None
        self.detector_name = "NONE"

        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "face_landmarker.task")

        if HAS_MEDIAPIPE and os.path.exists(model_path):
            try:
                options = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.detector_name = "MEDIAPIPE"
                print(f"[Face ROI Capturer SUCCESS] Loaded MediaPipe FaceLandmarker model ({model_path}).")
            except Exception as ex:
                print(f"[Face ROI Capturer WARNING] Could not load MediaPipe FaceLandmarker: {ex}")
        elif not os.path.exists(model_path):
            print(f"[Face ROI Capturer WARNING] Model file missing: {model_path}")

        # Haar Cascade Fallback initialization
        self.haar_cascade = None
        if self.landmarker is None and fallback_haar:
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self.haar_cascade = cascade
                self.detector_name = "HAAR_FALLBACK"
                print(f"[Face ROI Capturer FALLBACK] MediaPipe unavailable. Using OpenCV Haar Cascade face detector.")
            else:
                print(f"[Face ROI Capturer WARNING] Could not load Haar cascade from {cascade_path}")

    def start(self):
        """Initialize the video capture stream."""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True
        return False

    def stop(self):
        """Release video capture stream and MediaPipe resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass

    def get_frame(self):
        """Read a frame from webcam."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def check_roi_occlusion(self, raw_crop):
        """
        Lightweight occlusion & visibility check for an ROI crop.
        
        Checks:
        - Crop size
        - Mean brightness (avoids extreme dark shadow or extreme bright glare)
        - Intensity standard deviation (avoids uniform hair/cloth cover)
        - Skin-tone BGR color distribution (Red channel dominant over Blue)
        
        Returns:
            is_occluded (bool): True if ROI is occluded/unreliable, False if USABLE.
            status (str): 'USABLE' or 'OCCLUDED'
            mean_b (float): Mean brightness
            std_b (float): Intensity standard deviation
        """
        if raw_crop is None or raw_crop.size == 0 or raw_crop.shape[0] < 5 or raw_crop.shape[1] < 5:
            return True, "OCCLUDED", 0.0, 0.0

        mean_b = float(np.mean(raw_crop))
        std_b = float(np.std(raw_crop))

        # BGR channel means
        b_mean, g_mean, r_mean = np.mean(raw_crop, axis=(0, 1))

        # Occlusion heuristics:
        # 1. Dark shadow / hair cover (mean_b < 20) or extreme glare (mean_b > 245)
        # 2. Low texture variance (std_b < 3.5)
        # 3. Non-skin color balance (Red channel must be higher than Blue channel for skin)
        is_dark_or_glare = mean_b < 20.0 or mean_b > 245.0
        is_low_texture = std_b < 3.5
        is_not_skin_color = r_mean <= (b_mean + 2.0)

        is_occluded = is_dark_or_glare or is_low_texture or is_not_skin_color
        status = "OCCLUDED" if is_occluded else "USABLE"

        return is_occluded, status, round(mean_b, 1), round(std_b, 1)

    def extract_multi_roi(self, frame):
        """
        Extracts THREE separate facial ROIs using MediaPipe FaceLandmarker:
        1. Forehead ROI
        2. Left Cheek ROI (Left Malar)
        3. Right Cheek ROI (Right Malar)

        Performs occlusion/visibility check on each ROI independently.

        Returns:
            rois (dict): Dictionary of ROIs with crops, boxes, statuses, and metadata.
            full_face_box (tuple): (x, y, w, h) bounding box of full face.
            method (str): 'mediapipe', 'haar', or None.
        """
        if frame is None:
            return None, None, None

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Default multi-ROI result structure
        rois = {
            "forehead": {"crop": None, "box": None, "status": "OCCLUDED", "mean_b": 0.0, "std_b": 0.0},
            "left_cheek": {"crop": None, "box": None, "status": "OCCLUDED", "mean_b": 0.0, "std_b": 0.0},
            "right_cheek": {"crop": None, "box": None, "status": "OCCLUDED", "mean_b": 0.0, "std_b": 0.0},
        }

        full_face_box = None

        # 1. Try MediaPipe Tasks FaceLandmarker
        if self.landmarker is not None:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.landmarker.detect(mp_image)

                if result.face_landmarks and len(result.face_landmarks) > 0:
                    landmarks = result.face_landmarks[0]

                    # Full face bounding box from all landmarks
                    all_xs = [int(lm.x * w) for lm in landmarks]
                    all_ys = [int(lm.y * h) for lm in landmarks]
                    fx_min, fx_max = max(0, min(all_xs)), min(w, max(all_xs))
                    fy_min, fy_max = max(0, min(all_ys)), min(h, max(all_ys))
                    full_face_box = (fx_min, fy_min, fx_max - fx_min, fy_max - fy_min)

                    # ROI Landmark Index Definitions
                    roi_landmark_map = {
                        "forehead": [10, 67, 109, 338, 297, 332, 21, 251, 103, 104, 108, 151, 337, 336, 9],
                        "left_cheek": [117, 118, 119, 120, 121, 147, 187, 205, 206, 207, 50, 123],
                        "right_cheek": [346, 347, 348, 349, 350, 376, 411, 425, 426, 427, 280, 352]
                    }

                    for roi_name, indices in roi_landmark_map.items():
                        xs = [int(landmarks[idx].x * w) for idx in indices if idx < len(landmarks)]
                        ys = [int(landmarks[idx].y * h) for idx in indices if idx < len(landmarks)]

                        if not xs or not ys:
                            continue

                        min_x, max_x = max(0, min(xs)), min(w, max(xs))
                        min_y, max_y = max(0, min(ys)), min(h, max(ys))
                        # Enlarge focused patches moderately so low-light skin pixels
                        # are not lost, while keeping eyes/mouth out of the signal.
                        pad_x = int((max_x - min_x) * 0.22)
                        pad_y = int((max_y - min_y) * (0.28 if roi_name == "forehead" else 0.20))
                        min_x, max_x = max(0, min_x - pad_x), min(w, max_x + pad_x)
                        min_y, max_y = max(0, min_y - pad_y), min(h, max_y + pad_y)
                        rw, rh = max_x - min_x, max_y - min_y

                        if rw > 5 and rh > 5:
                            raw_crop = frame[min_y:max_y, min_x:max_x]
                            is_occ, status, mean_b, std_b = self.check_roi_occlusion(raw_crop)
                            rois[roi_name] = {
                                # Keep the capture contract raw. The deep
                                # engine applies enhancement once, immediately
                                # before model normalization, avoiding double
                                # temporal contrast manipulation.
                                "crop": raw_crop,
                                "box": (min_x, min_y, rw, rh),
                                "status": status,
                                "mean_b": mean_b,
                                "std_b": std_b
                            }

                    return rois, full_face_box, 'mediapipe'
            except Exception as e:
                pass

        # 2. Fallback: Haar Cascade
        if self.haar_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                full_face_box = (fx, fy, fw, fh)

                # Heuristic ROI patches from Haar face box
                # Forehead patch
                fh_x, fh_y = fx + int(fw * 0.25), fy + int(fh * 0.10)
                fh_w, fh_h = int(fw * 0.50), int(fh * 0.20)
                # Left cheek patch
                lc_x, lc_y = fx + int(fw * 0.15), fy + int(fh * 0.55)
                lc_w, lc_h = int(fw * 0.30), int(fh * 0.25)
                # Right cheek patch
                rc_x, rc_y = fx + int(fw * 0.55), fy + int(fh * 0.55)
                rc_w, rc_h = int(fw * 0.30), int(fh * 0.25)

                patches = {"forehead": (fh_x, fh_y, fh_w, fh_h),
                           "left_cheek": (lc_x, lc_y, lc_w, lc_h),
                           "right_cheek": (rc_x, rc_y, rc_w, rc_h)}

                for rname, (px, py, pw, ph) in patches.items():
                    px = max(0, min(w - 1, px))
                    py = max(0, min(h - 1, py))
                    pw = min(w - px, pw)
                    ph = min(h - py, ph)
                    if pw > 5 and ph > 5:
                        rcrop = frame[py:py+ph, px:px+pw]
                        is_occ, status, mean_b, std_b = self.check_roi_occlusion(rcrop)
                        rois[rname] = {
                            "crop": rcrop,
                            "box": (px, py, pw, ph),
                            "status": status,
                            "mean_b": mean_b,
                            "std_b": std_b
                        }

                return rois, full_face_box, 'haar'

        return rois, None, None

    def extract_roi(self, frame):
        """
        Backward-compatible single ROI extractor wrapper.
        Extracts multi-ROIs and returns best usable crop (preferring forehead).
        """
        rois, full_face_box, method = self.extract_multi_roi(frame)
        if rois is None or full_face_box is None:
            return None, None, None, None, {"is_low_light": True, "mean_brightness": 0.0}

        # Select best usable ROI (forehead -> left_cheek -> right_cheek)
        best_name = None
        for name in ["forehead", "left_cheek", "right_cheek"]:
            if rois[name]["status"] == "USABLE":
                best_name = name
                break

        if best_name is None:
            best_name = "forehead"  # Fallback to forehead if all occluded

        best_roi = rois[best_name]
        is_low_light, mean_b = self.roi_enhancer.check_brightness(best_roi["crop"])
        quality_meta = {"is_low_light": is_low_light, "mean_brightness": round(mean_b, 1)}

        return best_roi["crop"], full_face_box, best_roi["box"], method, quality_meta
