"""
rppg/capture.py — Webcam capture and face/ROI landmark detection.

Implements FR-1: Video Capture and Face/ROI Detection.
Primary: MediaPipe Face Mesh landmark detection for facial ROI (forehead & cheeks).
Fallback: OpenCV Haar Cascade face detector if MediaPipe is unavailable or fails.
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    try:
        mp_face_mesh = mp.solutions.face_mesh
    except AttributeError:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False
    mp_face_mesh = None


class FaceROICapturer:
    """
    Handles webcam video capture and facial Region of Interest (ROI) extraction.
    """

    def __init__(self, camera_index=0, width=640, height=480, fallback_haar=True):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

        # MediaPipe initialization
        self.face_mesh = None
        if HAS_MEDIAPIPE and mp_face_mesh is not None:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        # Haar Cascade Fallback initialization
        self.haar_cascade = None
        if fallback_haar:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)

    def start(self):
        """Initialize the video capture stream."""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)
        if not self.cap.isOpened():
            # Try default backend without CAP_DSHOW
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
        if self.face_mesh is not None:
            self.face_mesh.close()

    def get_frame(self):
        """Read a frame from webcam."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def extract_roi(self, frame):
        """
        Detect face and extract ROI bounding box (forehead / upper face region).
        
        Returns:
            roi_crop: Crop of ROI region from frame (or None if lost)
            roi_box: (x, y, w, h) bounding box
            detection_method: 'mediapipe', 'haar', or None
        """
        if frame is None:
            return None, None, None

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Try MediaPipe Face Mesh
        if self.face_mesh is not None:
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # MediaPipe Forehead Landmark Indices: 10, 67, 109, 338, 297, 332
                forehead_indices = [10, 67, 109, 338, 297, 332, 21, 251]
                xs = [int(landmarks[idx].x * w) for idx in forehead_indices]
                ys = [int(landmarks[idx].y * h) for idx in forehead_indices]
                
                min_x, max_x = max(0, min(xs)), min(w, max(xs))
                min_y, max_y = max(0, min(ys)), min(h, max(ys))
                
                roi_w = max_x - min_x
                roi_h = max_y - min_y

                if roi_w > 10 and roi_h > 10:
                    roi_crop = frame[min_y:max_y, min_x:max_x]
                    return roi_crop, (min_x, min_y, roi_w, roi_h), 'mediapipe'

        # 2. Fallback: Haar Cascade
        if self.haar_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                # Forehead estimate: upper 20-40% of the face rectangle
                roi_x = fx + int(fw * 0.25)
                roi_y = fy + int(fh * 0.10)
                roi_w = int(fw * 0.50)
                roi_h = int(fh * 0.20)

                roi_x = max(0, min(w - 1, roi_x))
                roi_y = max(0, min(h - 1, roi_y))
                roi_w = min(w - roi_x, roi_w)
                roi_h = min(h - roi_y, roi_h)

                if roi_w > 10 and roi_h > 10:
                    roi_crop = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    return roi_crop, (roi_x, roi_y, roi_w, roi_h), 'haar'

        return None, None, None
