"""Shoulder-line detection (MediaPipe PoseLandmarker, Apache 2.0) used by
placement.py to anchor the "chest" zone instead of the mask's bounding box,
which breaks on raised arms. Failures here (no person, low confidence)
degrade to None rather than raising, so the caller can fall back to
bbox-based geometry."""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/pose_landmarker_lite.task"
L_SHOULDER, R_SHOULDER = 11, 12
VISIBILITY_THRESHOLD = 0.5

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
        _landmarker = vision.PoseLandmarker.create_from_options(options)
    return _landmarker


def get_shoulder_y(image_rgb_array):
    """Shoulder line's y-coordinate in pixels, or None if not confidently
    detected."""
    try:
        landmarker = _get_landmarker()
        h = image_rgb_array.shape[0]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_array)
        result = landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        l_sh, r_sh = lm[L_SHOULDER], lm[R_SHOULDER]
        if l_sh.visibility < VISIBILITY_THRESHOLD or r_sh.visibility < VISIBILITY_THRESHOLD:
            return None
        return 0.5 * (l_sh.y + r_sh.y) * h
    except Exception:
        return None
