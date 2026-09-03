"""Pose landmarks (MediaPipe PoseLandmarker) used as a geometric sanity
check on top of the segmentation models, not a replacement for them. Pose
doesn't know what a collar or a zipper looks like -- what it reliably
knows is roughly where the actual person's body is, independent of
whatever the segmentation model predicts.

Uses an absolute path to the pose model file so detection works
regardless of the caller's current working directory.
"""

import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(TOOL_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "submission", "models", "pose_landmarker_lite.task")

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
VISIBILITY_THRESHOLD = 0.5

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"pose model not found at {MODEL_PATH}")
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
        _landmarker = vision.PoseLandmarker.create_from_options(options)
    return _landmarker


def get_pose_info(image_rgb_array):
    """Returns a dict of geometric facts about the detected person, in
    pixel coordinates matching image_rgb_array's own size, or None if no
    person was confidently detected. Infra failures (missing model file,
    etc.) raise instead of returning None -- only a genuine "no person
    here" result returns None, so callers can't confuse the two."""
    h, w = image_rgb_array.shape[0], image_rgb_array.shape[1]
    landmarker = _get_landmarker()  # raises on infra failure, not caught here

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_array)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks[0]
    l_sh, r_sh, l_hip, r_hip = lm[L_SHOULDER], lm[R_SHOULDER], lm[L_HIP], lm[R_HIP]
    core_landmarks = (l_sh, r_sh, l_hip, r_hip)
    if any(p.visibility < VISIBILITY_THRESHOLD for p in core_landmarks):
        return None

    xs = [p.x * w for p in lm if p.visibility >= VISIBILITY_THRESHOLD]
    ys = [p.y * h for p in lm if p.visibility >= VISIBILITY_THRESHOLD]

    return {
        "shoulder_y": 0.5 * (l_sh.y + r_sh.y) * h,
        "shoulder_center_x": 0.5 * (l_sh.x + r_sh.x) * w,
        "hip_y": 0.5 * (l_hip.y + r_hip.y) * h,
        "hip_center_x": 0.5 * (l_hip.x + r_hip.x) * w,
        # bounding box over every landmark MediaPipe was confident about,
        # for filtering predictions down to "near this actual person"
        "person_bbox": (min(xs), min(ys), max(xs), max(ys)),  # x0, y0, x1, y1
    }
