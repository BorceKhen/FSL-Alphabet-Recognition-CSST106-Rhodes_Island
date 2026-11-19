"""Hand landmark extraction and 63-dim geometric normalization using MediaPipe Tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.config import (
    FEATURE_DIM,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODELS_DIR,
)

HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"

# Standard 21-keypoint MediaPipe hand skeleton connections
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
)


def apply_clahe(bgr_image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE adaptive contrast normalization on the L channel in LAB space."""
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge((l_eq, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def normalize_landmarks(landmarks: Sequence | np.ndarray) -> np.ndarray:
    """Translate 21 (x, y, z) landmarks to wrist origin and scale by bounding box size.

    Returns a 63-dimensional float32 vector invariant to translation and scale.
    """
    if isinstance(landmarks, np.ndarray):
        coords = landmarks.astype(np.float32).reshape(21, 3)
    else:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # Translate relative to wrist (Landmark 0)
    coords = coords - coords[0]

    # Scale by hand bounding-box extent (max span across x, y)
    x_span = float(coords[:, 0].max() - coords[:, 0].min())
    y_span = float(coords[:, 1].max() - coords[:, 1].min())
    scale = max(x_span, y_span, 1e-6)
    coords = coords / scale

    vec = coords.flatten()
    assert vec.shape == (FEATURE_DIM,), f"Expected {FEATURE_DIM} features, got {vec.shape}"
    return vec


def draw_hand_skeleton(bgr_image: np.ndarray, landmarks: Sequence) -> np.ndarray:
    """Draw 21 hand landmarks and connections onto a BGR frame copy."""
    annotated = bgr_image.copy()
    h, w = annotated.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(annotated, points[start_idx], points[end_idx], (0, 220, 120), 2, cv2.LINE_AA)

    for idx, (px, py) in enumerate(points):
        color = (0, 140, 255) if idx == 0 else (255, 255, 255)
        radius = 5 if idx == 0 else 4
        cv2.circle(annotated, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (px, py), radius, (20, 20, 20), 1, cv2.LINE_AA)

    return annotated


class HandLandmarkExtractor:
    """Wrapper around MediaPipe HandLandmarker for IMAGE and VIDEO modes."""

    def __init__(
        self,
        model_path: Path = HAND_LANDMARKER_PATH,
        running_mode: str = "IMAGE",
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe hand_landmarker.task not found at {model_path}")

        mode = (
            mp_vision.RunningMode.VIDEO
            if running_mode.upper() == "VIDEO"
            else mp_vision.RunningMode.IMAGE
        )
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mode,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mode = mode
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def extract(
        self,
        bgr_image: np.ndarray,
        timestamp_ms: Optional[int] = None,
        use_clahe: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[Sequence]]:
        """Detect a single hand and return (63-dim normalized vector, raw landmarks)."""
        proc = apply_clahe(bgr_image) if use_clahe else bgr_image
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._mode == mp_vision.RunningMode.VIDEO:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required in VIDEO running mode")
            result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None, None

        landmarks = result.hand_landmarks[0]
        return normalize_landmarks(landmarks), landmarks

    def extract_with_fallback(
        self,
        bgr_image: np.ndarray,
        use_clahe: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[Sequence], np.ndarray]:
        """Try standard extraction, CLAHE, and 20% border padding for tight crops/low contrast.

        Returns (63-dim vector, landmarks, processed_image_used).
        """
        base = apply_clahe(bgr_image) if use_clahe else bgr_image
        vec, lms = self.extract(base, use_clahe=False)
        if vec is not None:
            return vec, lms, base

        if not use_clahe:
            clahe_img = apply_clahe(bgr_image)
            vec, lms = self.extract(clahe_img, use_clahe=False)
            if vec is not None:
                return vec, lms, clahe_img

        h, w = base.shape[:2]
        pad_y, pad_x = int(h * 0.2), int(w * 0.2)
        padded = cv2.copyMakeBorder(base, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        vec, lms = self.extract(padded, use_clahe=False)
        if vec is not None:
            return vec, lms, padded

        padded_clahe = apply_clahe(padded)
        vec, lms = self.extract(padded_clahe, use_clahe=False)
        if vec is not None:
            return vec, lms, padded_clahe

        return None, None, base

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
