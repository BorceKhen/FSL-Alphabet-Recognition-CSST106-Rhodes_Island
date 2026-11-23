"""Hand & Holistic landmark extraction and geometric normalization using MediaPipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.config import (
    FEATURE_DIM,
    FSL105_FEATURE_DIM,
    FSL105_SEQ_LEN,
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

# 15 upper-body MediaPipe Pose indices: nose, eyes, ears, mouth, shoulders, elbows, wrists, hips
POSE_KEYPOINT_INDICES: Tuple[int, ...] = (0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24)

# Connections within the 15-point upper-body subset
UPPER_BODY_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (7, 8), (7, 9), (9, 11), (8, 10), (10, 12),
    (7, 13), (8, 14), (13, 14),
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
)


def _is_low_light_or_low_contrast(bgr_image: np.ndarray) -> bool:
    """Fast downsampled check so auto-CLAHE only fires in genuinely dark or washed-out scenes."""
    small = bgr_image[::8, ::8]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    return mean_val < 85.0 or std_val < 32.0


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

    def extract_realtime(
        self,
        bgr_image: np.ndarray,
        timestamp_ms: Optional[int] = None,
        use_clahe: bool = False,
        auto_clahe_fallback: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[Sequence], bool]:
        """Extract hand landmarks with optional automatic CLAHE fallback on low-light miss."""
        vec, landmarks = self.extract(bgr_image, timestamp_ms=timestamp_ms, use_clahe=use_clahe)
        if vec is not None:
            return vec, landmarks, use_clahe

        if auto_clahe_fallback and not use_clahe and _is_low_light_or_low_contrast(bgr_image):
            vec_c, landmarks_c = self.extract(bgr_image, timestamp_ms=timestamp_ms, use_clahe=True)
            if vec_c is not None:
                return vec_c, landmarks_c, True

        return None, None, False

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


def _landmarks_to_array(landmarks: Any, expected_count: int) -> np.ndarray:
    """Convert MediaPipe LandmarkList, sequence, or ndarray into (expected_count, 3) float32."""
    if isinstance(landmarks, np.ndarray):
        arr = landmarks.astype(np.float32)
        if arr.ndim == 2 and arr.shape[0] == expected_count and arr.shape[1] >= 3:
            return arr[:, :3]
        raise ValueError(f"Expected ({expected_count}, 3) array, got {arr.shape}")
    lm_list = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    return np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)


def normalize_holistic_landmarks(
    pose_lms: Optional[Any],
    left_hand_lms: Optional[Any],
    right_hand_lms: Optional[Any],
) -> np.ndarray:
    """Normalize upper-body pose + left/right hand landmarks into a 293-dim float32 vector.

    Layout:
      - pose_norm (45): 15 upper-body keypoints translated by shoulder midpoint & scaled by shoulder width
      - lh_body (63): 21 left-hand keypoints in shoulder-normalized body frame
      - rh_body (63): 21 right-hand keypoints in shoulder-normalized body frame
      - lh_local (60): 20 non-wrist left-hand keypoints normalized by hand bounding-box span
      - rh_local (60): 20 non-wrist right-hand keypoints normalized by hand bounding-box span
      - presence flags (2): [has_left_hand, has_right_hand]
    """
    pose_15: Optional[np.ndarray] = None
    if pose_lms is not None:
        if isinstance(pose_lms, np.ndarray):
            if pose_lms.shape[0] == len(POSE_KEYPOINT_INDICES):
                pose_15 = pose_lms.astype(np.float32)[:, :3]
            elif pose_lms.shape[0] >= 25:
                pose_15 = pose_lms.astype(np.float32)[list(POSE_KEYPOINT_INDICES), :3]
            else:
                raise ValueError(f"Unexpected pose array shape: {pose_lms.shape}")
        else:
            raw_pose = _landmarks_to_array(pose_lms, 33)
            pose_15 = raw_pose[list(POSE_KEYPOINT_INDICES)]

    if pose_15 is not None:
        left_shoulder = pose_15[7]
        right_shoulder = pose_15[8]
        center = (left_shoulder + right_shoulder) * 0.5
        scale = max(float(np.linalg.norm(left_shoulder - right_shoulder)), 0.12)
        pose_norm = ((pose_15 - center) / scale).flatten()
    else:
        center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scale = 0.25
        pose_norm = np.zeros(len(POSE_KEYPOINT_INDICES) * 3, dtype=np.float32)

    if left_hand_lms is not None:
        lh = _landmarks_to_array(left_hand_lms, 21)
        lh_body = ((lh - center) / scale).flatten()
        lh_local = normalize_landmarks(lh)[3:]
        lh_flag = 1.0
    else:
        lh_body = np.zeros(63, dtype=np.float32)
        lh_local = np.zeros(60, dtype=np.float32)
        lh_flag = 0.0

    if right_hand_lms is not None:
        rh = _landmarks_to_array(right_hand_lms, 21)
        rh_body = ((rh - center) / scale).flatten()
        rh_local = normalize_landmarks(rh)[3:]
        rh_flag = 1.0
    else:
        rh_body = np.zeros(63, dtype=np.float32)
        rh_local = np.zeros(60, dtype=np.float32)
        rh_flag = 0.0

    flags = np.array([lh_flag, rh_flag], dtype=np.float32)
    vec = np.concatenate([pose_norm, lh_body, rh_body, lh_local, rh_local, flags]).astype(np.float32)
    vec = np.clip(vec, -5.0, 5.0)
    assert vec.shape == (FSL105_FEATURE_DIM,), f"Expected {FSL105_FEATURE_DIM}, got {vec.shape}"
    return vec


def trim_active_segment(
    vecs: np.ndarray,
    active_mask: np.ndarray,
    margin: int = 2,
) -> np.ndarray:
    """Trim leading/trailing idle frames to the active signing span."""
    if len(vecs) == 0:
        return vecs
    active_idx = np.flatnonzero(active_mask)
    if len(active_idx) >= 6:
        start = max(0, int(active_idx[0]) - margin)
        end = min(len(vecs), int(active_idx[-1]) + margin + 1)
        return vecs[start:end]

    start = int(len(vecs) * 0.1)
    end = int(len(vecs) * 0.9)
    if end - start >= 4:
        return vecs[start:end]
    return vecs


def resample_sequence(seq: np.ndarray, target_len: int = FSL105_SEQ_LEN) -> np.ndarray:
    """Temporally resample a (T, D) landmark sequence to (target_len, D) via linear interpolation."""
    feat_dim = seq.shape[1] if seq.ndim == 2 else FSL105_FEATURE_DIM
    if len(seq) == 0:
        return np.zeros((target_len, feat_dim), dtype=np.float32)

    if len(seq) == target_len:
        out = seq.astype(np.float32).copy()
        out[:, -2:] = (out[:, -2:] >= 0.5).astype(np.float32)
        return out

    if len(seq) == 1:
        out = np.repeat(seq.astype(np.float32), target_len, axis=0)
        out[:, -2:] = (out[:, -2:] >= 0.5).astype(np.float32)
        return out

    src_t = np.linspace(0.0, 1.0, len(seq), dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((target_len, feat_dim), dtype=np.float32)
    for col in range(feat_dim):
        out[:, col] = np.interp(dst_t, src_t, seq[:, col])
    out[:, -2:] = (out[:, -2:] >= 0.5).astype(np.float32)
    return out


def draw_holistic_skeleton(bgr_image: np.ndarray, overlay: Dict[str, Any]) -> np.ndarray:
    """Draw upper-body pose and left/right hand skeletons onto a BGR frame copy."""
    annotated = bgr_image.copy()
    h, w = annotated.shape[:2]

    pose_pts = overlay.get("pose") or []
    if len(pose_pts) == len(POSE_KEYPOINT_INDICES):
        px_pts = [(int(pt[0] * w), int(pt[1] * h)) for pt in pose_pts]
        for a, b in UPPER_BODY_CONNECTIONS:
            cv2.line(annotated, px_pts[a], px_pts[b], (245, 180, 66), 2, cv2.LINE_AA)
        for px, py in px_pts:
            cv2.circle(annotated, (px, py), 3, (255, 220, 130), -1, cv2.LINE_AA)

    for key, line_color in (("left_hand", (56, 189, 248)), ("right_hand", (52, 211, 153))):
        pts = overlay.get(key) or []
        if len(pts) == 21:
            px_pts = [(int(pt[0] * w), int(pt[1] * h)) for pt in pts]
            for a, b in HAND_CONNECTIONS:
                cv2.line(annotated, px_pts[a], px_pts[b], line_color, 2, cv2.LINE_AA)
            for idx, (px, py) in enumerate(px_pts):
                r = 5 if idx == 0 else 3
                cv2.circle(annotated, (px, py), r, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, (px, py), r, line_color, 1, cv2.LINE_AA)

    return annotated


class HolisticLandmarkExtractor:
    """Wrapper around MediaPipe Holistic for upper-body + two-hand sequence extraction."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.45,
    ) -> None:
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=not static_image_mode,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def reset(self) -> None:
        """Feed a tiny black frame to clear internal temporal tracking state between clips."""
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        self._holistic.process(blank)

    def _run_raw(self, bgr_image: np.ndarray) -> Any:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return self._holistic.process(rgb)

    @staticmethod
    def _build_overlay(res: Any) -> Dict[str, Any]:
        pose_pts = []
        hip_y = 0.88
        if res.pose_landmarks is not None:
            lms = res.pose_landmarks.landmark
            pose_pts = [
                [round(float(lms[idx].x), 4), round(float(lms[idx].y), 4)]
                for idx in POSE_KEYPOINT_INDICES
            ]
            hip_y = float((lms[23].y + lms[24].y) * 0.5)

        lh_pts = (
            [[round(float(lm.x), 4), round(float(lm.y), 4)] for lm in res.left_hand_landmarks.landmark]
            if res.left_hand_landmarks is not None
            else []
        )
        rh_pts = (
            [[round(float(lm.x), 4), round(float(lm.y), 4)] for lm in res.right_hand_landmarks.landmark]
            if res.right_hand_landmarks is not None
            else []
        )
        has_hands = bool(lh_pts or rh_pts)

        wrist_ys = []
        if lh_pts:
            wrist_ys.append(lh_pts[0][1])
        if rh_pts:
            wrist_ys.append(rh_pts[0][1])

        cutoff_y = min(hip_y + 0.05, 0.92)
        hands_raised = bool(wrist_ys and min(wrist_ys) < cutoff_y)

        return {
            "pose": pose_pts,
            "left_hand": lh_pts,
            "right_hand": rh_pts,
            "has_hands": has_hands,
            "hands_raised": hands_raised,
        }

    def extract(
        self,
        bgr_image: np.ndarray,
        use_clahe: bool = False,
        auto_clahe_fallback: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """Extract 293-dim holistic vector and 2D overlay points with optional CLAHE fallback."""
        proc = apply_clahe(bgr_image) if use_clahe else bgr_image
        res = self._run_raw(proc)
        used_clahe = use_clahe

        has_hands = res.left_hand_landmarks is not None or res.right_hand_landmarks is not None
        if (
            not has_hands
            and auto_clahe_fallback
            and not use_clahe
            and _is_low_light_or_low_contrast(bgr_image)
        ):
            res_c = self._run_raw(apply_clahe(bgr_image))
            if res_c.left_hand_landmarks is not None or res_c.right_hand_landmarks is not None:
                res = res_c
                used_clahe = True

        vec = normalize_holistic_landmarks(
            res.pose_landmarks,
            res.left_hand_landmarks,
            res.right_hand_landmarks,
        )
        overlay = self._build_overlay(res)
        return vec, overlay, used_clahe

    def extract_video_sequence(
        self,
        video_path: Path,
        sample_frames: int = 54,
        target_len: int = FSL105_SEQ_LEN,
        use_clahe: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract an active-trimmed, fixed-length (target_len, 293) sequence from a video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            n_pick = min(sample_frames, total_frames)
            target_indices = set(np.linspace(0, total_frames - 1, n_pick, dtype=int).tolist())
        else:
            target_indices = None

        vecs = []
        active_flags = []
        frame_idx = 0
        self.reset()

        while True:
            if not cap.grab():
                break
            if target_indices is None or frame_idx in target_indices:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    vec, overlay, _ = self.extract(
                        frame,
                        use_clahe=use_clahe,
                        auto_clahe_fallback=False,
                    )
                    vecs.append(vec)
                    active_flags.append(bool(overlay["hands_raised"] or overlay["has_hands"]))
            frame_idx += 1

        cap.release()

        if not vecs:
            empty = np.zeros((target_len, FSL105_FEATURE_DIM), dtype=np.float32)
            return empty, {"total_frames": frame_idx, "sampled_frames": 0, "active_frames": 0}

        vec_arr = np.stack(vecs, axis=0)
        mask_arr = np.array(active_flags, dtype=bool)
        trimmed = trim_active_segment(vec_arr, mask_arr, margin=2)
        seq = resample_sequence(trimmed, target_len=target_len)
        return seq, {
            "total_frames": frame_idx,
            "sampled_frames": len(vecs),
            "active_frames": int(mask_arr.sum()),
        }

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> "HolisticLandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
