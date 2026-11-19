"""Unified single-command FSL pipeline supporting images, folders, videos, and webcam."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.app import ModelPredictor
from src.config import CAMERA_HEIGHT, CAMERA_WIDTH, TARGET_FPS
from src.features import HandLandmarkExtractor, apply_clahe, draw_hand_skeleton
from src.postprocess import ProcessorState, WordFormationProcessor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def preprocess(
    frame: np.ndarray,
    extractor: HandLandmarkExtractor,
    use_clahe: bool = False,
    allow_pad_fallback: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[Sequence], np.ndarray]:
    """Stage 1: Optional CLAHE + MediaPipe 21-keypoint extraction & 63-dim normalization."""
    if allow_pad_fallback:
        return extractor.extract_with_fallback(frame, use_clahe=use_clahe)
    proc = apply_clahe(frame) if use_clahe else frame
    vec, landmarks = extractor.extract(proc, use_clahe=False)
    return vec, landmarks, proc


def infer(
    vec: Optional[np.ndarray],
    predictor: ModelPredictor,
    model_mode: str = "Deep MLP",
) -> Tuple[Optional[str], float]:
    """Stage 2: Classify 63-dim vector into A-Z label and probability."""
    if vec is None:
        return None, 0.0
    return predictor.predict(vec, mode=model_mode)


def postprocess(
    label: Optional[str],
    confidence: float,
    processor: Optional[WordFormationProcessor],
    timestamp_sec: float,
) -> ProcessorState:
    """Stage 3: Apply confidence thresholding, temporal smoothing, and dwell confirmation."""
    if processor is None:
        return ProcessorState(
            raw_label=label,
            raw_confidence=confidence,
            smoothed_label=label,
            dwell_progress=1.0 if label else 0.0,
            just_confirmed=None,
            text_buffer=label or "",
        )
    return processor.update(label, confidence, timestamp_sec)


def render(
    frame: np.ndarray,
    landmarks: Optional[Sequence],
    state: ProcessorState,
    show_text_bar: bool = True,
) -> np.ndarray:
    """Stage 4: Draw hand skeleton overlay, predicted label, confidence, and buffer."""
    out = draw_hand_skeleton(frame, landmarks) if landmarks is not None else frame.copy()
    h, w = out.shape[:2]

    cv2.rectangle(out, (0, 0), (w, 36), (15, 20, 27), -1)
    if state.raw_label is not None:
        active = state.smoothed_label or state.raw_label
        msg = f"Pred: {active} ({state.raw_confidence * 100:.1f}%)"
        color = (56, 189, 248) if state.raw_confidence >= 0.75 else (0, 165, 255)
    else:
        msg = "No hand detected"
        color = (0, 165, 255)

    cv2.putText(out, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    if show_text_bar and state.text_buffer:
        cv2.rectangle(out, (0, h - 38), (w, h), (15, 20, 27), -1)
        cv2.putText(
            out,
            f"Text: {state.text_buffer[-36:]}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 235, 242),
            2,
            cv2.LINE_AA,
        )
    return out


def run_images(
    inputs: list[Path],
    output_dir: Path,
    model_mode: str,
    use_clahe: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = ModelPredictor()
    csv_path = output_dir / "results.csv"

    with HandLandmarkExtractor(running_mode="IMAGE", min_detection_confidence=0.3) as ext, open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["file", "hand_detected", "predicted_label", "confidence", "latency_ms"])

        for img_path in inputs:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Skipping unreadable file: {img_path}")
                continue
            t0 = time.perf_counter()
            vec, landmarks, proc = preprocess(
                frame, ext, use_clahe=use_clahe, allow_pad_fallback=True
            )
            label, conf = infer(vec, predictor, model_mode=model_mode)
            state = postprocess(label, conf, processor=None, timestamp_sec=0.0)
            annotated = render(proc, landmarks, state, show_text_bar=False)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            out_img = output_dir / f"out_{img_path.stem}.png"
            cv2.imwrite(str(out_img), annotated)
            writer.writerow([img_path.name, vec is not None, label or "NONE", f"{conf:.4f}", f"{latency_ms:.2f}"])
            print(f"{img_path.name} -> {label or 'NONE'} ({conf * 100:.1f}%, {latency_ms:.1f} ms) -> {out_img.name}")

    print(f"Saved CSV summary: {csv_path}")


def run_video(
    video_path: Path,
    output_path: Path,
    model_mode: str,
    use_clahe: bool,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    predictor = ModelPredictor()
    processor = WordFormationProcessor()
    frame_idx = 0

    with HandLandmarkExtractor(running_mode="IMAGE") as ext:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_sec = frame_idx / fps
            vec, landmarks, proc = preprocess(frame, ext, use_clahe=use_clahe)
            label, conf = infer(vec, predictor, model_mode=model_mode)
            state = postprocess(label, conf, processor=processor, timestamp_sec=ts_sec)
            annotated = render(proc, landmarks, state, show_text_bar=True)
            writer.write(annotated)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"Processed {frame_idx} frames -> {output_path} | Transcribed text: {processor.get_text()!r}")


def run_webcam(cam_index: int, model_mode: str, use_clahe: bool) -> None:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {cam_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    predictor = ModelPredictor()
    processor = WordFormationProcessor()
    start = time.monotonic()

    with HandLandmarkExtractor(running_mode="IMAGE") as ext:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            vec, landmarks, proc = preprocess(frame, ext, use_clahe=use_clahe)
            label, conf = infer(vec, predictor, model_mode=model_mode)
            state = postprocess(label, conf, processor=processor, timestamp_sec=time.monotonic() - start)
            annotated = render(proc, landmarks, state, show_text_bar=True)
            cv2.imshow("FSL Pipeline (Press Q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="FSL End-to-End Pipeline Runner")
    parser.add_argument("--input", type=Path, help="Image file, image folder, or video file")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory or .mp4 file")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam index (e.g. 0)")
    parser.add_argument("--model", default="Deep MLP", choices=["Deep MLP", "Baseline SVM"])
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE lighting enhancement")
    args = parser.parse_args()

    if args.webcam is not None:
        run_webcam(args.webcam, args.model, args.clahe)
        return

    if args.input is None:
        parser.error("Specify either --input <path> or --webcam <index>.")

    if args.input.is_dir():
        imgs = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        run_images(imgs, args.output, args.model, args.clahe)
    elif args.input.suffix.lower() in VIDEO_EXTS:
        out_vid = args.output if args.output.suffix.lower() in VIDEO_EXTS else args.output / f"out_{args.input.name}"
        run_video(args.input, out_vid, args.model, args.clahe)
    elif args.input.suffix.lower() in IMAGE_EXTS:
        run_images([args.input], args.output, args.model, args.clahe)
    else:
        parser.error(f"Unsupported input: {args.input}")


if __name__ == "__main__":
    main()
