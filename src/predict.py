"""Single-command image prediction prototype for Milestone 2 (and standalone CLI inference)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly as `python src/predict.py` or as `python -m src.predict`
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.app import ModelPredictor
from src.features import HandLandmarkExtractor, apply_clahe, draw_hand_skeleton


def run_on_image(
    image_path: Path,
    output_dir: Path,
    model_mode: str = "Baseline SVM",
    use_clahe: bool = False,
) -> Path:
    """Run preprocessing -> landmark extraction -> classification -> annotated output image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    predictor = ModelPredictor()
    with HandLandmarkExtractor(running_mode="IMAGE", min_detection_confidence=0.3) as extractor:
        proc_img = apply_clahe(img) if use_clahe else img
        vec, landmarks = extractor.extract(proc_img, use_clahe=False)

    annotated = proc_img.copy()
    if vec is not None and landmarks is not None:
        annotated = draw_hand_skeleton(annotated, landmarks)
        label, conf = predictor.predict(vec, mode=model_mode)
        text = f"{label} ({conf * 100:.1f}%) [{model_mode}]"
        color = (0, 220, 120)
    else:
        label, conf = "No Hand", 0.0
        text = "No hand detected"
        color = (0, 165, 255)

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (15, 20, 27), -1)
    cv2.putText(
        annotated,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pred_{image_path.stem}.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"Input: {image_path.name} -> Prediction: {label} ({conf * 100:.2f}%) | Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="FSL Single-Image Prediction Prototype")
    parser.add_argument("--input", required=True, type=Path, help="Path to an input image")
    parser.add_argument("--output", default=Path("outputs"), type=Path, help="Output directory")
    parser.add_argument(
        "--model",
        default="Baseline SVM",
        choices=["Baseline SVM", "Deep MLP"],
        help="Classifier to use",
    )
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast enhancement")
    args = parser.parse_args()

    run_on_image(args.input, args.output, model_mode=args.model, use_clahe=args.clahe)


if __name__ == "__main__":
    main()
