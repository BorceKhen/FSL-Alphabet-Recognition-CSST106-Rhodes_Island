"""Extract 63-dim MediaPipe hand landmarks from the FSL dataset and create stratified splits."""

from __future__ import annotations

import json
from typing import Dict, List

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSES,
    DATA_DIR,
    DATASET_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.features import HandLandmarkExtractor


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    features: List[np.ndarray] = []
    labels: List[int] = []
    per_class_stats: Dict[str, Dict[str, int]] = {}

    print(f"Extracting landmarks from: {DATASET_DIR}")
    with HandLandmarkExtractor(running_mode="IMAGE", min_detection_confidence=0.3) as extractor:
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = DATASET_DIR / class_name
            image_paths = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            detected = 0
            failed = 0

            for img_path in image_paths:
                img = cv2.imread(str(img_path))
                if img is None:
                    failed += 1
                    continue
                vec, _, _ = extractor.extract_with_fallback(img)
                if vec is not None:
                    features.append(vec)
                    labels.append(class_idx)
                    detected += 1
                else:
                    failed += 1

            per_class_stats[class_name] = {
                "total": len(image_paths),
                "detected": detected,
                "failed": failed,
            }
            print(f"  Class {class_name}: {detected}/{len(image_paths)} detected ({failed} skipped)")

    X = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    # Stratified 70% / 15% / 15% split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y,
        random_state=RANDOM_SEED,
    )
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    out_npz = DATA_DIR / "fsl_landmarks.npz"
    np.savez_compressed(
        out_npz,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    label_map = {str(i): c for i, c in enumerate(CLASSES)}
    (MODELS_DIR / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    summary = {
        "total_images": sum(s["total"] for s in per_class_stats.values()),
        "total_detected": int(len(y)),
        "detection_rate": round(len(y) / sum(s["total"] for s in per_class_stats.values()), 4),
        "splits": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "per_class": per_class_stats,
    }
    (REPORTS_DIR / "dataset_extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nDataset preparation complete:")
    print(f"  Total detected: {summary['total_detected']}/{summary['total_images']} ({summary['detection_rate']*100:.2f}%)")
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"  Saved to: {out_npz}")


if __name__ == "__main__":
    main()
