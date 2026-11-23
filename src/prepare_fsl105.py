"""Extract 293-dim Holistic sequences and representative thumbnails from FSL-105 videos."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    FSL105_DIR,
    FSL105_FEATURE_DIM,
    FSL105_LABEL_MAP_PATH,
    FSL105_LANDMARKS_PATH,
    FSL105_SEQ_LEN,
    FSL105_THUMBS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.features import HolisticLandmarkExtractor
from src.translator import clean_fsl_label

_WORKER_EXTRACTOR: HolisticLandmarkExtractor | None = None


def _init_worker() -> None:
    global _WORKER_EXTRACTOR
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)
    _WORKER_EXTRACTOR = HolisticLandmarkExtractor(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )


def _process_clip_task(task: Tuple[int, str, int]) -> Tuple[int, int, np.ndarray, int, int]:
    idx, video_path_str, label_id = task
    assert _WORKER_EXTRACTOR is not None
    seq, stats = _WORKER_EXTRACTOR.extract_video_sequence(
        Path(video_path_str),
        sample_frames=44,
        target_len=FSL105_SEQ_LEN,
        use_clahe=False,
    )
    return idx, label_id, seq, int(stats["active_frames"]), int(stats["sampled_frames"])


def _extract_split(df: pd.DataFrame, max_workers: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    n = len(df)
    X = np.zeros((n, FSL105_SEQ_LEN, FSL105_FEATURE_DIM), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    active_counts = np.zeros((n,), dtype=np.int32)
    paths = df["vid_path"].astype(str).tolist()

    tasks = [
        (i, str(FSL105_DIR / row["vid_path"]), int(row["id_label"]))
        for i, (_, row) in enumerate(df.iterrows())
    ]

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as pool:
        futures = [pool.submit(_process_clip_task, t) for t in tasks]
        for fut in as_completed(futures):
            idx, label_id, seq, active_cnt, _ = fut.result()
            X[idx] = seq
            y[idx] = label_id
            active_counts[idx] = active_cnt
            done += 1
            if done % 50 == 0 or done == n:
                print(f"    Processed {done}/{n} clips...", flush=True)

    return X, y, active_counts, paths


def _save_class_thumbnails(train_df: pd.DataFrame) -> int:
    FSL105_THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for cid, group in train_df.groupby("id_label"):
        out_path = FSL105_THUMBS_DIR / f"{int(cid)}.jpg"
        if out_path.exists():
            saved += 1
            continue
        vid_path = FSL105_DIR / str(group.iloc[0]["vid_path"])
        cap = cv2.VideoCapture(str(vid_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = int(total * 0.55) if total > 0 else 60
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 86])
            saved += 1
    return saved


def main() -> None:
    if not FSL105_DIR.is_dir():
        raise FileNotFoundError(f"FSL-105 dataset folder not found at: {FSL105_DIR}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(FSL105_DIR / "labels.csv")
    train_df = pd.read_csv(FSL105_DIR / "train.csv")
    test_df = pd.read_csv(FSL105_DIR / "test.csv")

    label_map: Dict[str, Dict[str, Any]] = {}
    for _, row in labels_df.iterrows():
        cid = int(row["id"])
        raw_lbl = str(row["label"])
        label_map[str(cid)] = {
            "id": cid,
            "raw_label": raw_lbl,
            "label": clean_fsl_label(raw_lbl),
            "category": str(row["category"]).strip().upper(),
        }
    FSL105_LABEL_MAP_PATH.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    workers = min(6, max(2, (os.cpu_count() or 4) // 2))
    print(f"Preparing FSL-105 dataset from {FSL105_DIR} using {workers} worker processes...", flush=True)

    t0 = time.perf_counter()
    print(f"  Extracting train split ({len(train_df)} clips)...", flush=True)
    X_train, y_train, train_active, train_paths = _extract_split(train_df, max_workers=workers)

    print(f"  Extracting test split ({len(test_df)} clips)...", flush=True)
    X_test, y_test, test_active, test_paths = _extract_split(test_df, max_workers=workers)

    print("  Generating 105 class thumbnails...", flush=True)
    n_thumbs = _save_class_thumbnails(train_df)

    np.savez_compressed(
        FSL105_LANDMARKS_PATH,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_paths=np.array(train_paths),
        test_paths=np.array(test_paths),
        train_active_counts=train_active,
        test_active_counts=test_active,
    )

    elapsed = round(time.perf_counter() - t0, 2)
    train_zero = int((train_active == 0).sum())
    test_zero = int((test_active == 0).sum())
    summary = {
        "dataset_dir": str(FSL105_DIR),
        "num_classes": len(label_map),
        "seq_len": FSL105_SEQ_LEN,
        "feature_dim": FSL105_FEATURE_DIM,
        "train_clips": int(len(train_df)),
        "test_clips": int(len(test_df)),
        "train_zero_hand_clips": train_zero,
        "test_zero_hand_clips": test_zero,
        "train_median_active_frames": int(np.median(train_active)),
        "test_median_active_frames": int(np.median(test_active)),
        "thumbnails_saved": n_thumbs,
        "elapsed_seconds": elapsed,
    }
    (REPORTS_DIR / "fsl105_extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nFSL-105 landmark extraction complete:")
    print(f"  Train: {X_train.shape} | zero-hand clips: {train_zero}/{len(train_df)}")
    print(f"  Test:  {X_test.shape}  | zero-hand clips: {test_zero}/{len(test_df)}")
    print(f"  Thumbnails: {n_thumbs}/105 | Elapsed: {elapsed:.1f}s")
    print(f"  Saved to: {FSL105_LANDMARKS_PATH}")


if __name__ == "__main__":
    main()
