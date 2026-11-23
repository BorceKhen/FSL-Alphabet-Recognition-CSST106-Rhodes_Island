"""Inference wrapper for FSL-105 dynamic phrase/word recognition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch

from src.config import (
    FSL105_LABEL_MAP_PATH,
    FSL105_MODEL_PATH,
    FSL105_SCALER_PATH,
    FSL105_SVM_PATH,
)
from src.features import HolisticLandmarkExtractor
from src.train_fsl105 import FSLPhraseNet, extract_temporal_summary_features


class PhrasePredictor:
    """Loads and runs inference for FSLPhraseNet (Main) and Temporal-Summary RBF-SVM (Baseline)."""

    def __init__(self) -> None:
        if not FSL105_LABEL_MAP_PATH.exists():
            raise FileNotFoundError(f"Missing label map: {FSL105_LABEL_MAP_PATH}")
        raw_map = json.loads(FSL105_LABEL_MAP_PATH.read_text(encoding="utf-8"))
        self.label_entries: List[Dict[str, Any]] = [
            raw_map[str(i)] for i in range(len(raw_map))
        ]

        self.scaler = joblib.load(FSL105_SCALER_PATH)
        ckpt = torch.load(FSL105_MODEL_PATH, map_location="cpu", weights_only=True)
        self.net = FSLPhraseNet(
            input_dim=ckpt["input_dim"],
            num_classes=ckpt["num_classes"],
            hidden_dim=ckpt["hidden_dim"],
            gru_hidden=ckpt["gru_hidden"],
            dropout=ckpt["dropout"],
        )
        self.net.load_state_dict(ckpt["model_state_dict"])
        self.net.eval()

        self.svm = joblib.load(FSL105_SVM_PATH) if FSL105_SVM_PATH.exists() else None

    def predict_sequence_top_k(
        self,
        seq_40x293: np.ndarray,
        mode: str = "FSLPhraseNet",
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Predict top-k FSL-105 classes for a (T, 293) holistic sequence."""
        seq = seq_40x293.astype(np.float32)
        if seq.ndim == 2:
            seq = seq[None, :, :]  # (1, T, 293)

        if mode in ("FSL-105 SVM", "Baseline SVM") and self.svm is not None:
            feat = extract_temporal_summary_features(seq)
            probs = self.svm.predict_proba(feat)[0]
        else:
            N, T, _ = seq.shape
            scaled = seq.copy()
            scaled[:, :, :291] = self.scaler.transform(
                scaled[:, :, :291].reshape(-1, 291)
            ).reshape(N, T, 291)
            with torch.no_grad():
                logits = self.net(torch.from_numpy(scaled))
                probs = torch.softmax(logits, dim=1).numpy()[0]

        top_indices = np.argsort(probs)[::-1][:k]
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            entry = self.label_entries[int(idx)]
            results.append({
                "id": int(entry["id"]),
                "label": entry["label"],
                "category": entry["category"],
                "prob": round(float(probs[int(idx)]), 4),
            })
        return results

    def predict_video_top_k(
        self,
        video_path: Path,
        extractor: Optional[HolisticLandmarkExtractor] = None,
        mode: str = "FSLPhraseNet",
        k: int = 5,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract sequence from a video file and return (top_k_predictions, extraction_stats)."""
        if extractor is None:
            with HolisticLandmarkExtractor(static_image_mode=False, model_complexity=1) as ext:
                seq, stats = ext.extract_video_sequence(video_path)
        else:
            seq, stats = extractor.extract_video_sequence(video_path)
        return self.predict_sequence_top_k(seq, mode=mode, k=k), stats
