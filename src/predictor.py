"""Alphabet (A-Z) model inference wrapper for Deep MLP and Baseline RBF-SVM."""

from __future__ import annotations

import json
from typing import Tuple

import joblib
import numpy as np
import torch

from src.config import CLASSES, MODELS_DIR
from src.train_mlp import FSLDeepMLP


class ModelPredictor:
    """Loads and runs inference for either the Deep MLP or Baseline RBF-SVM."""

    def __init__(self) -> None:
        label_map_path = MODELS_DIR / "label_map.json"
        if label_map_path.exists():
            raw_map = json.loads(label_map_path.read_text(encoding="utf-8"))
            self.classes = [raw_map[str(i)] for i in range(len(raw_map))]
        else:
            self.classes = CLASSES

        self.mlp_scaler = joblib.load(MODELS_DIR / "mlp_scaler.joblib")
        ckpt = torch.load(MODELS_DIR / "mlp_model.pt", map_location="cpu", weights_only=True)
        self.mlp = FSLDeepMLP(
            input_dim=ckpt["input_dim"],
            hidden_dims=tuple(ckpt["hidden_dims"]),
            num_classes=ckpt["num_classes"],
            dropout=ckpt["dropout"],
        )
        self.mlp.load_state_dict(ckpt["model_state_dict"])
        self.mlp.eval()

        svm_path = MODELS_DIR / "svm_baseline.joblib"
        self.svm = joblib.load(svm_path) if svm_path.exists() else None

    def predict_top_k(
        self, vec_63: np.ndarray, mode: str = "Deep MLP", k: int = 3
    ) -> list[Tuple[str, float]]:
        x = vec_63.reshape(1, -1)
        if mode == "Baseline SVM" and self.svm is not None:
            probs = self.svm.predict_proba(x)[0]
        else:
            x_scaled = self.mlp_scaler.transform(x).astype(np.float32)
            with torch.no_grad():
                logits = self.mlp(torch.from_numpy(x_scaled))
                probs = torch.softmax(logits, dim=1).numpy()[0]

        top_indices = np.argsort(probs)[::-1][:k]
        return [(self.classes[int(idx)], float(probs[int(idx)])) for idx in top_indices]

    def predict(self, vec_63: np.ndarray, mode: str = "Deep MLP") -> Tuple[str, float]:
        return self.predict_top_k(vec_63, mode=mode, k=1)[0]
