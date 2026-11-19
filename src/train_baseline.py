"""Train and tune the Baseline RBF-SVM classifier on 63-dim landmark vectors."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import DATA_DIR, MODELS_DIR, RANDOM_SEED, REPORTS_DIR


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(DATA_DIR / "fsl_landmarks.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    param_grid = [
        {"C": c, "gamma": g}
        for c in (1.0, 10.0, 50.0, 100.0)
        for g in ("scale", 0.01, 0.05)
    ]

    best_acc = -1.0
    best_params = None
    best_pipeline = None
    tuning_results = []

    print(f"Tuning RBF-SVM on {len(X_train)} train / {len(X_val)} val samples...")
    for params in param_grid:
        t0 = time.perf_counter()
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=params["C"],
                gamma=params["gamma"],
                probability=True,
                random_state=RANDOM_SEED,
            )),
        ])
        pipe.fit(X_train, y_train)
        val_acc = float(accuracy_score(y_val, pipe.predict(X_val)))
        elapsed = time.perf_counter() - t0

        tuning_results.append({
            "C": params["C"],
            "gamma": params["gamma"],
            "val_accuracy": round(val_acc, 4),
            "train_seconds": round(elapsed, 2),
        })
        print(f"  C={params['C']:<5} gamma={str(params['gamma']):<6} -> val_acc={val_acc:.4f} ({elapsed:.1f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            best_pipeline = pipe

    model_path = MODELS_DIR / "svm_baseline.joblib"
    joblib.dump(best_pipeline, model_path)

    report = {
        "best_params": best_params,
        "best_val_accuracy": round(best_acc, 4),
        "trials": tuning_results,
    }
    (REPORTS_DIR / "svm_tuning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved best SVM ({best_params}, val_acc={best_acc:.4f}) to {model_path}")


if __name__ == "__main__":
    main()
