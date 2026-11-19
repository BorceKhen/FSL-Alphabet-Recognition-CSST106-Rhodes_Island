"""Evaluate Baseline SVM and Deep MLP on the held-out 15% Test Set."""

from __future__ import annotations

import json
import platform
import time
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import CLASSES, DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.train_mlp import FSLDeepMLP


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    acc = float(accuracy_score(y_true, y_pred))
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class = classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_mac), 4),
        "macro_recall": round(float(r_mac), 4),
        "macro_f1": round(float(f1_mac), 4),
        "weighted_precision": round(float(p_w), 4),
        "weighted_recall": round(float(r_w), 4),
        "weighted_f1": round(float(f1_w), 4),
        "per_class": per_class,
    }


def _benchmark_svm_latency(svm_pipe, X_sample: np.ndarray, n_iters: int = 500) -> float:
    # Warm-up run
    for i in range(20):
        svm_pipe.predict_proba(X_sample[i % len(X_sample) : (i % len(X_sample)) + 1])
    t0 = time.perf_counter()
    for i in range(n_iters):
        svm_pipe.predict_proba(X_sample[i % len(X_sample) : (i % len(X_sample)) + 1])
    return ((time.perf_counter() - t0) / n_iters) * 1000.0


def _benchmark_mlp_latency(model: FSLDeepMLP, X_scaled: np.ndarray, n_iters: int = 1000) -> float:
    model.eval()
    tensor_samples = torch.from_numpy(X_scaled.astype(np.float32))
    with torch.no_grad():
        # Warm-up run
        for i in range(50):
            _ = model(tensor_samples[i % len(tensor_samples) : (i % len(tensor_samples)) + 1])
        t0 = time.perf_counter()
        for i in range(n_iters):
            _ = model(tensor_samples[i % len(tensor_samples) : (i % len(tensor_samples)) + 1])
    return ((time.perf_counter() - t0) / n_iters) * 1000.0


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA_DIR / "fsl_landmarks.npz")
    X_test, y_test = data["X_test"], data["y_test"]

    results: Dict[str, Dict] = {"test_samples": int(len(y_test))}

    # 1. Baseline SVM
    svm_pipe = joblib.load(MODELS_DIR / "svm_baseline.joblib")
    svm_pred = svm_pipe.predict(X_test)
    svm_metrics = _compute_metrics(y_test, svm_pred)
    svm_metrics["cpu_latency_ms"] = round(_benchmark_svm_latency(svm_pipe, X_test), 3)
    results["svm_baseline"] = svm_metrics
    _save_confusion_matrix(
        y_test,
        svm_pred,
        f"Baseline RBF-SVM Confusion Matrix (Acc={svm_metrics['accuracy']*100:.2f}%)",
        REPORTS_DIR / "confusion_matrix_svm.png",
    )

    # 2. Deep MLP
    mlp_scaler = joblib.load(MODELS_DIR / "mlp_scaler.joblib")
    ckpt = torch.load(MODELS_DIR / "mlp_model.pt", map_location="cpu", weights_only=True)
    mlp_model = FSLDeepMLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=tuple(ckpt["hidden_dims"]),
        num_classes=ckpt["num_classes"],
        dropout=ckpt["dropout"],
    )
    mlp_model.load_state_dict(ckpt["model_state_dict"])
    mlp_model.eval()

    X_test_scaled = mlp_scaler.transform(X_test).astype(np.float32)
    with torch.no_grad():
        logits = mlp_model(torch.from_numpy(X_test_scaled))
        mlp_probs = torch.softmax(logits, dim=1).numpy()
        mlp_pred = mlp_probs.argmax(axis=1)

    mlp_metrics = _compute_metrics(y_test, mlp_pred)
    mlp_metrics["cpu_latency_ms"] = round(_benchmark_mlp_latency(mlp_model, X_test_scaled), 3)
    results["deep_mlp"] = mlp_metrics
    results["machine"] = f"{platform.system()} ({platform.processor() or platform.machine()})"
    _save_confusion_matrix(
        y_test,
        mlp_pred,
        f"Deep MLP Confusion Matrix (Acc={mlp_metrics['accuracy']*100:.2f}%)",
        REPORTS_DIR / "confusion_matrix_mlp.png",
    )

    # Save detailed test-set misclassifications for error analysis
    errors = []
    for idx in range(len(y_test)):
        if mlp_pred[idx] != y_test[idx]:
            errors.append({
                "test_index": int(idx),
                "true_label": CLASSES[int(y_test[idx])],
                "pred_label": CLASSES[int(mlp_pred[idx])],
                "pred_confidence": round(float(mlp_probs[idx, mlp_pred[idx]]), 4),
                "true_confidence": round(float(mlp_probs[idx, y_test[idx]]), 4),
            })
    (REPORTS_DIR / "test_errors_mlp.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    (REPORTS_DIR / "evaluation_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print(f"\n=== Test Set Evaluation Summary ({results['machine']}) ===")
    for name in ("svm_baseline", "deep_mlp"):
        m = results[name]
        print(
            f"{name:14s} | Acc={m['accuracy']:.4f} | Macro F1={m['macro_f1']:.4f} "
            f"| Prec={m['macro_precision']:.4f} | Rec={m['macro_recall']:.4f} "
            f"| Latency={m['cpu_latency_ms']:.3f} ms"
        )
    print(f"\nDeep MLP Per-Class Classification Report (Test Set, n={len(y_test)}):")
    print(classification_report(y_test, mlp_pred, target_names=CLASSES, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
