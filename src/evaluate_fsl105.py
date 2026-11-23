"""Evaluate FSL-105 Baseline SVM and FSLPhraseNet on the held-out official test.csv split."""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ImportError:
    sns = None
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import (
    FSL105_LABEL_MAP_PATH,
    FSL105_LANDMARKS_PATH,
    FSL105_MODEL_PATH,
    FSL105_SCALER_PATH,
    FSL105_SVM_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from src.train_fsl105 import FSLPhraseNet, extract_temporal_summary_features


def _topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    top_k = np.argsort(probs, axis=1)[:, -k:]
    hits = np.any(top_k == y_true[:, None], axis=1)
    return float(hits.mean())


def _scale_sequences(X: np.ndarray, scaler) -> np.ndarray:
    N, T, _ = X.shape
    out = X.astype(np.float32).copy()
    out[:, :, :291] = scaler.transform(out[:, :, :291].reshape(-1, 291)).reshape(N, T, 291)
    return out


def _benchmark_svm(svm_pipe, X_feat: np.ndarray, n_iters: int = 200) -> float:
    for i in range(20):
        svm_pipe.predict_proba(X_feat[i % len(X_feat) : (i % len(X_feat)) + 1])
    t0 = time.perf_counter()
    for i in range(n_iters):
        svm_pipe.predict_proba(X_feat[i % len(X_feat) : (i % len(X_feat)) + 1])
    return ((time.perf_counter() - t0) / n_iters) * 1000.0


def _benchmark_net(model: FSLPhraseNet, X_scaled: np.ndarray, n_iters: int = 200) -> float:
    model.eval()
    t_samples = torch.from_numpy(X_scaled.astype(np.float32))
    with torch.no_grad():
        for i in range(20):
            _ = model(t_samples[i % len(t_samples) : (i % len(t_samples)) + 1])
        t0 = time.perf_counter()
        for i in range(n_iters):
            _ = model(t_samples[i % len(t_samples) : (i % len(t_samples)) + 1])
    return ((time.perf_counter() - t0) / n_iters) * 1000.0


def _compute_metrics(
    y_true: np.ndarray, probs: np.ndarray, categories_by_id: List[str], unique_cats: List[str]
) -> Dict[str, Any]:
    y_pred = probs.argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    per_category: Dict[str, Dict[str, Any]] = {}
    for cat in unique_cats:
        mask = np.array([categories_by_id[int(cid)] == cat for cid in y_true], dtype=bool)
        if mask.any():
            per_category[cat] = {
                "samples": int(mask.sum()),
                "top1_accuracy": round(float((y_pred[mask] == y_true[mask]).mean()), 4),
                "top3_accuracy": round(_topk_accuracy(probs[mask], y_true[mask], 3), 4),
            }

    return {
        "top1_accuracy": round(acc, 4),
        "top3_accuracy": round(_topk_accuracy(probs, y_true, 3), 4),
        "top5_accuracy": round(_topk_accuracy(probs, y_true, 5), 4),
        "macro_precision": round(float(p_mac), 4),
        "macro_recall": round(float(r_mac), 4),
        "macro_f1": round(float(f1_mac), 4),
        "per_category": per_category,
    }


def _save_category_chart(
    svm_cats: Dict[str, Dict[str, Any]],
    net_cats: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    cats = list(net_cats.keys())
    y_pos = np.arange(len(cats))
    svm_vals = [svm_cats[c]["top1_accuracy"] * 100 for c in cats]
    net_vals = [net_cats[c]["top1_accuracy"] * 100 for c in cats]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    h = 0.36
    ax.barh(y_pos - h / 2, svm_vals, height=h, label="Baseline RBF-SVM", color="#71717a")
    ax.barh(y_pos + h / 2, net_vals, height=h, label="FSLPhraseNet (Main)", color="#ffae00")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlim(0, 111)
    ax.set_xlabel("Top-1 Test Accuracy (%)")
    ax.set_title("FSL-105 Held-Out Test Accuracy by Category", pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False)
    for idx, val in enumerate(net_vals):
        ax.text(val + 1.0, idx + h / 2, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_category_cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categories_by_id: List[str],
    unique_cats: List[str],
    out_path: Path,
) -> None:
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    yt_cat = [cat_to_idx[categories_by_id[int(c)]] for c in y_true]
    yp_cat = [cat_to_idx[categories_by_id[int(c)]] for c in y_pred]
    cm = confusion_matrix(yt_cat, yp_cat, labels=list(range(len(unique_cats))))

    fig, ax = plt.subplots(figsize=(9, 7.5))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrBr",
            xticklabels=unique_cats,
            yticklabels=unique_cats,
            cbar=False,
            ax=ax,
        )
    else:
        ax.imshow(cm, cmap="YlOrBr", aspect="auto")
        ax.set_xticks(range(len(unique_cats)))
        ax.set_yticks(range(len(unique_cats)))
        ax.set_xticklabels(unique_cats, rotation=45, ha="right")
        ax.set_yticklabels(unique_cats)
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(
                    c, r, str(cm[r, c]),
                    ha="center", va="center",
                    color="white" if cm[r, c] > thresh else "black",
                    fontsize=9,
                )
    ax.set_title("FSL-105 Category-Level Confusion Matrix (FSLPhraseNet)")
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("True Category")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(FSL105_LANDMARKS_PATH)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)
    test_paths = [str(p) for p in data["test_paths"]]

    label_map = json.loads(FSL105_LABEL_MAP_PATH.read_text(encoding="utf-8"))
    labels_by_id = [label_map[str(i)]["label"] for i in range(len(label_map))]
    categories_by_id = [label_map[str(i)]["category"] for i in range(len(label_map))]
    unique_cats = list(dict.fromkeys(categories_by_id))

    # 1. Baseline SVM
    svm_pipe = joblib.load(FSL105_SVM_PATH)
    X_test_feat = extract_temporal_summary_features(X_test)
    svm_probs = svm_pipe.predict_proba(X_test_feat)
    svm_metrics = _compute_metrics(y_test, svm_probs, categories_by_id, unique_cats)
    svm_metrics["cpu_latency_ms"] = round(_benchmark_svm(svm_pipe, X_test_feat), 3)
    svm_metrics["model_size_kb"] = round(FSL105_SVM_PATH.stat().st_size / 1024.0, 1)

    # 2. Main FSLPhraseNet
    scaler = joblib.load(FSL105_SCALER_PATH)
    ckpt = torch.load(FSL105_MODEL_PATH, map_location="cpu", weights_only=True)
    net = FSLPhraseNet(
        input_dim=ckpt["input_dim"],
        num_classes=ckpt["num_classes"],
        hidden_dim=ckpt["hidden_dim"],
        gru_hidden=ckpt["gru_hidden"],
        dropout=ckpt["dropout"],
    )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    X_test_scaled = _scale_sequences(X_test, scaler)
    with torch.no_grad():
        logits = net(torch.from_numpy(X_test_scaled))
        net_probs = torch.softmax(logits, dim=1).numpy()
        net_pred = net_probs.argmax(axis=1)

    net_metrics = _compute_metrics(y_test, net_probs, categories_by_id, unique_cats)
    net_metrics["cpu_latency_ms"] = round(_benchmark_net(net, X_test_scaled), 3)
    net_metrics["model_size_kb"] = round(FSL105_MODEL_PATH.stat().st_size / 1024.0, 1)

    results = {
        "split": "test",
        "test_samples": int(len(y_test)),
        "num_classes": len(label_map),
        "machine": f"{platform.system()} ({platform.processor() or platform.machine()})",
        "svm_baseline": svm_metrics,
        "fsl_phrasenet": net_metrics,
    }
    (REPORTS_DIR / "fsl105_evaluation_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    errors = []
    for idx in range(len(y_test)):
        if net_pred[idx] != y_test[idx]:
            errors.append({
                "test_index": int(idx),
                "vid_path": test_paths[idx],
                "true_label": labels_by_id[int(y_test[idx])],
                "pred_label": labels_by_id[int(net_pred[idx])],
                "pred_confidence": round(float(net_probs[idx, net_pred[idx]]), 4),
                "true_confidence": round(float(net_probs[idx, y_test[idx]]), 4),
            })
    (REPORTS_DIR / "fsl105_test_errors.json").write_text(
        json.dumps(errors, indent=2), encoding="utf-8"
    )

    _save_category_chart(
        svm_metrics["per_category"],
        net_metrics["per_category"],
        REPORTS_DIR / "fsl105_category_accuracy.png",
    )
    _save_category_cm(
        y_test,
        net_pred,
        categories_by_id,
        unique_cats,
        REPORTS_DIR / "fsl105_confusion_matrix.png",
    )

    # Copy key deliverables to outputs/final/
    final_dir = PROJECT_ROOT / "outputs" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for fname in (
        "fsl105_evaluation_metrics.json",
        "fsl105_category_accuracy.png",
        "fsl105_confusion_matrix.png",
        "fsl105_training_curves.png",
    ):
        src = REPORTS_DIR / fname
        if src.exists():
            shutil.copy2(src, final_dir / fname)

    print(f"\n=== FSL-105 Official Test Evaluation (n={len(y_test)}, 105 classes) ===")
    for key in ("svm_baseline", "fsl_phrasenet"):
        m = results[key]
        print(
            f"{key:15s} | Top-1={m['top1_accuracy']*100:5.2f}% | Top-3={m['top3_accuracy']*100:5.2f}% "
            f"| Top-5={m['top5_accuracy']*100:5.2f}% | Macro F1={m['macro_f1']:.4f} "
            f"| Latency={m['cpu_latency_ms']:.3f} ms | Size={m['model_size_kb']:.1f} KB"
        )


if __name__ == "__main__":
    main()
