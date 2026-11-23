"""Train Baseline Temporal-Summary RBF-SVM and Main FSLPhraseNet on FSL-105 sequences."""

from __future__ import annotations

import json
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    FSL105_FEATURE_DIM,
    FSL105_LANDMARKS_PATH,
    FSL105_MODEL_PATH,
    FSL105_NUM_CLASSES,
    FSL105_SCALER_PATH,
    FSL105_SEQ_LEN,
    FSL105_SVM_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)


def extract_temporal_summary_features(X_seq: np.ndarray) -> np.ndarray:
    """Compute fixed-length temporal summary statistics (N, 1636) for baseline SVM."""
    mean_all = X_seq.mean(axis=1)
    std_all = X_seq.std(axis=1)
    q1 = X_seq[:, :10, :].mean(axis=1)
    mid = X_seq[:, 10:30, :].mean(axis=1)
    q4 = X_seq[:, 30:, :].mean(axis=1)
    vel_max = np.abs(np.diff(X_seq[:, :, :171], axis=1)).max(axis=1)
    return np.concatenate([mean_all, std_all, q1, mid, q4, vel_max], axis=1).astype(np.float32)


class FSLPhraseNet(nn.Module):
    """Temporal Conv + Bidirectional GRU + Attentive-Max Pooling for 105 dynamic FSL signs."""

    def __init__(
        self,
        input_dim: int = FSL105_FEATURE_DIM,
        num_classes: int = FSL105_NUM_CLASSES,
        hidden_dim: int = 256,
        gru_hidden: int = 192,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        self.dropout = dropout

        # Input features (293) + first-order velocity on body-frame pose & hands (171) = 464
        in_with_vel = input_dim + 171
        self.spatial_in = nn.Sequential(
            nn.Linear(in_with_vel, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.spatial_res = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        half_dim = hidden_dim // 2
        self.conv3 = nn.Conv1d(hidden_dim, half_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden_dim, half_dim, kernel_size=5, padding=2)
        self.conv_norm = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn = nn.Linear(gru_hidden * 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden * 4),
            nn.Dropout(0.40),
            nn.Linear(gru_hidden * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(x[:, :, :171])
        vel[:, 1:, :] = x[:, 1:, :171] - x[:, :-1, :171]
        x_cat = torch.cat([x, vel], dim=-1)

        h = self.spatial_in(x_cat)
        h = h + self.spatial_res(h)

        h_t = h.transpose(1, 2)
        hc = self.conv_norm(torch.cat([self.conv3(h_t), self.conv5(h_t)], dim=1))
        h = (h_t + hc).transpose(1, 2)

        g, _ = self.gru(h)
        weights = torch.softmax(self.attn(g), dim=1)
        attn_pool = (g * weights).sum(dim=1)
        max_pool = g.max(dim=1).values
        pooled = torch.cat([attn_pool, max_pool], dim=-1)
        return self.head(pooled)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scale_sequences(
    X: np.ndarray, scaler: StandardScaler
) -> np.ndarray:
    """Scale continuous landmark channels [:, :, :291] while preserving presence flags [:, :, 291:]."""
    N, T, D = X.shape
    out = X.astype(np.float32).copy()
    cont = scaler.transform(out[:, :, :291].reshape(-1, 291)).reshape(N, T, 291)
    out[:, :, :291] = cont.astype(np.float32)
    return out


def _augment_batch(
    xb: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
) -> torch.Tensor:
    """Apply temporal sub-window crop/speed jitter, 2D rotation, scale/shift, noise, and frame drop."""
    bsz, seq_len, _ = xb.shape
    device = xb.device

    # 1. Temporal speed / sub-window crop jitter (p=0.6)
    x_aug = xb.clone()
    for i in range(bsz):
        if torch.rand(1, device=device).item() < 0.60:
            crop_ratio = 0.75 + 0.25 * torch.rand(1, device=device).item()
            win_len = max(12, int(round(seq_len * crop_ratio)))
            max_start = seq_len - win_len
            start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            seg = x_aug[i : i + 1, start : start + win_len, :].transpose(1, 2)  # (1, 293, win_len)
            res = F.interpolate(seg, size=seq_len, mode="linear", align_corners=True).transpose(1, 2)
            res[:, :, 291:] = (res[:, :, 291:] >= 0.5).to(res.dtype)
            x_aug[i] = res[0]

    # 2. Spatial rotation (±12 deg), scale (0.90-1.10), translation (±0.05), noise
    lh_flag = x_aug[:, :, 291:292]  # (B, T, 1)
    rh_flag = x_aug[:, :, 292:293]  # (B, T, 1)

    angles = (torch.rand(bsz, 1, 1, device=device) * 2.0 - 1.0) * (12.0 * np.pi / 180.0)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    scale = 0.90 + 0.20 * torch.rand(bsz, 1, 1, device=device)
    shift_xy = (torch.rand(bsz, 1, 1, 2, device=device) * 2.0 - 1.0) * 0.05

    def _transform_3d(block: torch.Tensor, apply_shift: bool) -> torch.Tensor:
        pts = block.view(bsz, seq_len, -1, 3)
        px = pts[:, :, :, 0]
        py = pts[:, :, :, 1]
        pz = pts[:, :, :, 2]
        rx = (px * cos_a - py * sin_a) * scale
        ry = (px * sin_a + py * cos_a) * scale
        rz = pz * scale
        out_pts = torch.stack([rx, ry, rz], dim=-1)
        if apply_shift:
            out_pts[:, :, :, :2] = out_pts[:, :, :, :2] + shift_xy
        return out_pts.view(bsz, seq_len, -1)

    pose_t = _transform_3d(x_aug[:, :, 0:45], apply_shift=True)
    pose_t = pose_t + torch.randn_like(pose_t) * 0.012

    lh_body_t = _transform_3d(x_aug[:, :, 45:108], apply_shift=True) * lh_flag
    lh_body_t = (lh_body_t + torch.randn_like(lh_body_t) * 0.015) * lh_flag

    rh_body_t = _transform_3d(x_aug[:, :, 108:171], apply_shift=True) * rh_flag
    rh_body_t = (rh_body_t + torch.randn_like(rh_body_t) * 0.015) * rh_flag

    lh_local_t = _transform_3d(x_aug[:, :, 171:231], apply_shift=False) * lh_flag
    lh_local_t = (lh_local_t + torch.randn_like(lh_local_t) * 0.015) * lh_flag

    rh_local_t = _transform_3d(x_aug[:, :, 231:291], apply_shift=False) * rh_flag
    rh_local_t = (rh_local_t + torch.randn_like(rh_local_t) * 0.015) * rh_flag

    cont = torch.cat([pose_t, lh_body_t, rh_body_t, lh_local_t, rh_local_t], dim=-1)

    # Scale continuous features, keep binary flags unscaled
    cont_scaled = (cont - scaler_mean) / scaler_std
    out = torch.cat([cont_scaled, x_aug[:, :, 291:]], dim=-1)

    # 3. Random temporal frame drop (mask 1-3 frames with p=0.30)
    for i in range(bsz):
        if torch.rand(1, device=device).item() < 0.30:
            n_drop = int(torch.randint(1, 4, (1,), device=device).item())
            drop_idx = torch.randperm(seq_len, device=device)[:n_drop]
            out[i, drop_idx, :] = 0.0

    return out


def _train_baseline_svm(
    X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> None:
    print("\nTraining Baseline Temporal-Summary RBF-SVM...")
    X_tr_feat = extract_temporal_summary_features(X_tr)
    X_val_feat = extract_temporal_summary_features(X_val)

    best_acc = -1.0
    best_c = 5.0
    best_pipe = None
    tuning_results = []

    n_comp = min(256, X_tr_feat.shape[0] - 1, X_tr_feat.shape[1])
    for c_val in (1.0, 5.0, 20.0, 50.0):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=RANDOM_SEED)),
            ("svc", SVC(C=c_val, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED)),
        ])
        pipe.fit(X_tr_feat, y_tr)
        val_acc = float(accuracy_score(y_val, pipe.predict(X_val_feat)))
        tuning_results.append({"C": c_val, "val_accuracy": round(val_acc, 4)})
        print(f"  SVM C={c_val:4.1f} -> val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_c = c_val
            best_pipe = pipe

    assert best_pipe is not None
    # Refit best config on full X_train (tr + val) for deployment parity? Keep tr+val separate or refit?
    # Keep trained on X_tr so validation comparison is apples-to-apples, or refit after tuning:
    # Actually refitting on full train split (X_tr + X_val) isn't strictly needed; keeping X_tr keeps val fair.
    joblib.dump(best_pipe, FSL105_SVM_PATH)
    (REPORTS_DIR / "fsl105_svm_tuning.json").write_text(
        json.dumps({"best_C": best_c, "best_val_acc": round(best_acc, 4), "trials": tuning_results}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Baseline FSL-105 SVM (C={best_c}, val_acc={best_acc:.4f}) -> {FSL105_SVM_PATH}")


def _eval_phrase_net(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += len(yb)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    _set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(FSL105_LANDMARKS_PATH)
    X_full_train = data["X_train"].astype(np.float32)
    y_full_train = data["y_train"].astype(np.int64)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full_train,
        y_full_train,
        test_size=0.15,
        stratify=y_full_train,
        random_state=RANDOM_SEED,
    )
    print(f"FSL-105 splits -> train={len(X_tr)}, val={len(X_val)} (official test={len(data['X_test'])} untouched)")

    # 1. Baseline SVM
    _train_baseline_svm(X_tr, y_tr, X_val, y_val)

    # 2. Fit Scaler on continuous channels [:291]
    scaler = StandardScaler()
    scaler.fit(X_tr[:, :, :291].reshape(-1, 291))
    scaler.scale_ = np.maximum(scaler.scale_, 1e-3)
    joblib.dump(scaler, FSL105_SCALER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scaler_std = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    X_val_scaled = _scale_sequences(X_val, scaler)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val)),
        batch_size=32,
        shuffle=False,
    )

    model = FSLPhraseNet(
        input_dim=FSL105_FEATURE_DIM,
        num_classes=FSL105_NUM_CLASSES,
        hidden_dim=256,
        gru_hidden=192,
        dropout=0.30,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    eval_criterion = nn.CrossEntropyLoss()
    base_lr = 1.5e-3
    epochs = 120
    warmup_epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=2e-2)

    def _lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        progress = float(ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    no_improve = 0
    patience = 25
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining FSLPhraseNet on {device} for up to {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb_aug = _augment_batch(xb, scaler_mean, scaler_std)

            use_mixup = torch.rand(1, device=device).item() < 0.25 and xb_aug.size(0) > 1
            optimizer.zero_grad()
            if use_mixup:
                lam = float(np.random.beta(0.3, 0.3))
                perm = torch.randperm(xb_aug.size(0), device=device)
                mixed_x = lam * xb_aug + (1.0 - lam) * xb_aug[perm]
                logits = model(mixed_x)
                loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb[perm])
            else:
                logits = model(xb_aug)
                loss = criterion(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            run_loss += loss.item() * len(yb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += len(yb)

        scheduler.step()
        train_loss = run_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = _eval_phrase_net(model, val_loader, eval_criterion, device)

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        improved = (val_acc > best_val_acc + 1e-4) or (
            abs(val_acc - best_val_acc) <= 1e-4 and val_loss < best_val_loss - 1e-4
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": FSL105_FEATURE_DIM,
                    "seq_len": FSL105_SEQ_LEN,
                    "num_classes": FSL105_NUM_CLASSES,
                    "hidden_dim": model.hidden_dim,
                    "gru_hidden": model.gru_hidden,
                    "dropout": model.dropout,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                },
                FSL105_MODEL_PATH,
            )
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve == 0:
            print(
                f"  Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    (REPORTS_DIR / "fsl105_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ep_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(ep_range, history["train_loss"], label="Train Loss")
    ax1.plot(ep_range, history["val_loss"], label="Val Loss")
    ax1.set_title("FSLPhraseNet Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(ep_range, history["train_acc"], label="Train Acc")
    ax2.plot(ep_range, history["val_acc"], label="Val Acc")
    ax2.set_title("FSLPhraseNet Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fsl105_training_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nBest FSLPhraseNet val_acc={best_val_acc:.4f} (val_loss={best_val_loss:.4f}) -> {FSL105_MODEL_PATH}")


if __name__ == "__main__":
    main()
