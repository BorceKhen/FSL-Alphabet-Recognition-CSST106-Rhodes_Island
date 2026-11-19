"""Train the Deep MLP Neural Network on 63-dim normalized hand landmarks."""

from __future__ import annotations

import json
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DATA_DIR,
    FEATURE_DIM,
    MLP_BATCH_SIZE,
    MLP_DROPOUT,
    MLP_EPOCHS,
    MLP_HIDDEN_DIMS,
    MLP_LR,
    MLP_PATIENCE,
    MLP_WEIGHT_DECAY,
    MODELS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
    REPORTS_DIR,
)


class FSLDeepMLP(nn.Module):
    """Deep Multi-Layer Perceptron with BatchNorm, ReLU, and Dropout."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: tuple[int, ...] = MLP_HIDDEN_DIMS,
        num_classes: int = NUM_CLASSES,
        dropout: float = MLP_DROPOUT,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_loader(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
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
    return total_loss / total, correct / total


def _augment_landmarks(xb: torch.Tensor, scaler_mean: torch.Tensor, scaler_std: torch.Tensor) -> torch.Tensor:
    """Apply small 2D rotations (±15 deg), scale jitter, and Gaussian noise to raw 63-dim landmarks."""
    bsz = xb.shape[0]
    pts = xb.view(bsz, 21, 3)

    angles = (torch.rand(bsz, device=xb.device) * 2.0 - 1.0) * (15.0 * np.pi / 180.0)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x = pts[:, :, 0]
    y = pts[:, :, 1]
    z = pts[:, :, 2]

    x_rot = x * cos_a.unsqueeze(1) - y * sin_a.unsqueeze(1)
    y_rot = x * sin_a.unsqueeze(1) + y * cos_a.unsqueeze(1)

    scale = 0.95 + 0.10 * torch.rand(bsz, 1, device=xb.device)
    pts_aug = torch.stack([x_rot * scale, y_rot * scale, z * scale], dim=-1)
    pts_aug = pts_aug + torch.randn_like(pts_aug) * 0.012
    flat = pts_aug.view(bsz, FEATURE_DIM)
    return (flat - scaler_mean) / scaler_std


def main() -> None:
    _set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(DATA_DIR / "fsl_landmarks.npz")

    scaler = StandardScaler()
    scaler.fit(data["X_train"])
    X_train_raw = data["X_train"].astype(np.float32)
    X_val = scaler.transform(data["X_val"]).astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_val = data["y_val"].astype(np.int64)

    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scaler_std = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    joblib.dump(scaler, MODELS_DIR / "mlp_scaler.joblib")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_raw), torch.from_numpy(y_train)),
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=MLP_BATCH_SIZE,
        shuffle=False,
    )

    model = FSLDeepMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"Training FSLDeepMLP on {device} ({len(X_train_raw)} train / {len(X_val)} val) with landmark augmentation...")
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = _augment_landmarks(xb, scaler_mean, scaler_std)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(yb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += len(yb)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = _evaluate_loader(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": FEATURE_DIM,
                    "hidden_dims": MLP_HIDDEN_DIMS,
                    "num_classes": NUM_CLASSES,
                    "dropout": MLP_DROPOUT,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                },
                MODELS_DIR / "mlp_model.pt",
            )
        else:
            epochs_without_improve += 1

        if epoch % 10 == 0 or epochs_without_improve == 0:
            print(
                f"  Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        if epochs_without_improve >= MLP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    (REPORTS_DIR / "mlp_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss")
    ax1.set_title("Deep MLP Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    ax2.plot(epochs_range, history["train_acc"], label="Train Acc")
    ax2.plot(epochs_range, history["val_acc"], label="Val Acc")
    ax2.set_title("Deep MLP Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "mlp_training_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nBest Deep MLP val_acc={best_val_acc:.4f} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
