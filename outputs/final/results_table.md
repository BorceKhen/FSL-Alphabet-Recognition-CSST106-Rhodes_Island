# KUMPAS — Final Evaluation Results Summary

## 1. FSL-105 Dynamic Phrase & Word Recognition (`426` held-out `test.csv` clips, `105` classes)

Reproduced via:
```powershell
.venv\Scripts\python.exe -m src.evaluate_fsl105
```

| Model | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Macro F1 | CPU Latency | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Temporal-Summary RBF-SVM** (`PCA-256, C=20.0`) | 92.96% | 97.42% | 97.89% | 0.9291 | **1.479 ms** | **7,029.4 KB** |
| **FSLPhraseNet (Main: Conv1D + BiGRU + Attn-Max)** | **97.89%** | **99.77%** | **99.77%** | **0.9789** | 7.235 ms | 7,276.4 KB |

- **Extraction Coverage (`Holistic model_complexity=1`, `40 × 293` normalized sequences)**:
  - Train split: `1,703 / 1,704` (`99.94%` active hand detection)
  - Official test split: `426 / 426` (`100.00%` active hand detection)

---

## 2. FSL Alphabet A–Z Fingerspelling (`1,753` held-out test samples, `26` classes)

Reproduced via:
```powershell
.venv\Scripts\python.exe -m src.evaluate --split test --weights models/mlp_model.pt
```

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | CPU Latency (Warm-Up) | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline RBF-SVM** (`C=50.0, gamma=0.05`) | 98.23% | 98.27% | 98.23% | 98.23% | 0.466 ms | 2,183.8 KB |
| **Deep MLP (Main, Augmented)** (`256-128-64`) | **98.52%** | **98.53%** | **98.52%** | **0.9852** | **0.160 ms** | **246.0 KB** |

- **Source**: Kaggle Filipino Sign Language (FSL) Alphabet Dataset (`11,700` RGB images, `26` classes `A–Z`, `450` per class, CC0 Public Domain)
- **MediaPipe Extraction Rate**: `11,683 / 11,700` (`99.85%`)
- **Stratified Split (`seed=42`)**: Train `8,178` (70%) / Validation `1,752` (15%) / Test `1,753` (15%, frozen)
