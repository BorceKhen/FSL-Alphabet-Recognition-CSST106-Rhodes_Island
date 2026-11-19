# Milestone 5 — Evaluation Results Summary

Reproduced via:

```bash
python -m src.evaluate
```

## Held-Out Test Set (`1,753` samples, 15% stratified split)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline RBF-SVM** (`C=50.0, gamma=0.05`) | 98.23% | 98.27% | 98.23% | 98.23% | 0.860 ms |
| **Deep MLP (Main, Augmented)** (`256-128-64`) | **98.52%** | **98.53%** | **98.52%** | **0.9852** | **0.255 ms** |

## Dataset & Landmark Extraction Coverage
- **Source**: Kaggle Filipino Sign Language (FSL) Alphabet Dataset (`11,700` RGB images, `26` classes `A–Z`, `450` per class)
- **MediaPipe Extraction Rate**: `11,683 / 11,700` (`99.85%`)
- **Stratified Split (`seed=42`)**: Train `8,178` (70%) / Validation `1,752` (15%) / Test `1,753` (15%)
