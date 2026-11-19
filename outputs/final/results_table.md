# Milestone 5 — Final Evaluation Results Summary

Reproduced from `v1.0` via:

```bash
python -m src.evaluate --split test --weights models/mlp_model.pt
```

## Held-Out Test Set (`1,753` samples, 15% stratified split)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | CPU Latency (Warm-Up) | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline RBF-SVM** (`C=50.0, gamma=0.05`) | 98.23% | 98.27% | 98.23% | 98.23% | 1.040 ms | 2,183.8 KB |
| **Deep MLP (Main, Augmented)** (`256-128-64`) | **98.52%** | **98.53%** | **98.52%** | **0.9852** | **0.287 ms** | **246.0 KB** |

## Dataset & Landmark Extraction Coverage
- **Source**: Kaggle Filipino Sign Language (FSL) Alphabet Dataset (`11,700` RGB images, `26` classes `A–Z`, `450` per class, CC0 Public Domain)
- **MediaPipe Extraction Rate**: `11,683 / 11,700` (`99.85%`)
- **Stratified Split (`seed=42`)**: Train `8,178` (70%) / Validation `1,752` (15%) / Test `1,753` (15%, frozen)
