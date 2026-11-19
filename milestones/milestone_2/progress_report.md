# Milestone 2 — 50% Progress Check Report

**Project**: Computer Vision-Based Filipino Sign Language (FSL) Alphabet Recognition System for Text-Based Communication  
**Checkpoint Tag**: `cp2-50`  
**Repository**: https://github.com/BorceKhen/FSL-Alphabet-Recognition-CSST106-Rhodes_Island

---

## 1. Progress Table

| Task (from Proposal) | Owner | Status | Evidence (File / Artifact / Metric) |
| :--- | :--- | :--- | :--- |
| Project proposal & technical specification | Systems Lead / Technical Writer | Done | Proposal document & `src/config.py` |
| Dataset acquisition, structuring & 70/15/15 split | Lead ML Engineer | Done | `data/README.md`, `data/fsl_landmarks.npz`, `reports/dataset_extraction_summary.json` (`11,683 / 11,700` detected, `99.85%`) |
| Preprocessing & 63-dim landmark normalization | Lead ML Engineer | Done | `src/features.py` (`apply_clahe`, `normalize_landmarks`, `HandLandmarkExtractor`), `milestones/milestone_2/preprocessing_strip.png` |
| Baseline RBF-SVM model & validation evaluation | ML / CV Specialist | Done | `src/train_baseline.py`, `models/svm_baseline.joblib`, `milestones/milestone_2/baseline_val_metrics.json` (`val_acc = 0.9840`, `val_macro_f1 = 0.9840`) |
| Single-command prototype (`predict.py`) | Backend / Full-Stack | Done | `src/predict.py`, `milestones/milestone_2/pred_sample_D.png` (`D`, `98.32%`) |
| Deep MLP neural network training | ML / CV Specialist | Pending (Milestone 3) | `src/train_mlp.py` |
| Confidence thresholding, 10-frame smoothing & 1.0s dwell | Backend Developer | Pending (Milestone 3–4) | `src/postprocess.py` |
| Real-time UI & system benchmark testing | Full-Stack / UI Developer | Pending (Milestone 4–5) | `src/web_app.py`, `src/app.py` |

---

## 2. Percentage Complete & Justification
- **Milestone 2 Target**: `50%` (Environment + Clean Split Dataset + Reusable Preprocessing + Baseline Validation Metric + Single-Command Prototype).
- **Achieved Progress**: **55% complete** (`5 / 8` core modules completed, covering 100% of the Milestone 2 foundation tasks).
- **Blockers & Mitigation**:
  - `17 / 11,700` dataset images failed initial MediaPipe detection due to tight cropping or motion blur; mitigated via `20%` border-replicate padding and `LAB` `L`-channel CLAHE fallback (`99.85%` coverage).
  - Live webcam orientation differences (e.g., side-profile `D` vs. front-facing camera) will be addressed in Milestone 3 via `±15°` rotation and jitter augmentation.
- **Target Dates**:
  - Core Deep MLP & Testing (Milestone 3): Oct 29
  - Post-Processing & Live UI Integration (Milestone 4): Nov 12
  - Final Presentation & Defense (Milestone 5): Nov 19

---

## 3. Reproducing Milestone 2 Deliverables

1. **Run the single-command prototype**:
   ```powershell
   python src/predict.py --input milestones/milestone_2/sample_D.jpg --output milestones/milestone_2/
   ```
2. **Inspect the before/after preprocessing strip**:
   - [`milestones/milestone_2/preprocessing_strip.png`](./preprocessing_strip.png)
3. **Verify Baseline RBF-SVM validation numbers**:
   - [`milestones/milestone_2/baseline_val_metrics.json`](./baseline_val_metrics.json) (`val_accuracy: 0.9840`, `val_macro_f1: 0.9840` on `1,752` validation samples)
4. **Read the 1-page Dataset Card**:
   - [`data/README.md`](../../data/README.md)

---

## 4. Prepared Technical Q&A (In-Class Review)

- **"Why 63-dim MediaPipe landmarks instead of raw RGB pixels?"**
  Raw pixels are sensitive to background clutter, skin tone, and lighting. Extracting 21 3D hand keypoints `(x, y, z)` and normalizing them relative to the wrist (`Landmark 0`) and bounding-box span produces a compact 63-dim geometric vector that is invariant to hand position and camera distance.
- **"Which split did your 98.40% baseline number come from?"**
  From the **15% validation split** (`1,752` samples) during RBF-SVM hyperparameter tuning (`C=50.0, gamma=0.05`). The 15% test split (`1,753` samples) remained locked and untouched during tuning.
- **"What does CLAHE do if the photo is darker?"**
  `apply_clahe()` converts the frame to `LAB` color space and applies Contrast Limited Adaptive Histogram Equalization only to the `L` (lightness) channel, boosting local hand contrast in dim or backlit scenes without distorting hue.
