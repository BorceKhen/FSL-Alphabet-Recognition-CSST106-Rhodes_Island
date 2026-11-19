# Dataset Card — Filipino Sign Language (FSL) Alphabet Dataset

## 1. Source & License
- **Dataset**: [Filipino Sign Language (FSL) Dataset on Kaggle](https://www.kaggle.com/datasets/japorton/fsl-dataset) by John Gabriel Porton
- **License**: Creative Commons Zero (`CC0: Public Domain`)

## 2. Dataset Size, Classes & Splits
- **Total Raw Images**: `11,700` RGB `.jpg` images across `26` alphabet classes (`A`–`Z`), `450` images per class.
- **Image Resolution**: `300 × 300` pixels.
- **MediaPipe Landmark Extraction Coverage**: `11,683 / 11,700` (`99.85%`) valid 21-keypoint hand detections (`17` images skipped across `A`, `J`, `M`, `O`, `Q`, `X`, `Z` due to severe motion blur or extreme cropping).
- **Stratified Split (`random_state=42`, locked before tuning)**:
  - **Training Set (70%)**: `8,178` samples
  - **Validation Set (15%)**: `1,752` samples
  - **Test Set (15%, held out)**: `1,753` samples

## 3. Labeling & Verification
- Images are pre-labeled by directory name (`A` through `Z`).
- Labels and visual quality were verified via a 20-class contact sheet (`A`–`T`) during Milestone 1 and automated per-class landmark extraction counts (`reports/dataset_extraction_summary.json`).

## 4. Environmental Conditions & Known Gaps
- **Covered**: Indoor daylight, overhead fluorescent lighting, mild shadows, varied indoor backgrounds, and minor hand orientation/scale differences.
- **Known Gaps**:
  - Subtle finger-placement overlap among closed-fist letters (`A`, `E`, `M`, `N`, `S`, `T`).
  - Side-profile orientation bias in letters like `D`, which can look closer to `O`, `Z`, or `I` when signed flat toward a front-facing webcam.
  - Static-frame representation for letters `J` and `Z` (which involve motion in continuous signing).

## 5. Preprocessing Applied
1. Color conversion (`BGR` → `RGB`), with optional `LAB` `L`-channel **CLAHE** and `20%` border-replicate padding fallback for tight crops / low contrast.
2. **MediaPipe `HandLandmarker`** extraction of `21` 3D keypoints `(x, y, z)`.
3. **Wrist-relative translation** (subtracting Landmark `0`) and **bounding-box scale normalization** (dividing by `max(x_span, y_span)`), yielding a **63-dimensional** invariant feature vector saved to `data/fsl_landmarks.npz`.
