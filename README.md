# KUMPAS: Two-Way Filipino Sign Language (FSL) Communication System

A real-time computer vision communication bridge designed to enable direct, effortless two-way interaction between Deaf/hard-of-hearing FSL signers and non-signing hearing individuals.

Upgraded from the baseline A–Z fingerspelling prototype, **KUMPAS** now combines:
1. **Dynamic FSL-105 Word & Phrase Recognition (`FSLPhraseNet`)**: Recognizes **105 everyday Filipino Sign Language words & phrases** across 10 practical communication categories (*Greetings, Survival, Numbers, Calendar, Days, Family, Relationships, Colors, Food, Drinks*) from upper-body pose and two-hand MediaPipe Holistic sequences (`97.89%` Top-1 / `99.77%` Top-3 held-out test accuracy, `7.235 ms` CPU latency).
2. **A–Z Fingerspelling + Bilingual Predictive Autocomplete**: Preserves the proposal's 26-letter MediaPipe `HandLandmarker` + PyTorch Deep MLP pipeline (`98.52%` test accuracy, `0.160–0.287 ms` CPU latency) with `τ = 0.75`, 10-frame majority voting, `1.0 s` dwell-time confirmation, `2.0 s` auto-space, and instant English/Filipino word completion chips for names, places, and healthcare/pharmacy terms.
3. **Natural EN/FIL Sentence Composition & Auto-Speak TTS**: Converts signed glosses and fingerspelled words into natural English or Filipino (Tagalog) sentences and speaks them aloud automatically.
4. **Two-Way Hearing Listener Bridge & Fullscreen Display Card**: Hearing listeners can reply via **Voice-to-Text microphone** (Web Speech API) or quick-reply/text input, with automatic FSL-105 sign lookup pills attached to replies for the Deaf user, plus a high-contrast fullscreen listener card for noisy or counter environments.
5. **Proposal-Aligned Low-Light Resilience & On-Demand Sign Guide**: Automatic CLAHE contrast fallback activates whenever standard tracking misses a hand, while the A–Z and 105-phrase visual references live inside a searchable **Sign Guide** modal rather than cluttering the communication workspace.

---

## Requirements & Setup

- Python `3.10+` (verified on Python `3.12.3`)
- Webcam (for live signing; uploaded video/image files and the built-in held-out **Test Clip** picker work without a camera)
- Packages in [`requirements.txt`](./requirements.txt) (note: `mediapipe==0.10.21` requires `protobuf>=4.25.3,<5`)

```powershell
git clone https://github.com/BorceKhen/FSL-Alphabet-Recognition-CSST106-Rhodes_Island.git
cd FSL-Alphabet-Recognition-CSST106-Rhodes_Island
python -m venv --system-site-packages .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## How to Run the Application

### Option 1: Web Communication UI (Recommended)
```powershell
.venv\Scripts\python.exe -m src.web_app
```
Open **http://127.0.0.1:5000** in your browser (or double-click `run_app.bat` on Windows).
- **Phrase Mode (`105 Signs`)**: Start the camera (or click **Test Clip** / **Upload**) to sign everyday FSL phrases, compose natural EN/FIL sentences, speak aloud, and exchange two-way replies.
- **Fingerspell Mode (`A–Z`)**: Hold a steady alphabet sign for `1.0 s` and tap predictive autocomplete chips to spell names or custom words rapidly.

### Option 2: Unified CLI Pipeline (`src/run.py`)
```powershell
# Dynamic FSL-105 phrase recognition on video or webcam
.venv\Scripts\python.exe src/run.py --mode phrase --input "data\FSL-105 A dataset for recognizing 105 Filipino sign language videos\clips\0\17.MOV" --output outputs/demo_phrase.mp4
.venv\Scripts\python.exe src/run.py --mode phrase --webcam 0

# A-Z fingerspelling on image, folder, video, or webcam
.venv\Scripts\python.exe src/run.py --mode alphabet --input path/to/images_or_image.jpg --output outputs/run_test/
.venv\Scripts\python.exe src/run.py --mode alphabet --webcam 0
```

### Option 3: PyQt6 Desktop UI (A–Z)
```powershell
python -m src.app
```

### Run Unit & Integration Tests
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## Reproduced Evaluation Results

### 1. FSL-105 Dynamic Phrase & Word Recognition (`426` held-out `test.csv` clips, `105` classes)
Reproduce with:
```powershell
.venv\Scripts\python.exe -m src.evaluate_fsl105
```

| Model | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Macro F1 | CPU Latency | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Temporal-Summary RBF-SVM** (`PCA-256, C=20.0`) | 92.96% | 97.42% | 97.89% | 0.9291 | **1.479 ms** | **7,029.4 KB** |
| **FSLPhraseNet (Main: Conv1D + BiGRU + Attn-Max)** | **97.89%** | **99.77%** | **99.77%** | **0.9789** | 7.235 ms | 7,276.4 KB |

### 2. FSL Alphabet A–Z Fingerspelling (`1,753` held-out test samples, `26` classes)
Reproduce with:
```powershell
.venv\Scripts\python.exe -m src.evaluate --split test --weights models/mlp_model.pt
```

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | CPU Latency | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline RBF-SVM** (`C=50.0, gamma=0.05`) | 98.23% | 98.27% | 98.23% | 0.9823 | 0.466 ms | 2,183.8 KB |
| **Deep MLP (Main, Augmented)** (`256-128-64`) | **98.52%** | **98.53%** | **98.52%** | **0.9852** | **0.160 ms** | **246.0 KB** |

---

## Project Structure

```text
FSL/
├── README.md
├── requirements.txt
├── run_app.bat
├── src/
│   ├── config.py             # Paths, A-Z & FSL-105 hyperparameters and thresholds
│   ├── features.py           # HandLandmarker (63-dim), Holistic (293-dim), auto-CLAHE fallback
│   ├── prepare_dataset.py    # A-Z Kaggle landmark extraction & stratified splits
│   ├── prepare_fsl105.py     # FSL-105 Holistic sequence & thumbnail extraction
│   ├── train_baseline.py     # A-Z Baseline RBF-SVM training
│   ├── train_mlp.py          # A-Z PyTorch Deep MLP training
│   ├── train_fsl105.py       # FSL-105 Baseline RBF-SVM & FSLPhraseNet training
│   ├── evaluate.py           # A-Z test-set metrics & confusion matrices
│   ├── evaluate_fsl105.py    # FSL-105 test-set metrics & category charts
│   ├── predictor.py          # Shared A-Z ModelPredictor
│   ├── phrase_predictor.py   # FSL-105 PhrasePredictor
│   ├── translator.py         # 105-sign EN/FIL sentence composer, autocomplete & listener matcher
│   ├── postprocess.py        # WordFormationProcessor & PhraseSignProcessor
│   ├── run.py                # Unified CLI (--mode phrase | alphabet)
│   ├── app.py                # PyQt6 desktop GUI
│   ├── web_app.py            # Flask Two-Way Communication backend
│   ├── templates/index.html  # Two-way KUMPAS web interface
│   └── static/               # CSS & JS client logic
├── tests/
│   └── test_pipeline.py      # Unit & Web API integration tests
├── data/                     # Cached landmarks (fsl_landmarks.npz, fsl105_landmarks.npz) & thumbnails
├── models/                   # Saved PyTorch/joblib weights, scalers, label maps
├── reports/                  # Raw JSON metrics, curves, confusion matrices, error logs
└── outputs/final/            # Final deliverables, charts, and summary tables
```

---

## Team & Contributions

**Team Members**: John Benedict M. Hornilla, Johann Francois P. Tanyag, Ron Ken S. Borce
