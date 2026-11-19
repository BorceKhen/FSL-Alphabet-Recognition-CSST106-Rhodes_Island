"""Central configuration for the FSL Alphabet Recognition System."""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(
    r"C:\Users\kentr\OneDrive\Desktop\CS Stuff\BS ComSci (4th Year)\Computer Vision\FSL Dataset\Collated"
)
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Classes A-Z
CLASSES = [chr(ord("A") + i) for i in range(26)]
NUM_CLASSES = len(CLASSES)
FEATURE_DIM = 63  # 21 hand landmarks * (x, y, z)

# Dataset split ratios (stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Camera & Preprocessing
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Post-Processing & Word Formation
CONFIDENCE_THRESHOLD = 0.75  # tau
SMOOTHING_WINDOW = 10        # frames for sliding-window majority vote
DWELL_TIME_SECONDS = 1.0     # steady hold duration before confirming a letter
SPACE_PAUSE_SECONDS = 2.0    # optional neutral/no-hand pause to insert space

# Training hyperparameters
MLP_HIDDEN_DIMS = (256, 128, 64)
MLP_DROPOUT = 0.30
MLP_BATCH_SIZE = 64
MLP_EPOCHS = 120
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_PATIENCE = 15
