"""Central configuration: dataset selection, paths, and hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "utd-mhad"
INERTIAL_DIR = DATA_DIR / "Inertial"
RGB_DIR = DATA_DIR / "RGB"
POSE_CACHE_DIR = PROJECT_ROOT / "data" / "pose_cache"  # raw MediaPipe landmarks (.npy)
MODEL_DIR = PROJECT_ROOT / "data" / "models"            # MediaPipe .task files
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
SELECTED_ACTIONS = [5, 7, 12, 13, 14, 15, 17, 20]  # the 8 athletic actions
ACTION_NAMES = {
    5: "throw",
    7: "basketball_shoot",
    12: "bowling",
    13: "boxing",
    14: "baseball_swing",
    15: "tennis_forehand",
    17: "tennis_serve",
    20: "catch",
}
# Contiguous class indices 0..N-1 for the classifier.
ACTION_TO_LABEL = {a: i for i, a in enumerate(SELECTED_ACTIONS)}
LABEL_TO_ACTION = {i: a for a, i in ACTION_TO_LABEL.items()}

ALL_SUBJECTS = list(range(1, 9))
TRAIN_SUBJECTS = [1, 2, 3, 4, 5]
TEST_SUBJECTS = [6, 7, 8]
TRIALS = [1, 2, 3, 4]

# File naming convention in UTD-MHAD
INERTIAL_FILE_FMT = "a{action}_s{subject}_t{trial}_inertial.mat"
RGB_FILE_FMT = "a{action}_s{subject}_t{trial}_color.avi"
INERTIAL_MAT_KEY = "d_iner"  # [num_frames, 6]: accel_x/y/z, gyro_x/y/z
IMU_CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]

# ---------------------------------------------------------------------------
# Sensor properties
# ---------------------------------------------------------------------------
IMU_SAMPLE_RATE = 50  # Hz
VIDEO_FPS = 15
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
IMU_LOWPASS_CUTOFF = 20.0     # Hz, Butterworth low-pass before features
IMU_FILTER_ORDER = 4
POSE_MODEL_COMPLEXITY = 2     # MediaPipe Pose: 0, 1, 2 (= lite/full/heavy Tasks model)
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF = 0.5
POSE_MAX_FAIL_FRAC = 0.3      # trials with more failed frames are flagged unreliable

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
HIDDEN_DIMS = [256, 128]
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20  # epochs without val-loss improvement
# Subject-wise validation split inside TRAIN_SUBJECTS, used for early stopping
# (and fitting scalers on the remaining subjects only).
VAL_SUBJECTS = [5]
FIT_SUBJECTS = [s for s in TRAIN_SUBJECTS if s not in VAL_SUBJECTS]
SEEDS = [42, 43, 44, 45, 46]
MODELS_DIR = RESULTS_DIR / "models"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 1000            # bootstrap resamples for confidence intervals
CI_LEVEL = 0.95
N_SEEDS = 5                   # repeated runs for seed-variance estimates

RANDOM_SEED = 42
