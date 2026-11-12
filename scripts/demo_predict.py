"""Predict the action for one trial with the production fused Random Forest.

Usage: python scripts/demo_predict.py [trial_name]   (default: a15_s7_t1)
"""

import re
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_loader import load_imu  # noqa: E402
from src.imu_features import extract_imu_features  # noqa: E402
from src.normalization import extract_pose_features, normalize_pose  # noqa: E402
from src.pose_extraction import get_trial_pose  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "a15_s7_t1"
match = re.fullmatch(r"a(\d+)_s(\d+)_t(\d+)", name)
if not match:
    sys.exit(f"Bad trial name '{name}', expected e.g. a15_s7_t1")
a, s, t = map(int, match.groups())
if a not in config.ACTION_NAMES:
    sys.exit(f"Action {a} is not one of {config.SELECTED_ACTIONS}")

rf = joblib.load(config.MODELS_DIR / "rf_fused_production.joblib")
scalers = joblib.load(config.MODELS_DIR / "scalers.joblib")

fmt = dict(action=a, subject=s, trial=t)
trial = {**fmt, "video_path": str(config.RGB_DIR / config.RGB_FILE_FMT.format(**fmt))}
imu = extract_imu_features(load_imu(config.INERTIAL_DIR / config.INERTIAL_FILE_FMT.format(**fmt)))
pose = extract_pose_features(normalize_pose(get_trial_pose(trial)[0]))

x = np.hstack([scalers["imu"].transform(imu[None]), scalers["pose"].transform(pose[None])])
proba = rf.predict_proba(x.astype(np.float32))[0]
names = [config.ACTION_NAMES[config.LABEL_TO_ACTION[c]] for c in rf.classes_]
best = int(np.argmax(proba))

print(f"Trial:      {name}")
print(f"True:       {config.ACTION_NAMES[a]}")
print(f"Predicted:  {names[best]}")
print(f"Confidence: {proba[best]:.2f}")
print("All classes:")
for i in np.argsort(proba)[::-1]:
    print(f"  {names[i]:<18} {proba[i]:.2f}")
