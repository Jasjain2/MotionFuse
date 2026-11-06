"""Load UTD-MHAD inertial (.mat) and RGB video (.avi) data for selected actions."""

import logging
from collections import Counter
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import scipy.io

from src import config

log = logging.getLogger(__name__)


def load_imu(mat_path: Path | str) -> np.ndarray:
    """Load inertial data as float32 array of shape [num_frames, 6]
    (accel_x/y/z, gyro_x/y/z)."""
    mat = scipy.io.loadmat(str(mat_path))
    if config.INERTIAL_MAT_KEY not in mat:
        raise KeyError(f"'{config.INERTIAL_MAT_KEY}' not in {mat_path}; "
                       f"keys: {[k for k in mat if not k.startswith('__')]}")
    imu = np.asarray(mat[config.INERTIAL_MAT_KEY], dtype=np.float32)
    if imu.ndim != 2 or imu.shape[1] != len(config.IMU_CHANNELS):
        raise ValueError(f"Unexpected IMU shape {imu.shape} in {mat_path}")
    return imu


def iter_video_frames(video_path: Path | str, stride: int = 1) -> Iterator[np.ndarray]:
    """Yield RGB frames (H, W, 3, uint8) from a video, every `stride`-th frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()


def load_video_frames(video_path: Path | str, stride: int = 1) -> np.ndarray:
    """Load all (strided) frames into an array of shape [T, H, W, 3]."""
    frames = list(iter_video_frames(video_path, stride))
    if not frames:
        raise IOError(f"No frames read from {video_path}")
    return np.stack(frames)


def video_info(video_path: Path | str) -> dict:
    """Return frame count, fps, width, height without decoding frames."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        return {
            "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def load_trials(subjects: list[int],
                actions: list[int] = config.SELECTED_ACTIONS,
                require_video: bool = True) -> list[dict]:
    """Load every available trial for the given subjects/actions.

    Each trial is a dict:
        {"action": int, "subject": int, "trial": int,
         "imu": np.ndarray [num_frames, 6], "video_path": str}

    Trials missing the inertial file (or video, if `require_video`) are skipped
    and reported.
    """
    trials, skipped = [], []
    for a in actions:
        for s in subjects:
            for t in config.TRIALS:
                mat_path = config.INERTIAL_DIR / config.INERTIAL_FILE_FMT.format(
                    action=a, subject=s, trial=t)
                vid_path = config.RGB_DIR / config.RGB_FILE_FMT.format(
                    action=a, subject=s, trial=t)
                if not mat_path.exists() or (require_video and not vid_path.exists()):
                    skipped.append((a, s, t))
                    continue
                trials.append({
                    "action": a,
                    "subject": s,
                    "trial": t,
                    "imu": load_imu(mat_path),
                    "video_path": str(vid_path),
                })
    if skipped:
        log.warning("Skipped %d missing trial(s): %s", len(skipped),
                    ", ".join(f"a{a}_s{s}_t{t}" for a, s, t in skipped))
    return trials


def load_split(require_video: bool = True) -> dict[str, list[dict]]:
    """Return {"train": [...], "test": [...]} using the subject split in config."""
    return {
        "train": load_trials(config.TRAIN_SUBJECTS, require_video=require_video),
        "test": load_trials(config.TEST_SUBJECTS, require_video=require_video),
    }


def print_summary(split: dict[str, list[dict]]) -> None:
    """Print trial counts per action, per subject, and train/test totals."""
    all_trials = split["train"] + split["test"]
    by_action = Counter(tr["action"] for tr in all_trials)
    by_subject = Counter(tr["subject"] for tr in all_trials)
    expected_per_action = len(config.ALL_SUBJECTS) * len(config.TRIALS)
    expected_per_subject = len(config.SELECTED_ACTIONS) * len(config.TRIALS)

    print("\n=== Trials per action ===")
    for a in config.SELECTED_ACTIONS:
        print(f"  a{a:<3} {config.ACTION_NAMES[a]:<18} "
              f"{by_action[a]:>3} / {expected_per_action}")

    print("\n=== Trials per subject ===")
    for s in config.ALL_SUBJECTS:
        tag = "train" if s in config.TRAIN_SUBJECTS else "test"
        print(f"  s{s} ({tag:<5}) {by_subject[s]:>3} / {expected_per_subject}")

    lengths = [tr["imu"].shape[0] for tr in all_trials]
    print("\n=== Totals ===")
    print(f"  train: {len(split['train'])}  (subjects {config.TRAIN_SUBJECTS})")
    print(f"  test:  {len(split['test'])}  (subjects {config.TEST_SUBJECTS})")
    print(f"  total: {len(all_trials)}")
    if lengths:
        print(f"  IMU length (samples @ {config.IMU_SAMPLE_RATE} Hz): "
              f"min {min(lengths)}, median {int(np.median(lengths))}, max {max(lengths)}")


if __name__ == "__main__":
    print_summary(load_split())
