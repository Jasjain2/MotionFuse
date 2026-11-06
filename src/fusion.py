"""Fuse IMU and pose features: per-modality standardization + concatenation.

Each modality gets its own StandardScaler fit on training rows only, so the
raw scale gap (gyro in deg/s vs normalized pose coordinates) doesn't let one
modality dominate, and no test statistics leak into preprocessing.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_loader import load_split
from src.imu_features import extract_imu_features_batch
from src.normalization import extract_pose_features_batch


def _scale(train: np.ndarray, *others: np.ndarray | None):
    scaler = StandardScaler().fit(train)
    out = [scaler.transform(train).astype(np.float32)]
    out += [None if o is None else scaler.transform(o).astype(np.float32) for o in others]
    return scaler, out


def concatenate_features(imu_train: np.ndarray, imu_test: np.ndarray,
                         pose_train: np.ndarray, pose_test: np.ndarray,
                         imu_val: np.ndarray | None = None,
                         pose_val: np.ndarray | None = None) -> dict:
    """Standardize each modality on its training rows, then concatenate.

    Optional `*_val` arrays (e.g. a held-out validation subject) are
    transformed with the same training-fit scalers.

    Returns a dict with keys "imu", "pose", "fused", each mapping to
    {"train", "val", "test"} scaled arrays ("val" is None if not given;
    fused is [n, 76 + 396 = 472]), plus "scalers": {"imu", "pose"}.
    """
    for name, a, b in [("train", imu_train, pose_train), ("test", imu_test, pose_test),
                       ("val", imu_val, pose_val)]:
        if (a is None) != (b is None) or (a is not None and len(a) != len(b)):
            raise ValueError(f"IMU and pose {name} sets must have matching rows")

    imu_scaler, (imu_tr, imu_te, imu_va) = _scale(imu_train, imu_test, imu_val)
    pose_scaler, (pose_tr, pose_te, pose_va) = _scale(pose_train, pose_test, pose_val)

    def cat(a, b):
        return None if a is None else np.hstack([a, b])

    return {
        "imu": {"train": imu_tr, "val": imu_va, "test": imu_te},
        "pose": {"train": pose_tr, "val": pose_va, "test": pose_te},
        "fused": {"train": cat(imu_tr, pose_tr), "val": cat(imu_va, pose_va),
                  "test": cat(imu_te, pose_te)},
        "scalers": {"imu": imu_scaler, "pose": pose_scaler},
    }


def load_features(groups: dict[str, list[int]] | None = None) -> dict:
    """Unscaled IMU + pose features and labels per subject group.

    groups maps group name -> subjects; default is fit (1-4) / val (5) /
    test (6-8) from config. Returns {group: {"imu", "pose", "y", "meta"}},
    meta being the per-trial dicts from extract_pose_features_batch.
    """
    split = load_split()
    trials = split["train"] + split["test"]
    X_imu, y, meta = extract_imu_features_batch(trials)
    X_pose, y_pose, meta_pose = extract_pose_features_batch(trials)
    key = [(m["action"], m["subject"], m["trial"]) for m in meta]
    assert key == [(m["action"], m["subject"], m["trial"]) for m in meta_pose]
    assert (y == y_pose).all()

    subjects = np.array([m["subject"] for m in meta])
    if groups is None:
        groups = {"fit": config.FIT_SUBJECTS, "val": config.VAL_SUBJECTS,
                  "test": config.TEST_SUBJECTS}
    out = {}
    for g, subs in groups.items():
        mask = np.isin(subjects, subs)
        out[g] = {"imu": X_imu[mask], "pose": X_pose[mask], "y": y[mask],
                  "meta": [m for m, keep in zip(meta_pose, mask) if keep]}
    return out
