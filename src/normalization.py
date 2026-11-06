"""Coordinate frame normalization and feature extraction for pose sequences.

normalize_pose: per frame, translate so mid-hip is the origin and divide by
torso length (mid-hip -> mid-shoulder). The result is invariant to where the
person stands in the frame and how far they are from the camera.

extract_pose_features: [num_frames, 33, 3] -> 396-dim vector
    198  position mean, std    (33 landmarks x 3 coords x 2)
    198  velocity mean, std    (frame-to-frame deltas, 33 x 3 x 2)
"""

import logging

import numpy as np

from src import config
from src.pose_extraction import LANDMARK_NAMES, N_LANDMARKS, extract_poses

log = logging.getLogger(__name__)

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
# A torso shorter than this fraction of the trial's median is a detection artifact.
MIN_TORSO_FRAC = 0.1
COORDS = ["x", "y", "z"]
N_POSE_FEATURES = 4 * N_LANDMARKS * len(COORDS)


def normalize_pose(pose: np.ndarray,
                   aspect_ratio: float | None = config.VIDEO_HEIGHT / config.VIDEO_WIDTH
                   ) -> np.ndarray:
    """Hip-centred, torso-scaled pose of the same shape [num_frames, 33, 3].

    MediaPipe normalizes x by image width and y by image height, so on a 4:3
    video one unit of y is shorter than one unit of x. `aspect_ratio` (H / W)
    rescales y into x units first so distances are geometrically correct.
    Pass None to skip.
    """
    pose = np.asarray(pose, dtype=np.float64).copy()
    if pose.ndim != 3 or pose.shape[1:] != (N_LANDMARKS, 3):
        raise ValueError(f"Expected [num_frames, 33, 3], got {pose.shape}")
    if aspect_ratio is not None:
        pose[..., 1] *= aspect_ratio

    mid_hip = pose[:, [LEFT_HIP, RIGHT_HIP]].mean(axis=1, keepdims=True)  # [T, 1, 3]
    pose -= mid_hip

    mid_shoulder = pose[:, [LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=1)   # [T, 3]
    torso = np.linalg.norm(mid_shoulder, axis=1)                          # [T]
    bad = torso < max(1e-6, MIN_TORSO_FRAC * np.median(torso))
    if bad.all():
        log.warning("Torso length ~0 in every frame; returning translated pose unscaled")
        return pose.astype(np.float32)
    torso[bad] = np.median(torso[~bad])
    return (pose / torso[:, None, None]).astype(np.float32)


def pose_feature_names() -> list[str]:
    """Names in the same order as extract_pose_features output."""
    base = [f"{lm}_{c}" for lm in LANDMARK_NAMES for c in COORDS]
    return [f"{b}_{kind}" for kind in ["pos_mean", "pos_std", "vel_mean", "vel_std"]
            for b in base]


def extract_pose_features(norm_pose: np.ndarray) -> np.ndarray:
    """Map a normalized pose sequence [num_frames, 33, 3] to a vector [396]."""
    pos = np.asarray(norm_pose, dtype=np.float64).reshape(len(norm_pose), -1)  # [T, 99]
    vel = np.diff(pos, axis=0) if len(pos) > 1 else np.zeros_like(pos)       # per-frame delta
    feats = np.concatenate([pos.mean(axis=0), pos.std(axis=0),
                            vel.mean(axis=0), vel.std(axis=0)])
    assert feats.shape == (N_POSE_FEATURES,)
    return feats.astype(np.float32)


def extract_pose_features_batch(trials: list[dict], use_cache: bool = True
                                ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run (cached) pose extraction, normalization and feature extraction.

    Returns:
        X_pose:   [num_trials, 396] float32
        y:        [num_trials] int64 class indices (config.ACTION_TO_LABEL)
        metadata: per-row {"action", "subject", "trial", "pose_num_frames",
                  "pose_num_failed", "pose_fail_rate", "pose_unreliable"}
    """
    poses, infos = extract_poses(trials, use_cache=use_cache)
    X = np.stack([extract_pose_features(normalize_pose(p)) for p in poses]) if trials \
        else np.empty((0, N_POSE_FEATURES), dtype=np.float32)
    y = np.array([config.ACTION_TO_LABEL[tr["action"]] for tr in trials], dtype=np.int64)
    metadata = [{"action": tr["action"], "subject": tr["subject"], "trial": tr["trial"],
                 "pose_num_frames": info["num_frames"],
                 "pose_num_failed": len(info["failed_frames"]),
                 "pose_fail_rate": info["fail_rate"], "pose_unreliable": info["unreliable"]}
                for tr, info in zip(trials, infos)]
    return X, y, metadata


if __name__ == "__main__":
    from src.data_loader import load_trials
    from src.pose_extraction import get_trial_pose

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    trial = load_trials([config.TRAIN_SUBJECTS[0]], actions=[config.SELECTED_ACTIONS[0]])[0]
    pose, info = get_trial_pose(trial)
    norm = normalize_pose(pose)
    feats = extract_pose_features(norm)
    names = pose_feature_names()

    print(f"Video:         {trial['video_path']}")
    print(f"Raw pose:      {pose.shape}, fail rate {info['fail_rate']:.1%}, "
          f"unreliable={info['unreliable']}")
    mid_hip = norm[:, [LEFT_HIP, RIGHT_HIP]].mean(axis=1)
    torso = np.linalg.norm(norm[:, [LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=1) - mid_hip, axis=1)
    print(f"Normalized:    |mid-hip| max {np.abs(mid_hip).max():.2e}, "
          f"torso length {torso.min():.3f}..{torso.max():.3f} (expect ~1)")
    print(f"Feature shape: {feats.shape}  (expected ({N_POSE_FEATURES},))")
    for i in [0, 1, 2, 48, 99 + 48, 198 + 48, 297 + 48]:
        print(f"  [{i:3d}] {names[i]:<28} {feats[i]: .5f}")
    assert len(names) == N_POSE_FEATURES == 396
    assert np.isfinite(feats).all(), "NaN/Inf in features"
    print("OK: no NaN/Inf")
