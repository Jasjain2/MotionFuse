"""MediaPipe pose landmark extraction from video frames, with on-disk caching.

Per trial: [num_frames, 33, 3] landmarks (x, y in [0, 1] of image width/height;
z is depth relative to the hips, roughly on the same scale as x, not metric).

Backends: the legacy `mp.solutions.pose` API is used when available. Recent
MediaPipe releases (incl. all wheels for Python >= 3.13) removed it, so we fall
back to the Tasks `PoseLandmarker` in VIDEO mode, which is the same BlazePose
model family (model_complexity 0/1/2 -> lite/full/heavy .task file).

Cache: raw MediaPipe output (failed frames = NaN, plus visibility) is stored in
config.POSE_CACHE_DIR, so gap filling can change without rerunning MediaPipe.
Delete the cache dir to force re-extraction.
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src import config
from src.data_loader import iter_video_frames, video_info

log = logging.getLogger(__name__)

N_LANDMARKS = 33
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]

_TASK_VARIANTS = {0: "lite", 1: "full", 2: "heavy"}
_TASK_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                   "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")


def _trial_id(trial: dict) -> str:
    return f"a{trial['action']}_s{trial['subject']}_t{trial['trial']}"


def _cache_path(trial: dict) -> Path:
    return config.POSE_CACHE_DIR / f"{_trial_id(trial)}_c{config.POSE_MODEL_COMPLEXITY}.npy"


def _has_solutions_api() -> bool:
    import mediapipe as mp
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")


def _task_model_path() -> Path:
    variant = _TASK_VARIANTS[config.POSE_MODEL_COMPLEXITY]
    path = config.MODEL_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = _TASK_MODEL_URL.format(v=variant)
        log.info("Downloading MediaPipe model %s", url)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def _landmarks_to_rows(landmarks) -> list[tuple]:
    return [(lm.x, lm.y, lm.z, np.nan if lm.visibility is None else lm.visibility)
            for lm in landmarks]


def _run_solutions(frames, fps):
    import mediapipe as mp
    with mp.solutions.pose.Pose(static_image_mode=False,
                                model_complexity=config.POSE_MODEL_COMPLEXITY,
                                min_detection_confidence=config.POSE_MIN_DETECTION_CONF,
                                min_tracking_confidence=config.POSE_MIN_TRACKING_CONF) as pose:
        for rgb in frames:
            res = pose.process(rgb)
            yield _landmarks_to_rows(res.pose_landmarks.landmark) if res.pose_landmarks else None


def _run_tasks(frames, fps):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_task_model_path())),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONF,
        min_pose_presence_confidence=config.POSE_MIN_DETECTION_CONF,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONF,
    )
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        for i, rgb in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = landmarker.detect_for_video(image, int(round(i * 1000 / fps)))
            yield _landmarks_to_rows(res.pose_landmarks[0]) if res.pose_landmarks else None


def run_mediapipe(video_path: Path | str) -> np.ndarray:
    """Run MediaPipe on every frame. Returns [num_frames, 33, 4] (x, y, z,
    visibility) with NaN rows where detection failed."""
    fps = video_info(video_path)["fps"]
    if not fps or fps <= 0:  # OpenCV reports -1/0 when metadata is missing
        fps = config.VIDEO_FPS
    runner = _run_solutions if _has_solutions_api() else _run_tasks
    rows = [np.full((N_LANDMARKS, 4), np.nan) if r is None else np.asarray(r)
            for r in runner(iter_video_frames(video_path), fps)]
    if not rows:
        raise IOError(f"No frames read from {video_path}")
    return np.stack(rows).astype(np.float32)


def _frame_ranges(idx: np.ndarray) -> str:
    """[3, 4, 5, 9] -> '3-5, 9'"""
    if idx.size == 0:
        return ""
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return ", ".join(str(s) if s == e else f"{s}-{e}" for s, e in zip(starts, ends))


def fill_missing(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate frames with failed detection (NaN) per landmark
    coordinate. Leading/trailing gaps hold the nearest detected frame.

    Returns (filled pose, indices of failed frames). If no frame was detected,
    the pose is all zeros.
    """
    num_frames = pose.shape[0]
    detected = ~np.isnan(pose).any(axis=(1, 2))
    failed = np.flatnonzero(~detected)
    if not detected.any():
        return np.zeros_like(pose), failed
    if failed.size == 0:
        return pose.copy(), failed
    ok = np.flatnonzero(detected)
    flat = pose.reshape(num_frames, -1).copy()
    for j in range(flat.shape[1]):
        flat[failed, j] = np.interp(failed, ok, flat[ok, j])
    return flat.reshape(pose.shape), failed


def get_trial_pose(trial: dict, use_cache: bool = True) -> tuple[np.ndarray, dict]:
    """Pose landmarks for one trial, from cache if available.

    Returns:
        pose: [num_frames, 33, 3] with failed frames interpolated
        info: {"num_frames", "failed_frames", "fail_rate", "unreliable"}
    """
    cache = _cache_path(trial)
    if use_cache and cache.exists():
        raw = np.load(cache)
    else:
        raw = run_mediapipe(trial["video_path"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.save(f, raw)
        tmp.replace(cache)

    pose, failed = fill_missing(raw[..., :3])
    fail_rate = failed.size / raw.shape[0]
    unreliable = fail_rate > config.POSE_MAX_FAIL_FRAC
    tid = _trial_id(trial)
    if failed.size:
        log.log(logging.WARNING if unreliable else logging.INFO,
                "%s: detection failed on %d/%d frames (%.0f%%)%s: frames %s",
                tid, failed.size, raw.shape[0], 100 * fail_rate,
                " -> UNRELIABLE" if unreliable else "", _frame_ranges(failed))
    return pose.astype(np.float32), {
        "num_frames": raw.shape[0],
        "failed_frames": failed.tolist(),
        "fail_rate": fail_rate,
        "unreliable": bool(unreliable),
    }


def extract_poses(trials: list[dict], use_cache: bool = True) -> tuple[list[np.ndarray], list[dict]]:
    """get_trial_pose over many trials with a progress bar."""
    poses, infos = [], []
    with logging_redirect_tqdm():
        for tr in tqdm(trials, desc="Pose extraction", unit="trial"):
            pose, info = get_trial_pose(tr, use_cache=use_cache)
            poses.append(pose)
            infos.append(info)
    n_bad = sum(i["unreliable"] for i in infos)
    if n_bad:
        log.warning("%d/%d trials flagged unreliable (> %.0f%% frames failed)",
                    n_bad, len(trials), 100 * config.POSE_MAX_FAIL_FRAC)
    return poses, infos
