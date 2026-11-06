"""Feature extraction from inertial (accelerometer + gyroscope) data.

Each trial ([num_frames, 6] at 50 Hz) is reduced to a fixed 76-dim vector:
    42  per-channel stats   (mean, std, min, max, median, rms, zcr) x 6
    30  per-channel FFT     (top-5 spectral peak magnitudes) x 6
     4  cross-channel       (mean, std of |accel| and |gyro|)
"""

import numpy as np
import scipy.fft
from scipy.signal import find_peaks

from src import config

STAT_NAMES = ["mean", "std", "min", "max", "median", "rms", "zcr"]
N_FFT_PEAKS = 5
CROSS_NAMES = ["acc_mag_mean", "acc_mag_std", "gyro_mag_mean", "gyro_mag_std"]
N_FEATURES = (len(STAT_NAMES) + N_FFT_PEAKS) * len(config.IMU_CHANNELS) + len(CROSS_NAMES)


def feature_names() -> list[str]:
    """Names in the same order as extract_imu_features output."""
    names = [f"{ch}_{s}" for ch in config.IMU_CHANNELS for s in STAT_NAMES]
    names += [f"{ch}_fft_peak{k + 1}" for ch in config.IMU_CHANNELS
              for k in range(N_FFT_PEAKS)]
    return names + CROSS_NAMES


def _stat_features(imu: np.ndarray) -> np.ndarray:
    """[6 * 7] channel-major: all stats for accel_x, then accel_y, ..."""
    n = imu.shape[0]
    sign_changes = np.sum(np.diff(np.signbit(imu), axis=0), axis=0)
    stats = np.stack([
        imu.mean(axis=0),
        imu.std(axis=0),
        imu.min(axis=0),
        imu.max(axis=0),
        np.median(imu, axis=0),
        np.sqrt(np.mean(imu ** 2, axis=0)),
        sign_changes / n,
    ], axis=1)  # [6, 7]
    return stats.ravel()


def _fft_peak_magnitudes(x: np.ndarray, k: int = N_FFT_PEAKS) -> np.ndarray:
    """Magnitudes of the k largest spectral peaks of a 1D signal, excluding DC.

    Magnitudes are divided by the signal length so they're comparable across
    trials of different duration (raw rfft magnitudes scale with N). If fewer
    than k local peaks exist, the remaining slots fall back to the largest
    non-peak bins, then zeros.
    """
    mag = np.abs(scipy.fft.rfft(x)) / len(x)
    mag = mag[1:]  # drop DC
    peaks, _ = find_peaks(mag)
    top = np.sort(mag[peaks])[::-1][:k]
    if len(top) < k:
        rest = np.sort(np.delete(mag, peaks))[::-1][:k - len(top)]
        top = np.concatenate([top, rest])
    return np.pad(top, (0, k - len(top)))


def _fft_features(imu: np.ndarray) -> np.ndarray:
    return np.concatenate([_fft_peak_magnitudes(imu[:, c]) for c in range(imu.shape[1])])


def _cross_channel_features(imu: np.ndarray) -> np.ndarray:
    acc_mag = np.linalg.norm(imu[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(imu[:, 3:6], axis=1)
    return np.array([acc_mag.mean(), acc_mag.std(), gyro_mag.mean(), gyro_mag.std()])


def extract_imu_features(imu: np.ndarray) -> np.ndarray:
    """Map one raw IMU recording [num_frames, 6] to a 1D feature vector [76]."""
    imu = np.asarray(imu, dtype=np.float64)
    if imu.ndim != 2 or imu.shape[1] != len(config.IMU_CHANNELS):
        raise ValueError(f"Expected [num_frames, 6], got {imu.shape}")
    if imu.shape[0] < 2:
        raise ValueError(f"Need at least 2 frames, got {imu.shape[0]}")
    feats = np.concatenate([
        _stat_features(imu),
        _fft_features(imu),
        _cross_channel_features(imu),
    ])
    assert feats.shape == (N_FEATURES,)
    return feats.astype(np.float32)


def extract_imu_features_batch(trials: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Extract features for a list of trial dicts from data_loader.

    Returns:
        X_imu:    [num_trials, 76] float32
        y:        [num_trials] int64 class indices (config.ACTION_TO_LABEL)
        metadata: per-row {"action", "subject", "trial"}
    """
    X = np.stack([extract_imu_features(tr["imu"]) for tr in trials]) if trials \
        else np.empty((0, N_FEATURES), dtype=np.float32)
    y = np.array([config.ACTION_TO_LABEL[tr["action"]] for tr in trials], dtype=np.int64)
    metadata = [{"action": tr["action"], "subject": tr["subject"], "trial": tr["trial"]}
                for tr in trials]
    return X, y, metadata


if __name__ == "__main__":
    import sys

    from src.data_loader import load_imu

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        a, s = config.SELECTED_ACTIONS[0], config.TRAIN_SUBJECTS[0]
        path = config.INERTIAL_DIR / config.INERTIAL_FILE_FMT.format(action=a, subject=s, trial=1)
    imu = load_imu(path)
    feats = extract_imu_features(imu)
    names = feature_names()

    print(f"File:           {path}")
    print(f"Raw IMU shape:  {imu.shape}")
    print(f"Feature shape:  {feats.shape}  (expected ({N_FEATURES},))")
    for i in [0, 1, 6, 42, 43, 72, 73, 74, 75]:
        print(f"  [{i:2d}] {names[i]:<22} {feats[i]: .5f}")
    assert len(names) == N_FEATURES == 76
    assert not np.isnan(feats).any(), "NaN in features"
    assert np.isfinite(feats).all(), "Inf in features"
    print("OK: no NaN/Inf")
