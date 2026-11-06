"""Extract (cached) MediaPipe poses + pose features for all trials and summarize
detection failures.

Raw landmarks are cached in data/pose_cache/, so only uncached trials run
MediaPipe. Summary is printed and saved to results/pose_extraction_summary.txt;
per-trial failed-frame logs go to results/pose_extraction.log.
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_loader import load_split  # noqa: E402
from src.normalization import extract_pose_features_batch  # noqa: E402


def build_summary(metadata: list[dict], X: np.ndarray, elapsed: float) -> list[str]:
    """Format the detection-failure summary as lines of text."""
    names = [f"a{m['action']}_s{m['subject']}_t{m['trial']}" for m in metadata]
    rates = np.array([m["pose_fail_rate"] for m in metadata])
    failed = sorted((m for m in metadata if m["pose_num_failed"]),
                    key=lambda m: -m["pose_fail_rate"])
    unreliable = [n for n, m in zip(names, metadata) if m["pose_unreliable"]]
    frames = np.array([m["pose_num_frames"] for m in metadata])

    lines = [
        "POSE EXTRACTION SUMMARY",
        f"generated {datetime.now():%Y-%m-%d %H:%M}, model_complexity="
        f"{config.POSE_MODEL_COMPLEXITY}, run time {elapsed / 60:.1f} min",
        "=" * 60,
        f"Total trials processed:         {len(metadata)}",
        f"Pose feature matrix:            {X.shape}, finite: {bool(np.isfinite(X).all())}",
        f"Frames per trial:               min {frames.min()}, median "
        f"{int(np.median(frames))}, max {frames.max()}",
        f"Trials with 0 failed frames:    {len(metadata) - len(failed)}",
        f"Trials with >=1 failed frame:   {len(failed)}",
    ]
    lines += [f"    a{m['action']}_s{m['subject']}_t{m['trial']:<4} "
              f"{100 * m['pose_fail_rate']:5.1f}%  "
              f"({m['pose_num_failed']}/{m['pose_num_frames']} frames)" for m in failed]
    lines.append(f"Unreliable (> {config.POSE_MAX_FAIL_FRAC:.0%} frames failed): {len(unreliable)}")
    lines += [f"    {n}" for n in unreliable]
    worst = f"  ({names[int(rates.argmax())]})" if rates.max() > 0 else ""
    lines += [
        f"pose_fail_rate mean:            {100 * rates.mean():.2f}%",
        f"pose_fail_rate max:             {100 * rates.max():.2f}%{worst}",
    ]
    return lines


def main() -> None:
    """Extract poses and features for all trials and save the summary."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
        handlers=[logging.FileHandler(config.RESULTS_DIR / "pose_extraction.log", mode="w"),
                  logging.StreamHandler()])

    split = load_split()
    trials = split["train"] + split["test"]
    cached = sum(1 for tr in config.POSE_CACHE_DIR.glob(f"*_c{config.POSE_MODEL_COMPLEXITY}.npy"))
    print(f"{len(trials)} trials ({len(split['train'])} train, {len(split['test'])} test); "
          f"{cached} already in {config.POSE_CACHE_DIR}")

    t0 = time.time()
    X, _, metadata = extract_pose_features_batch(trials)
    lines = build_summary(metadata, X, time.time() - t0)

    out = config.RESULTS_DIR / "pose_extraction_summary.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
