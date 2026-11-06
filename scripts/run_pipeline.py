"""End-to-end pipeline: pose extraction -> baselines (training) -> evaluation report.

Equivalent to running, in order:
    python scripts/run_pose_extraction.py
    python scripts/run_baselines.py
    python scripts/generate_report.py
Requires the dataset in data/utd-mhad/ (python setup_data.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_report  # noqa: E402
import run_baselines  # noqa: E402
import run_pose_extraction  # noqa: E402


def main() -> None:
    """Run every pipeline stage in order."""
    for name, stage in [("pose extraction", run_pose_extraction.main),
                        ("baselines", run_baselines.main),
                        ("evaluation report", generate_report.main)]:
        print(f"\n{'#' * 60}\n# {name}\n{'#' * 60}")
        stage()


if __name__ == "__main__":
    main()
