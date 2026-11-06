"""MLP capacity / training-data experiments on fused features.

For each architecture variant (src.models.MODEL_VARIANTS):
    Round 1: train on subjects 1-4, early-stop on subject 5, 5 seeds.
    Round 2: retrain on subjects 1-5 for a fixed epoch count = median Round-1
             best epoch, 5 seeds, no early stopping.
Scalers are refit on each round's training subjects. Random Forest is run on
both training sets as the reference.

The extra variants (64 units, linear) only run if the 128-unit MLP on 1-5
scores below EXTRA_THRESHOLD, or with --all-variants.

Outputs: results/mlp_tuning.csv (summary), results/mlp_tuning_runs.csv,
models in results/models/tuning/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.fusion import concatenate_features, load_features  # noqa: E402
from src.models import MODEL_VARIANTS  # noqa: E402
from src.train import predict_logits, save_model, train_model  # noqa: E402

EXTRA_THRESHOLD = 0.92
CORE_VARIANTS = ["MLP (256-128, dp=0.3)", "MLP (128, dp=0.5)"]
EXTRA_VARIANTS = ["Linear (logistic reg.)", "MLP (64, dp=0.5)"]
OUT_DIR = config.MODELS_DIR / "tuning"
NUM_CLASSES = len(config.SELECTED_ACTIONS)


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label).strip("_").lower()


def run_variant(label: str, d14: dict, d15: dict, y: dict) -> list[dict]:
    """Run both rounds (s1-4 early stopping, s1-5 fixed epochs) for one variant."""
    kw = MODEL_VARIANTS[label]
    X14, X15 = d14["fused"], d15["fused"]
    dim = X14["train"].shape[1]
    rows = []

    # Round 1: subjects 1-4, early stopping on subject 5
    best_epochs = []
    for seed in config.SEEDS:
        model, hist = train_model(X14["train"], y["fit"], X14["val"], y["val"],
                                  dim, NUM_CLASSES, {"seed": seed, **kw})
        acc = float((predict_logits(model, X14["test"]).argmax(1).numpy() == y["test"]).mean())
        best_epochs.append(hist["best_epoch"])
        save_model(model, OUT_DIR / f"{_slug(label)}_s1-4_seed{seed}.pt", **kw, history=hist)
        rows.append({"model": label, "train_subjects": "1-4", "seed": seed, "test_acc": acc,
                     "val_acc": hist["val_acc"][hist["best_epoch"] - 1],
                     "epochs": hist["best_epoch"]})
    n_epochs = int(np.median(best_epochs))
    r1 = np.mean([r["test_acc"] for r in rows])
    print(f"{label:<24} 1-4: {100 * r1:.1f}%  best epochs {best_epochs} -> median {n_epochs}")

    # Round 2: subjects 1-5, fixed epoch count
    accs = []
    for seed in config.SEEDS:
        model, hist = train_model(X15["train"], y["train_all"], None, None, dim, NUM_CLASSES,
                                  {"seed": seed, "max_epochs": n_epochs, **kw})
        acc = float((predict_logits(model, X15["test"]).argmax(1).numpy() == y["test"]).mean())
        accs.append(acc)
        save_model(model, OUT_DIR / f"{_slug(label)}_s1-5_seed{seed}.pt", **kw, history=hist)
        rows.append({"model": label, "train_subjects": "1-5", "seed": seed, "test_acc": acc,
                     "val_acc": np.nan, "epochs": n_epochs})
    print(f"{label:<24} 1-5: {100 * np.mean(accs):.1f}%  ({n_epochs} epochs)")
    return rows


def run_rf(X_train, y_train, X_test, y_test, subjects: str) -> list[dict]:
    """Random Forest reference over all seeds on one training set."""
    rows = []
    for seed in config.SEEDS:
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        acc = float(rf.fit(X_train, y_train).score(X_test, y_test))
        rows.append({"model": "Random Forest", "train_subjects": subjects, "seed": seed,
                     "test_acc": acc, "val_acc": np.nan, "epochs": np.nan})
    print(f"{'Random Forest':<24} {subjects}: {100 * np.mean([r['test_acc'] for r in rows]):.1f}%")
    return rows


def main() -> None:
    """Run the MLP tuning experiments and print the comparison table."""
    p = argparse.ArgumentParser()
    p.add_argument("--all-variants", action="store_true",
                   help="run extra variants regardless of EXTRA_THRESHOLD")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_features()
    fit, val, test = data["fit"], data["val"], data["test"]
    all_imu = np.vstack([fit["imu"], val["imu"]])
    all_pose = np.vstack([fit["pose"], val["pose"]])
    y = {"fit": fit["y"], "val": val["y"], "test": test["y"],
         "train_all": np.concatenate([fit["y"], val["y"]])}

    d14 = concatenate_features(fit["imu"], test["imu"], fit["pose"], test["pose"],
                               imu_val=val["imu"], pose_val=val["pose"])
    d15 = concatenate_features(all_imu, test["imu"], all_pose, test["pose"])

    rows = []
    for label in CORE_VARIANTS:
        rows += run_variant(label, d14, d15, y)

    small_15 = np.mean([r["test_acc"] for r in rows
                        if r["model"] == "MLP (128, dp=0.5)" and r["train_subjects"] == "1-5"])
    if args.all_variants or small_15 < EXTRA_THRESHOLD:
        print(f"\n128-unit MLP on 1-5 = {100 * small_15:.1f}% "
              f"(threshold {100 * EXTRA_THRESHOLD:.0f}%); running extra variants")
        for label in EXTRA_VARIANTS:
            rows += run_variant(label, d14, d15, y)

    rows += run_rf(d14["fused"]["train"], y["fit"], d14["fused"]["test"], y["test"], "1-4")
    rows += run_rf(d15["fused"]["train"], y["train_all"], d15["fused"]["test"], y["test"], "1-5")

    runs = pd.DataFrame(rows)
    runs.to_csv(config.RESULTS_DIR / "mlp_tuning_runs.csv", index=False)
    summary = (runs.groupby(["model", "train_subjects"], sort=False)
               .agg(acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
                    acc_min=("test_acc", "min"), acc_max=("test_acc", "max"),
                    val_acc_mean=("val_acc", "mean"), epochs=("epochs", "median"))
               .reset_index())
    summary.to_csv(config.RESULTS_DIR / "mlp_tuning.csv", index=False)

    print("\n| Model                     | Train Subjects | Epochs  | Val acc (s5) "
          "| Test accuracy (mean ± std) |")
    print("|---------------------------|----------------|---------|--------------"
          "|----------------------------|")
    for r in summary.itertuples():
        ep = "" if np.isnan(r.epochs) else (f"ES ~{r.epochs:.0f}" if r.train_subjects == "1-4"
                                            else f"{r.epochs:.0f}")
        va = "" if np.isnan(r.val_acc_mean) else f"{100 * r.val_acc_mean:.1f}%"
        acc = f"{100 * r.acc_mean:.1f}% ± {100 * r.acc_std:.1f}%"
        print(f"| {r.model:<25} | {r.train_subjects:<14} | {ep:<7} | {va:<12} | {acc:<26} |")
    print(f"\nTest: {len(y['test'])} trials (subjects {config.TEST_SUBJECTS}); "
          f"std across {len(config.SEEDS)} seeds (ddof=1). ES = early stopping "
          "(median best epoch shown).")


if __name__ == "__main__":
    main()
