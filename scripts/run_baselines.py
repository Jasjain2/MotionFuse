"""Three-way baseline comparison: IMU-only vs pose-only vs fused MLP, plus
Random Forest and RBF-SVM on fused features as reference models.

Split (subject-wise, no subject appears in two sets):
    fit:  config.FIT_SUBJECTS  (1-4)  -> training rows, scaler fitting
    val:  config.VAL_SUBJECTS  (5)    -> MLP early stopping only
    test: config.TEST_SUBJECTS (6-8)  -> reported accuracy
All models, including RF/SVM, train on the same fit subjects so the
comparison is like-for-like.

Outputs:
    results/baseline_comparison.csv  summary table
    results/baseline_runs.csv        one row per (model, features, seed)
    results/models/                  MLP .pt files, RF/SVM + scalers .joblib
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.fusion import concatenate_features, load_features  # noqa: E402
from src.train import predict_logits, save_model, train_model  # noqa: E402

FEATURE_SETS = ["imu", "pose", "fused"]


def main() -> None:
    """Train and compare all baseline models; save table and models."""
    sys.stdout.reconfigure(encoding="utf-8")  # "±" survives Windows pipes/redirects
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_features()
    fused = concatenate_features(data["fit"]["imu"], data["test"]["imu"],
                                 data["fit"]["pose"], data["test"]["pose"],
                                 imu_val=data["val"]["imu"], pose_val=data["val"]["pose"])
    joblib.dump(fused["scalers"], config.MODELS_DIR / "scalers.joblib")
    y_fit, y_val, y_test = data["fit"]["y"], data["val"]["y"], data["test"]["y"]
    num_classes = len(config.SELECTED_ACTIONS)
    print(f"fit {len(y_fit)} (s{config.FIT_SUBJECTS}), val {len(y_val)} "
          f"(s{config.VAL_SUBJECTS}), test {len(y_test)} (s{config.TEST_SUBJECTS})")

    runs = []
    for feats in FEATURE_SETS:
        Xs = fused[feats]
        dim = Xs["train"].shape[1]
        for seed in config.SEEDS:
            model, hist = train_model(Xs["train"], y_fit, Xs["val"], y_val,
                                      dim, num_classes, {"seed": seed})
            pred = predict_logits(model, Xs["test"]).argmax(1).numpy()
            acc = float((pred == y_test).mean())
            save_model(model, config.MODELS_DIR / f"mlp_{feats}_seed{seed}.pt",
                       hidden_dims=config.HIDDEN_DIMS, dropout=config.DROPOUT,
                       features=feats, seed=seed, history=hist)
            runs.append({"model": "MLP", "features": feats, "n_features": dim, "seed": seed,
                         "test_acc": acc, "val_acc": hist["val_acc"][hist["best_epoch"] - 1],
                         "best_epoch": hist["best_epoch"],
                         "epochs_run": len(hist["val_loss"])})
            print(f"  MLP {feats:<5} seed {seed}: test {acc:.3f}  "
                  f"(best epoch {hist['best_epoch']}/{len(hist['val_loss'])})")

    Xf = fused["fused"]
    for seed in config.SEEDS:
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        rf.fit(Xf["train"], y_fit)
        acc = float(rf.score(Xf["test"], y_test))
        joblib.dump(rf, config.MODELS_DIR / f"rf_fused_seed{seed}.joblib")
        runs.append({"model": "Random Forest", "features": "fused",
                     "n_features": Xf["train"].shape[1], "seed": seed, "test_acc": acc,
                     "val_acc": float(rf.score(Xf["val"], y_val))})
        print(f"  RF  fused seed {seed}: test {acc:.3f}")

    # RBF-SVM is deterministic, so one fit; std across seeds would be 0 by construction.
    svm = SVC(kernel="rbf").fit(Xf["train"], y_fit)
    acc = float(svm.score(Xf["test"], y_test))
    joblib.dump(svm, config.MODELS_DIR / "svm_fused.joblib")
    runs.append({"model": "SVM (RBF)", "features": "fused",
                 "n_features": Xf["train"].shape[1], "seed": None, "test_acc": acc,
                 "val_acc": float(svm.score(Xf["val"], y_val))})
    print(f"  SVM fused: test {acc:.3f}")

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(config.RESULTS_DIR / "baseline_runs.csv", index=False)
    summary = (runs_df.groupby(["model", "features"], sort=False)
               .agg(n_features=("n_features", "first"), n_runs=("test_acc", "size"),
                    acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
                    acc_min=("test_acc", "min"), acc_max=("test_acc", "max"),
                    val_acc_mean=("val_acc", "mean"))
               .reset_index())
    summary["acc_std"] = summary["acc_std"].fillna(0.0)
    summary.to_csv(config.RESULTS_DIR / "baseline_comparison.csv", index=False)

    names = {"imu": "IMU-only", "pose": "Pose-only", "fused": "Fused"}
    print("\n| Model          | Features  | Accuracy (mean ± std) |")
    print("|----------------|-----------|-----------------------|")
    for r in summary.itertuples():
        acc = f"{100 * r.acc_mean:.1f}% ± {100 * r.acc_std:.1f}%"
        if r.n_runs == 1:
            acc = f"{100 * r.acc_mean:.1f}% (deterministic)"
        print(f"| {r.model:<14} | {names[r.features]:<9} | {acc:<21} |")
    print(f"\nTest set: {len(y_test)} trials (subjects {config.TEST_SUBJECTS}); "
          f"std is across {len(config.SEEDS)} seeds, sample std (ddof=1).")
    print(f"Saved {config.RESULTS_DIR / 'baseline_comparison.csv'} and models to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
