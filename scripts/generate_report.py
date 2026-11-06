"""Full evaluation of the production model (Random Forest, fused features,
trained on subjects 1-4) with MLP results for comparison. Writes everything to
results/.

Production model: one RF retrained with config.RANDOM_SEED. We deliberately
don't pick the best of the 5 baseline seeds, since that would select on the
test set; the 5-seed mean ± std is reported as the headline number instead.

Outputs (results/):
    per_action_metrics.csv, confusion_matrix_{imu,pose,fused}.png
    per_subject_accuracy.csv
    uncertainty_analysis.csv, reliability_diagram.png, uncertainty_histogram.png
    failure_cases.txt
    modality_contribution.csv, modality_contribution.png
    summary.txt
    models/rf_fused_production.joblib
"""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src.fusion import concatenate_features, load_features  # noqa: E402
from src.models import MODEL_VARIANTS  # noqa: E402
from src.train import predict_logits, train_model  # noqa: E402

MLP_VARIANTS = ["MLP (256-128, dp=0.3)", "MLP (64, dp=0.5)"]
R = config.RESULTS_DIR


def pct(x: float) -> str:
    """Format a fraction as a percentage with one decimal."""
    return f"{100 * x:.1f}%"


def mean_std(xs) -> str:
    """Format values as 'mean% ± std%' (sample std)."""
    return f"{100 * np.mean(xs):.1f}% ± {100 * np.std(xs, ddof=1):.1f}%"


def main() -> None:
    """Run all evaluations and write every output to results/."""
    sys.stdout.reconfigure(encoding="utf-8")
    R.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_features()
    fit, val, test = data["fit"], data["val"], data["test"]
    feats = concatenate_features(fit["imu"], test["imu"], fit["pose"], test["pose"],
                                 imu_val=val["imu"], pose_val=val["pose"])
    y_fit, y_val, y_test, meta = fit["y"], val["y"], test["y"], test["meta"]
    n_test = len(y_test)

    # -- Random Forests: 3 modalities x 5 seeds --------------------------------
    rf_acc, rf_action_acc, rf_pred, rf_proba, rf_all_preds = {}, {}, {}, {}, {}
    for m in ev.MODALITY_NAMES:
        accs, action_accs, preds = [], [], []
        for seed in config.SEEDS:
            rf = ev.train_rf(feats[m]["train"], y_fit, seed)
            pred = rf.predict(feats[m]["test"])
            accs.append((pred == y_test).mean())
            preds.append(pred)
            action_accs.append(ev.per_action_accuracy(y_test, pred))
            if seed == config.RANDOM_SEED:
                rf_pred[m], rf_proba[m] = pred, rf.predict_proba(feats[m]["test"])
                if m == "fused":
                    joblib.dump(rf, config.MODELS_DIR / "rf_fused_production.joblib")
        rf_acc[m], rf_action_acc[m] = np.array(accs), np.stack(action_accs)
        rf_all_preds[m] = np.stack(preds)

    pred, proba = rf_pred["fused"], rf_proba["fused"]
    correct = pred == y_test
    prod_acc = correct.mean()
    ci_lo, ci_hi = ev.bootstrap_accuracy_ci(correct)

    # -- MLP comparison (fused, s1-4, early stopping on s5) --------------------
    mlp_acc = {}
    for label in MLP_VARIANTS:
        accs = []
        for seed in config.SEEDS:
            model, _ = train_model(feats["fused"]["train"], y_fit, feats["fused"]["val"], y_val,
                                   feats["fused"]["train"].shape[1], len(ev.ACTION_LABELS),
                                   {"seed": seed, **MODEL_VARIANTS[label]})
            accs.append((predict_logits(model, feats["fused"]["test"]).argmax(1).numpy()
                         == y_test).mean())
        mlp_acc[label] = accs

    # -- 1. Per-action ----------------------------------------------------------
    per_action = ev.per_action_metrics(y_test, pred)
    per_action.to_csv(R / "per_action_metrics.csv", index=False)
    print("Per-action metrics, fused RF (production, worst first):")
    print(per_action.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for m in ev.MODALITY_NAMES:
        ev.plot_confusion_matrix(y_test, rf_pred[m], f"Random Forest, {ev.MODALITY_NAMES[m]}",
                                 R / f"confusion_matrix_{m}.png")

    # -- 2. Per-subject ---------------------------------------------------------
    per_subject = ev.per_subject_analysis(y_test, pred, meta)
    per_subject.to_csv(R / "per_subject_accuracy.csv", index=False)
    worst_by_subject = ev.worst_actions_by_subject(per_subject)
    subj_overall = per_subject[per_subject["action"] == "ALL"].iloc[0]
    subj_acc = {s: subj_overall[f"s{s}_acc"] for s in config.TEST_SUBJECTS}

    # -- 3. Uncertainty ---------------------------------------------------------
    unc = ev.uncertainty_analysis(proba, y_test)
    ev.uncertainty_table(unc).to_csv(R / "uncertainty_analysis.csv", index=False)
    ev.plot_reliability_diagram(unc, "Reliability, fused Random Forest", R / "reliability_diagram.png")
    ev.plot_entropy_histogram(unc, "Predictive entropy, fused Random Forest",
                              R / "uncertainty_histogram.png")

    # -- 4. Failure cases -------------------------------------------------------
    fails = ev.failure_cases(y_test, proba, meta)
    (R / "failure_cases.txt").write_text(
        ev.format_failure_report(fails, n_test, "Failure cases, fused Random Forest "
                                 f"(seed {config.RANDOM_SEED}, test s6-8)") + "\n",
        encoding="utf-8")

    # -- 5. Modality contribution ----------------------------------------------
    contrib = ev.modality_contribution(rf_action_acc, rf_all_preds, y_test)
    contrib.to_csv(R / "modality_contribution.csv", index=False)
    ev.plot_modality_contribution(contrib, len(config.SEEDS), R / "modality_contribution.png")

    # -- 6. Summary -------------------------------------------------------------
    L = []
    add = L.append
    add("MOTIONFUSE EVALUATION SUMMARY")
    add("=" * 60)
    add(f"Data: UTD-MHAD, 8 athletic actions. Train subjects {config.FIT_SUBJECTS} "
        f"({len(y_fit)} trials), validation s{config.VAL_SUBJECTS} (MLP early stopping only), "
        f"test {config.TEST_SUBJECTS} ({n_test} trials).")
    add(f"Production model: Random Forest (100 trees), fused IMU+pose features "
        f"({feats['fused']['train'].shape[1]} dims), seed {config.RANDOM_SEED}.")
    add("")
    add("OVERALL ACCURACY (test s6-8)")
    add(f"  Fused RF, {len(config.SEEDS)} seeds:   {mean_std(rf_acc['fused'])}")
    add(f"  Production model (seed {config.RANDOM_SEED}): {pct(prod_acc)}  "
        f"(95% bootstrap CI over test trials: {pct(ci_lo)} - {pct(ci_hi)})")
    add(f"  IMU-only RF:  {mean_std(rf_acc['imu'])}")
    add(f"  Pose-only RF: {mean_std(rf_acc['pose'])}")
    add("")
    add("MLP COMPARISON (fused, same train subjects 1-4, early stopping on s5)")
    for label, accs in mlp_acc.items():
        add(f"  {label:<24} {mean_std(accs)}")
    add(f"  {'Random Forest':<24} {mean_std(rf_acc['fused'])}")
    add("")

    acc_by_action = per_action.set_index("action")["accuracy"]
    best, worst = acc_by_action.max(), acc_by_action.min()
    add("ACTIONS (production model; each action has 12 test trials)")
    add(f"  Best:  {', '.join(a for a, v in acc_by_action.items() if v == best)}  {pct(best)}")
    add(f"  Worst: {', '.join(a for a, v in acc_by_action.items() if v == worst)}  {pct(worst)}")
    worst_rows = per_action[per_action["accuracy"] < 1.0]
    for r in worst_rows.itertuples():
        add(f"    {r.action:<17} acc {pct(r.accuracy):>6}  precision {r.precision:.2f}  "
            f"F1 {r.f1:.2f}")
    add("")

    best_s = max(subj_acc, key=subj_acc.get)
    worst_s = min(subj_acc, key=subj_acc.get)
    add("SUBJECTS (each subject has 32 test trials)")
    for s in config.TEST_SUBJECTS:
        tag = " (best)" if s == best_s else " (worst)" if s == worst_s else ""
        bad = ", ".join(f"{a} {pct(v)}" for a, v in worst_by_subject[f"s{s}"]) or "none"
        add(f"  s{s}: {pct(subj_acc[s])}{tag}  weak actions: {bad}")
    shared = per_subject[per_subject["pattern"] == "all subjects"]["action"].tolist()
    specific = per_subject[per_subject["pattern"].str.startswith("subject-specific")]
    add(f"  Errors in all test subjects: {', '.join(shared) or 'none'}")
    add(f"  Subject-specific errors: " + (", ".join(
        f"{r.action} ({r.failing_subjects})" for r in specific.itertuples()) or "none"))
    add("")

    o = unc["overall"]
    add("UNCERTAINTY (RF confidence = fraction of trees voting for the predicted class)")
    add(f"  Mean confidence: correct {o['mean_conf_correct']:.2f}, "
        f"incorrect {o['mean_conf_incorrect']:.2f}")
    add(f"  Mean entropy:    correct {o['mean_entropy_correct']:.2f}, "
        f"incorrect {o['mean_entropy_incorrect']:.2f} nats")
    add(f"  Expected calibration error (10 bins): {o['ece']:.3f}  "
        f"(mean confidence {o['mean_confidence']:.2f} vs accuracy {o['accuracy']:.2f})")
    direction = "UNDER-confident" if o["mean_confidence"] < o["accuracy"] else "OVER-confident"
    add(f"  -> the RF is {direction}: vote fractions spread over similar classes, so "
        "confidence ranks trials usefully but should not be read as a probability.")
    low = unc["per_action"].head(3)
    add("  Lowest-confidence actions: " + ", ".join(
        f"{r.action} {r.mean_confidence:.2f}" for r in low.itertuples()))
    add("")

    add("FAILURE PATTERNS (see failure_cases.txt)")
    if fails.empty:
        add("  No misclassifications.")
    else:
        pairs = fails.groupby(["true", "predicted"]).size().sort_values(ascending=False)
        for (t, p), n in pairs.items():
            add(f"  {t} -> {p}: {n}")
    add("")

    add(f"KEY FUSION FINDING (per-action accuracy, mean of {len(config.SEEDS)} RF seeds)")
    add(f"  Thresholds: a modality 'works' at >= {pct(ev.SUFFICIENT_ACC)}; fusion 'adds little' "
        f"if <= {100 * ev.LITTLE_GAIN:.0f} pts; 'critical' if >= {100 * ev.CRITICAL_GAIN:.0f} pts "
        "over the best single modality.")
    for cat, group in contrib.groupby("category", sort=False):
        add(f"  {cat}: {', '.join(group['action'])}")
    add("  Largest fusion gains:")
    for r in contrib[contrib["fusion_gain"] > ev.LITTLE_GAIN].itertuples():
        add(f"    {r.action}: IMU {pct(r.imu_acc)}, pose {pct(r.pose_acc)} -> fused "
            f"{pct(r.fused_acc)} (+{100 * r.fusion_gain:.1f} pts, {r.category})")
        add(f"      most common error (pooled over seeds): IMU-only -> "
            f"{r.imu_top_confusion or 'none'}; pose-only -> {r.pose_top_confusion or 'none'}; "
            f"fused -> {r.fused_top_confusion or 'none'}")
    complementary = [r.action for r in contrib.itertuples()
                     if r.fusion_gain > ev.LITTLE_GAIN and r.imu_top_confusion and r.pose_top_confusion
                     and r.imu_top_confusion.split(" (")[0] != r.pose_top_confusion.split(" (")[0]]
    add("  Why (interpretation): fusion helps most where the two modalities make different "
        "mistakes, so each carries the evidence that breaks the other's tie. A likely "
        "mechanism: the wrist IMU captures arm dynamics (speed, rotation, impact) but not "
        "body configuration, while pose captures whole-body shape but only a coarse 15 fps "
        "view of fast wrist motion. This is a hypothesis consistent with the confusions "
        "above, not something the experiment isolates.")
    if complementary:
        add(f"  Complementary errors (different top confusion per modality): "
            f"{', '.join(complementary)}")
    add("")

    add("LIMITATIONS")
    for text in [
        f"Small dataset: 8 subjects, {len(y_fit)} training and {n_test} test trials. One test "
        f"trial = {100 / n_test:.1f} pts of accuracy; the 95% CI above spans "
        f"{100 * (ci_hi - ci_lo):.0f} pts. Per-subject-per-action cells have 4 trials (25-pt "
        "steps), so per-action and per-subject differences are indicative, not significant.",
        "No temporal modeling: features are whole-trial summaries (mean/std, FFT peak "
        "magnitudes, mean/std velocity). The order of motion phases (wind-up, strike, "
        "follow-through) is discarded; a sequence model (1D CNN/GRU on landmarks + IMU) "
        "could use it but needs more data.",
        "Offline pipeline: each trial is pre-segmented and processed whole. MediaPipe heavy "
        "takes ~4-8 s per trial on CPU; there is no streaming segmentation and IMU/video are "
        "aligned only by trial boundaries, not timestamps.",
        "RF confidence is a tree-vote fraction, not a calibrated probability, and is not "
        "comparable to a neural network's softmax. Neither is calibrated by default; if "
        "confidences drive decisions, calibrate (e.g. isotonic/Platt via cross-validation on "
        "training subjects) and re-check the reliability diagram.",
        "Model selection: the validation subject (s5) is near-saturated (97-99% for all MLPs), "
        "so it cannot rank models; leave-one-subject-out CV on s1-5 would be a sounder "
        "selection protocol. The test set has been viewed across several sessions of "
        "comparisons, so treat the headline numbers as slightly optimistic.",
        "Setting: single fixed camera, lab conditions, one right-wrist IMU; MediaPipe z is "
        "relative depth, not metric. Generalization to field footage or other sensor "
        "placements is untested.",
    ]:
        add(f"  - {text}")

    summary = "\n".join(L) + "\n"
    (R / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"All outputs written to {R}")


if __name__ == "__main__":
    main()
