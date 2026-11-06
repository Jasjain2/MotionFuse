"""Evaluation: per-action and per-subject metrics, uncertainty, failure cases,
modality contribution, and their plots.

Class labels are the contiguous indices from config.ACTION_TO_LABEL; plots and
tables use the action names. "Per-action accuracy" is the fraction of that
action's trials classified correctly (identical to its recall).
"""

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

from src import config  # noqa: E402

ACTION_LABELS = [config.ACTION_NAMES[a] for a in config.SELECTED_ACTIONS]  # index = class label
MODALITY_NAMES = {"imu": "IMU-only", "pose": "Pose-only", "fused": "Fused"}

# Modality-contribution thresholds
SUFFICIENT_ACC = 0.90   # a modality "works" on an action at or above this accuracy
LITTLE_GAIN = 0.05      # fusion "adds little" if it beats a modality by at most this
CRITICAL_GAIN = 0.10    # fusion is "critical" if it beats the best single modality by this

# Plot styling: reference data-viz palette, light mode
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (CVD-validated all-pairs)
MODALITY_COLORS = dict(zip(MODALITY_NAMES, SERIES))
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


# ---------------------------------------------------------------------------
# Models / basic metrics
# ---------------------------------------------------------------------------
def train_rf(X: np.ndarray, y: np.ndarray, seed: int = config.RANDOM_SEED) -> RandomForestClassifier:
    """Fit a 100-tree Random Forest with the given seed."""
    return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(X, y)


def bootstrap_accuracy_ci(correct: np.ndarray, n_boot: int = config.N_BOOTSTRAP,
                          level: float = config.CI_LEVEL,
                          seed: int = config.RANDOM_SEED) -> tuple[float, float]:
    """Percentile bootstrap CI for accuracy, resampling test trials."""
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    accs = rng.choice(correct, size=(n_boot, len(correct)), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(accs, alpha)), float(np.quantile(accs, 1 - alpha))


def per_action_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """[num_classes] fraction of each action's trials predicted correctly."""
    return np.array([(y_pred[y_true == c] == c).mean() for c in range(len(ACTION_LABELS))])


def per_action_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Accuracy, precision, recall, F1 per action, sorted worst accuracy first."""
    labels = list(range(len(ACTION_LABELS)))
    prec, rec, f1, n = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                       zero_division=0)
    df = pd.DataFrame({"action": ACTION_LABELS, "n": n,
                       "accuracy": per_action_accuracy(y_true, y_pred),
                       "precision": prec, "recall": rec, "f1": f1})
    return df.sort_values(["accuracy", "f1", "action"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _new_fig(w: float, h: float, **kw):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE, **kw)
    for a in np.atleast_1d(ax):
        a.set_facecolor(SURFACE)
    return fig, ax


def _style_axes(ax, grid_axis: str = "y") -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, path) -> None:
    """Save a confusion-matrix heatmap with action names on both axes."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ACTION_LABELS))))
    fig, ax = _new_fig(7.5, 6.2)
    sns.heatmap(cm, annot=True, fmt="d", cmap=SEQ_BLUE, vmin=0, vmax=cm.sum(axis=1).max(),
                square=True, linewidths=2, linecolor=SURFACE, cbar_kws={"shrink": 0.75},
                xticklabels=ACTION_LABELS, yticklabels=ACTION_LABELS, ax=ax,
                annot_kws={"fontsize": 10})
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title}  (accuracy {acc:.1%}, n={cm.sum()})", color=TEXT, fontsize=12,
                 loc="left", pad=12)
    ax.set_xlabel("Predicted action")
    ax.set_ylabel("True action")
    _style_axes(ax, grid_axis="")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=TEXT_2, labelsize=8)
    cbar.outline.set_visible(False)
    cbar.set_label("Trials", color=TEXT_2)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Per-subject analysis
# ---------------------------------------------------------------------------
def per_subject_analysis(y_true: np.ndarray, y_pred: np.ndarray, meta: list[dict]) -> pd.DataFrame:
    """Accuracy per (action, subject), wide: one row per action plus an "ALL"
    row; columns s{N}_acc, mean_acc, failing_subjects, pattern."""
    subjects = np.array([m["subject"] for m in meta])
    subs = sorted(set(subjects))
    correct = y_true == y_pred
    rows = []
    for c, name in enumerate(ACTION_LABELS):
        row = {"action": name}
        for s in subs:
            mask = (y_true == c) & (subjects == s)
            row[f"s{s}_acc"] = correct[mask].mean() if mask.any() else np.nan
        failing = [s for s in subs if row[f"s{s}_acc"] < 1.0]
        row["failing_subjects"] = ",".join(f"s{s}" for s in failing)
        row["pattern"] = ("no errors" if not failing else
                          "all subjects" if len(failing) == len(subs) else
                          f"subject-specific ({row['failing_subjects']})" if len(failing) == 1
                          else f"{len(failing)} of {len(subs)} subjects")
        rows.append(row)
    overall = {"action": "ALL", **{f"s{s}_acc": correct[subjects == s].mean() for s in subs},
               "failing_subjects": "", "pattern": ""}
    df = pd.DataFrame(rows + [overall])
    df.insert(1 + len(subs), "mean_acc", df[[f"s{s}_acc" for s in subs]].mean(axis=1))
    return df


def worst_actions_by_subject(per_subject: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """{"s6": [(action, acc), ...]} for actions below 100%, worst first."""
    actions = per_subject[per_subject["action"] != "ALL"]
    out = {}
    for col in [c for c in per_subject.columns if c.endswith("_acc") and c != "mean_acc"]:
        bad = actions[actions[col] < 1.0].sort_values(col)
        out[col.removesuffix("_acc")] = list(zip(bad["action"], bad[col]))
    return out


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def uncertainty_analysis(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    """Confidence = max class probability (for RF: fraction of tree votes).

    Returns {"overall": dict, "per_action": DataFrame, "reliability": DataFrame,
             "confidence", "entropy", "correct": per-trial arrays}.
    """
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    p = np.clip(proba, 1e-12, 1)
    entropy = -(proba * np.log(p)).sum(axis=1)  # nats; max = ln(8) = 2.08
    correct = pred == y_true

    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rel = []
    for b in range(n_bins):
        m = bin_idx == b
        rel.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "count": int(m.sum()),
                    "mean_confidence": conf[m].mean() if m.any() else np.nan,
                    "accuracy": correct[m].mean() if m.any() else np.nan})
    rel = pd.DataFrame(rel)
    filled = rel["count"] > 0
    ece = float((rel.loc[filled, "count"] / len(conf) *
                 (rel.loc[filled, "accuracy"] - rel.loc[filled, "mean_confidence"]).abs()).sum())

    per_action = pd.DataFrame([{
        "action": name,
        "mean_confidence": conf[y_true == c].mean(),
        "mean_entropy": entropy[y_true == c].mean(),
        "accuracy": correct[y_true == c].mean(),
    } for c, name in enumerate(ACTION_LABELS)]).sort_values("mean_confidence").reset_index(drop=True)

    overall = {
        "n": len(conf), "accuracy": correct.mean(), "mean_confidence": conf.mean(),
        "mean_conf_correct": conf[correct].mean(),
        "mean_conf_incorrect": conf[~correct].mean() if (~correct).any() else np.nan,
        "mean_entropy_correct": entropy[correct].mean(),
        "mean_entropy_incorrect": entropy[~correct].mean() if (~correct).any() else np.nan,
        "ece": ece,
    }
    return {"overall": overall, "per_action": per_action, "reliability": rel,
            "confidence": conf, "entropy": entropy, "correct": correct}


def uncertainty_table(unc: dict) -> pd.DataFrame:
    """All numeric uncertainty results in one long CSV-friendly frame."""
    overall = pd.DataFrame([{"section": "overall", "metric": k, "value": v}
                            for k, v in unc["overall"].items()])
    per_action = unc["per_action"].assign(section="per_action")
    rel = unc["reliability"].assign(section="reliability_bin")
    return pd.concat([overall, per_action, rel], ignore_index=True)


def plot_reliability_diagram(unc: dict, title: str, path) -> None:
    """Save a reliability diagram (fraction correct per confidence bin)."""
    rel = unc["reliability"]
    filled = rel[rel["count"] > 0]
    width = rel["bin_hi"].iloc[0] - rel["bin_lo"].iloc[0]
    fig, ax = _new_fig(6.4, 5.4)
    ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_2, linewidth=1.2, label="Perfect calibration")
    ax.bar(filled["bin_lo"] + width / 2, filled["accuracy"], width=width,
           color=SERIES[0], edgecolor=SURFACE, linewidth=2, label="Fraction correct")
    ax.plot(filled["mean_confidence"], filled["accuracy"], "o", color=TEXT, markersize=5,
            label="Mean confidence in bin")
    for r in filled.itertuples():
        ax.annotate(f"n={r.count}", (r.bin_lo + width / 2, r.accuracy), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, color=TEXT_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Confidence (max predicted probability)")
    ax.set_ylabel("Fraction correct")
    ax.set_title(f"{title}  (ECE {unc['overall']['ece']:.3f})", color=TEXT, fontsize=12, loc="left")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2, loc="upper left")
    _save(fig, path)


def plot_entropy_histogram(unc: dict, title: str, path) -> None:
    """Save entropy histograms for correct vs incorrect predictions."""
    entropy, correct = unc["entropy"], unc["correct"]
    bins = np.linspace(0, np.log(len(ACTION_LABELS)), 21)
    fig, axes = _new_fig(6.8, 5.0, nrows=2, sharex=True)
    for ax, mask, name, color in [(axes[0], correct, "Correct", SERIES[0]),
                                  (axes[1], ~correct, "Incorrect", SERIES[1])]:
        ax.hist(entropy[mask], bins=bins, color=color, edgecolor=SURFACE, linewidth=2)
        mean = entropy[mask].mean() if mask.any() else np.nan
        ax.set_title(f"{name}  (n={mask.sum()}, mean entropy {mean:.2f})",
                     color=TEXT, fontsize=10, loc="left")
        ax.set_ylabel("Trials")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        _style_axes(ax)
    axes[1].set_xlabel("Predictive entropy (nats; max ln 8 = 2.08)")
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.02, ha="left")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------
def failure_cases(y_true: np.ndarray, proba: np.ndarray, meta: list[dict]) -> pd.DataFrame:
    """One row per misclassified trial, most confident mistakes first."""
    pred = proba.argmax(axis=1)
    rows = []
    for i in np.flatnonzero(pred != y_true):
        order = np.argsort(-proba[i])
        rows.append({
            "subject": meta[i]["subject"], "action_id": meta[i]["action"],
            "trial": meta[i]["trial"],
            "true": ACTION_LABELS[y_true[i]], "predicted": ACTION_LABELS[pred[i]],
            "confidence": proba[i, pred[i]], "p_true": proba[i, y_true[i]],
            "true_rank": int(np.flatnonzero(order == y_true[i])[0]) + 1,
            "runner_up": ACTION_LABELS[order[1]], "p_runner_up": proba[i, order[1]],
        })
    return pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True) \
        if rows else pd.DataFrame(rows)


def format_failure_report(fails: pd.DataFrame, n_total: int, title: str) -> str:
    """Render failure cases and confusion patterns as plain text."""
    lines = [title, "=" * len(title), f"{len(fails)} of {n_total} test trials misclassified "
             f"({len(fails) / n_total:.1%})", ""]
    if fails.empty:
        return "\n".join(lines + ["No misclassifications."])
    lines += ["Confusion patterns (true -> predicted):"]
    groups = fails.groupby(["true", "predicted"]).agg(
        count=("trial", "size"), subjects=("subject", lambda s: ",".join(f"s{x}" for x in sorted(s))),
        mean_conf=("confidence", "mean")).sort_values("count", ascending=False)
    for (t, p), g in groups.iterrows():
        lines.append(f"  {t:>16} -> {p:<16} x{g['count']}  subjects [{g['subjects']}]  "
                     f"mean confidence {g['mean_conf']:.2f}")
    lines += ["", "Misclassified trials (most confident mistakes first):",
              f"  {'trial':<12} {'true':<16} {'predicted':<16} {'conf':>5} {'p(true)':>8} "
              f"{'true rank':>9}  runner-up"]
    for r in fails.itertuples():
        tid = f"a{r.action_id}_s{r.subject}_t{r.trial}"
        lines.append(f"  {tid:<12} {r.true:<16} {r.predicted:<16} {r.confidence:5.2f} "
                     f"{r.p_true:8.2f} {r.true_rank:9d}  {r.runner_up} ({r.p_runner_up:.2f})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Modality contribution
# ---------------------------------------------------------------------------
def top_confusion(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> str:
    """Most common wrong prediction for class c, e.g. 'catch (3)', or ''."""
    wrong = Counter(y_pred[(y_true == c) & (y_pred != c)])
    if not wrong:
        return ""
    label, n = wrong.most_common(1)[0]
    return f"{ACTION_LABELS[label]} ({n})"


def classify_contribution(imu: float, pose: float, fused: float) -> str:
    """Label an action by which modality it needs, using the module thresholds."""
    imu_ok = imu >= SUFFICIENT_ACC and fused - imu <= LITTLE_GAIN
    pose_ok = pose >= SUFFICIENT_ACC and fused - pose <= LITTLE_GAIN
    gain = fused - max(imu, pose)
    if imu_ok and pose_ok:
        return "either sufficient"
    if imu_ok:
        return "IMU sufficient"
    if pose_ok:
        return "pose sufficient"
    if max(imu, pose) < SUFFICIENT_ACC and gain >= CRITICAL_GAIN:
        return "fusion critical"
    if gain > LITTLE_GAIN:
        return "fusion helps"
    if gain < -LITTLE_GAIN:
        return "fusion hurts"
    return "hard for all" if fused < SUFFICIENT_ACC else "no clear gain"


def modality_contribution(acc_by_modality: dict[str, np.ndarray],
                          preds_by_modality: dict[str, np.ndarray],
                          y_true: np.ndarray) -> pd.DataFrame:
    """acc_by_modality: {modality: [n_seeds, num_classes]} per-action accuracy.
    preds_by_modality: {modality: [n_seeds, n_test]} predictions; the most
    common confusion is counted over all seeds pooled. Sorted by fusion gain,
    largest first."""
    n_seeds = next(iter(preds_by_modality.values())).shape[0]
    y_pooled = np.tile(y_true, n_seeds)
    rows = []
    for c, name in enumerate(ACTION_LABELS):
        r = {"action": name}
        for m in MODALITY_NAMES:
            r[f"{m}_acc"] = acc_by_modality[m][:, c].mean()
            r[f"{m}_acc_std"] = acc_by_modality[m][:, c].std(ddof=1)
        r["fusion_gain"] = r["fused_acc"] - max(r["imu_acc"], r["pose_acc"])
        r["category"] = classify_contribution(r["imu_acc"], r["pose_acc"], r["fused_acc"])
        for m in MODALITY_NAMES:
            r[f"{m}_top_confusion"] = top_confusion(y_pooled, preds_by_modality[m].ravel(), c)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("fusion_gain", ascending=False).reset_index(drop=True)


def plot_modality_contribution(contrib: pd.DataFrame, n_seeds: int, path) -> None:
    """Save a grouped bar chart of per-action accuracy by modality."""
    order = contrib.sort_values("fused_acc")["action"].tolist()
    df = contrib.set_index("action").loc[order]
    x = np.arange(len(order))
    width = 0.27
    fig, ax = _new_fig(10, 5.2)
    for k, m in enumerate(MODALITY_NAMES):
        ax.bar(x + (k - 1) * width, 100 * df[f"{m}_acc"], width=width,
               color=MODALITY_COLORS[m], edgecolor=SURFACE, linewidth=2,
               label=MODALITY_NAMES[m])
    ax.set_xticks(x, order, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Per-action test accuracy (%)")
    ax.set_title(f"Random Forest accuracy by action and modality  "
                 f"(mean of {n_seeds} seeds, train s1-4, test s6-8)",
                 color=TEXT, fontsize=12, loc="left", pad=28)
    _style_axes(ax)
    ax.legend(frameon=False, ncols=3, fontsize=9, labelcolor=TEXT_2,
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.3)
    _save(fig, path)
