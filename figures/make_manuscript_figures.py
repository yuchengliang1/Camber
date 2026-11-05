"""Rebuild the quantitative manuscript figures from CAMBER result artifacts.

Usage (from the repository root):
    python figures/make_manuscript_figures.py --source .

The script reads the saved analysis outputs; it does not train a model or alter
the source results. Figure 4 displays the task names corresponding to the
saved 0/1/2 class labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull


BLUE = "#2b6686"
TEAL = "#538d88"
ORANGE = "#bd7653"
GRAY = "#8a9aa3"
INK = "#253642"
GRID = "#dce3e7"
PERFORMANCE_BLUE = "#0072B2"
PERFORMANCE_ORANGE = "#D55E00"
REFERENCE_GRAY = "#626B70"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_close(actual, expected, label, tol=1e-5):
    if not np.isclose(actual, expected, atol=tol, rtol=0):
        raise ValueError(f"{label}: {actual} differs from {expected}")


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save(fig, output: Path, name: str):
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(path)


def panel(ax, letter, title):
    ax.set_title(f"({letter.lower()})  {title}", loc="left", fontweight="bold", pad=8)


def horizontal_estimates(ax, names, values, cis, *, xlim, color=BLUE, baseline=None):
    """Horizontal point-and-interval display used by the other result figures."""
    y = np.arange(len(names))
    values = np.asarray(values, float)
    cis = np.asarray(cis, float)
    ax.errorbar(
        values, y,
        xerr=np.vstack((values - cis[:, 0], cis[:, 1] - values)),
        fmt="o", color=color, ecolor=INK, elinewidth=1.1, capsize=2.5,
        markersize=5, zorder=3,
    )
    if baseline is not None:
        ax.axvline(baseline, color=ORANGE, ls=":", lw=1.2, zorder=1)
    ax.set_yticks(y, names)
    ax.set_xlim(*xlim)
    ax.set_ylim(len(names) - 0.45, -0.55)
    ax.grid(axis="x", color=GRID, lw=0.65, zorder=0)


def vertical_distributions(ax, names, distributions, values, cis, *, ylim,
                           color=PERFORMANCE_BLUE, baseline=None,
                           baseline_color=REFERENCE_GRAY, jitter_seed=0):
    x = np.arange(len(names))
    values = np.asarray(values, float)
    cis = np.asarray(cis, float)
    boxes = ax.boxplot(
        distributions, vert=True, positions=x, widths=0.5, whis=1.5,
        showfliers=False, patch_artist=True, manage_ticks=False, zorder=1,
    )
    for box in boxes["boxes"]:
        box.set(facecolor=color, edgecolor=color, alpha=0.16, linewidth=0.8)
    for median in boxes["medians"]:
        median.set(color=INK, linewidth=1.0)
    for item in (*boxes["whiskers"], *boxes["caps"]):
        item.set(color=GRAY, linewidth=0.8)
    jitter = np.random.default_rng(jitter_seed)
    for index, distribution in enumerate(distributions):
        distribution = np.asarray(distribution, float)
        ax.scatter(
            index + jitter.uniform(-0.12, 0.12, len(distribution)),
            distribution,
            s=8, color=GRAY, alpha=0.58, edgecolors="none", zorder=2,
        )
    ax.errorbar(
        x, values,
        yerr=np.vstack((values - cis[:, 0], cis[:, 1] - values)),
        fmt="D", color=color, ecolor=color, elinewidth=1.2, capsize=2.5,
        markersize=4.6, markeredgecolor="white", markeredgewidth=0.45,
        zorder=3,
    )
    if baseline is not None:
        ax.axhline(baseline, color=baseline_color, ls="--", lw=1.0, zorder=1)
    ax.set_xticks(x, names, rotation=60, ha="right", rotation_mode="anchor", fontsize=6.4)
    ax.tick_params(axis="x", pad=2)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, lw=0.65, zorder=0)


def figure_3(results: Path, output: Path):
    summary = read_json(results / "09_summary/method_summary.json")
    pairs = read_json(results / "09_summary/paired_comparisons.json")["families"]
    subjects = read_csv(results / "09_summary/subject_accuracy.csv")
    methods = summary["methods"]
    full = methods["Full"]["mean_subject_accuracy_percent"]
    assert summary["status"] == "PASS" and summary["n_subjects"] == 12
    check_close(full, 76.38888888888889, "fixed CAMBER accuracy")
    check_close(np.mean([float(r["Full"]) for r in subjects]), full, "participant mean")

    net = [
        ("Broad\nEEG", "Broad_EEGNet_EEG"),
        ("Broad\nMEG", "Broad_EEGNet_MEG"),
        ("Broad\ndecision", "Broad_EEGNet_decision"),
        ("Broad\nfeature", "Broad_EEGNet_concat"),
        ("Band\nEEG", "Band_EEGNet_EEG"),
        ("Band\nMEG", "Band_EEGNet_MEG"),
        ("Band\ndecision", "Band_EEGNet_decision"),
        ("Band\nfeature", "Band_EEGNet_concat"),
    ]
    controls = [
        ("Interaction\nconcat.", "Matched_concat"),
        ("No paths /\ncontroller", "No_path_controller"),
        ("Uniform\ntemporal", "Uniform_temporal"),
        ("Equal branch\ngate", "Fixed_branch_gate"),
        ("Fixed Anchor\nmixing", "Fixed_anchor_mixing"),
        ("Static\ntemporal", "Learned_static_temporal"),
        ("Static\nbranch", "Learned_static_branch"),
        ("Static\nboth", "Learned_static_both"),
        ("No\nAnchor", "No_Anchor_from_scratch"),
    ]
    fig, axs = plt.subplots(
        2, 2, figsize=(5.25, 5.70),
        gridspec_kw={"height_ratios": [1, 1.08]},
        layout="constrained",
    )
    for row, data, family, labels in (
        (0, net, "eegnet_benchmarks", ("A", "B")),
        (1, controls, "architecture_and_selection", ("C", "D")),
    ):
        names = [label for label, _ in data]
        keys = [key for _, key in data]
        values = [methods[k]["mean_subject_accuracy_percent"] for k in keys]
        cis = [methods[k]["accuracy_ci95_percent"] for k in keys]
        pair = {r["comparator"]: r for r in pairs[family]}
        participant_accuracy = [
            np.array([float(record[key]) for record in subjects]) for key in keys
        ]
        participant_difference = [
            np.array([float(record["Full"]) - float(record[key]) for record in subjects])
            for key in keys
        ]
        vertical_distributions(
            axs[row, 0], names, participant_accuracy, values, cis,
            ylim=(35, 95), baseline=full, jitter_seed=20261027 + row,
        )
        vertical_distributions(
            axs[row, 1], names,
            participant_difference,
            [pair[k]["full_minus_comparator_pp"] for k in keys],
            [pair[k]["bootstrap_ci95_pp"] for k in keys],
            ylim=(-3, 29), color=PERFORMANCE_ORANGE, baseline=0,
            jitter_seed=20261101 + row,
        )
        for k, difference in zip(keys, participant_difference):
            check_close(
                full - methods[k]["mean_subject_accuracy_percent"],
                pair[k]["full_minus_comparator_pp"], k,
            )
            check_close(np.mean(difference), pair[k]["full_minus_comparator_pp"],
                        f"participant differences for {k}")
        panel(axs[row, 0], labels[0], "Mean participant accuracy")
        panel(axs[row, 1], labels[1], "Paired CAMBER advantage")
        axs[row, 0].set_ylabel("Accuracy (%)")
        axs[row, 1].set_ylabel("CAMBER minus comparator (pp)")
    save(fig, output, "figure_3_performance.pdf")


def figure_4_confusion(results: Path, output: Path):
    base = results / "01_training_validation/full_and_ablations"
    broad = results / "02_model_comparison/eegnet_broad"
    variants = {
        "CAMBER": "hierarchical_pair_anchor_v25_full",
        "Interaction concat.": "hierarchical_pair_anchor_v25_matched_concat",
        "Equal branch weights": "hierarchical_pair_anchor_v25_fixed_branch_gate",
    }
    matrices = {name: np.zeros((3, 3), dtype=int) for name in variants}
    matrices["Broad EEGNet feature fusion"] = np.zeros((3, 3), dtype=int)
    for number in range(1, 13):
        subject = f"sub{number:02d}"
        reference_y = None
        reference_block = None
        for name, variant in variants.items():
            path = base / subject / variant / "oof_predictions.npz"
            with np.load(path, allow_pickle=False) as archive:
                y = np.asarray(archive["y"], dtype=int)
                block = np.asarray(archive["block"], dtype=int)
                prediction = np.asarray(archive["prediction"], dtype=int)
                probability = np.asarray(archive["probability"], dtype=float)
                covered = np.asarray(archive["covered"], dtype=bool)
            if not (y.shape == block.shape == prediction.shape == covered.shape == (150,)):
                raise ValueError(f"Unexpected OOF vector shape in {path}")
            if probability.shape != (150, 3) or not np.isfinite(probability).all():
                raise ValueError(f"Invalid OOF probabilities in {path}")
            unique_y, class_counts = np.unique(y, return_counts=True)
            if not covered.all() or not np.array_equal(unique_y, [0, 1, 2]) or not np.array_equal(class_counts, [50, 50, 50]):
                raise ValueError(f"Incomplete or unbalanced OOF coverage in {path}")
            unique_block, block_counts = np.unique(block, return_counts=True)
            if not np.array_equal(unique_block, np.arange(5)) or not np.array_equal(block_counts, [30] * 5):
                raise ValueError(f"Unexpected held-out blocks in {path}")
            if not np.isin(prediction, [0, 1, 2]).all() or not np.array_equal(prediction, probability.argmax(axis=1)):
                raise ValueError(f"Saved predictions do not match probabilities in {path}")
            if not np.allclose(probability.sum(axis=1), 1, atol=1e-5, rtol=0):
                raise ValueError(f"Probabilities do not sum to one in {path}")
            if reference_y is None:
                reference_y, reference_block = y, block
            elif not np.array_equal(y, reference_y) or not np.array_equal(block, reference_block):
                raise ValueError(f"Structured OOF variants are not aligned in {subject}")
            np.add.at(matrices[name], (y, prediction), 1)

        with np.load(broad / subject / "oof_predictions.npz", allow_pickle=False) as archive:
            y = np.asarray(archive["y"], dtype=int)
            block = np.asarray(archive["block"], dtype=int)
            prediction = np.asarray(archive["eegnet_feature_concat_prediction"], dtype=int)
            probability = np.asarray(archive["eegnet_feature_concat"], dtype=float)
            coverage = np.asarray(archive["test_coverage"], dtype=int)
        if not (y.shape == block.shape == prediction.shape == coverage.shape == (150,)):
            raise ValueError(f"Unexpected broadband EEGNet OOF shape in {subject}")
        if probability.shape != (150, 3) or not np.isfinite(probability).all():
            raise ValueError(f"Invalid broadband EEGNet probabilities in {subject}")
        if not np.array_equal(coverage, np.ones(150, dtype=int)):
            raise ValueError(f"Incomplete broadband EEGNet OOF coverage in {subject}")
        if not np.array_equal(y, reference_y) or not np.array_equal(block, reference_block):
            raise ValueError(f"Broadband EEGNet OOF trials are not aligned in {subject}")
        if not np.isin(prediction, [0, 1, 2]).all() or not np.array_equal(prediction, probability.argmax(axis=1)):
            raise ValueError(f"Broadband EEGNet predictions do not match probabilities in {subject}")
        if not np.allclose(probability.sum(axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError(f"Broadband EEGNet probabilities do not sum to one in {subject}")
        np.add.at(matrices["Broad EEGNet feature fusion"], (y, prediction), 1)

    audit = read_json(results / "05_nested_selection/independent_audit_and_comparison.json")
    expected = np.asarray(audit["fixed_classification"]["confusion_matrix"], dtype=int)
    if not np.array_equal(matrices["CAMBER"], expected):
        raise ValueError("CAMBER confusion matrix differs from the independent fixed-model audit")
    if any(not np.array_equal(matrix.sum(axis=1), [600, 600, 600]) for matrix in matrices.values()):
        raise ValueError("Expected 600 test trials in each true class for both models")
    check_close(100 * np.trace(matrices["CAMBER"]) / 1800,
                audit["fixed_mean_subject_accuracy_percent"], "confusion accuracy")
    method_summary = read_json(results / "09_summary/method_summary.json")["methods"]
    for name, summary_key in (
        ("Interaction concat.", "Matched_concat"),
        ("Equal branch weights", "Fixed_branch_gate"),
        ("Broad EEGNet feature fusion", "Broad_EEGNet_concat"),
    ):
        check_close(100 * np.trace(matrices[name]) / 1800,
                    method_summary[summary_key]["mean_subject_accuracy_percent"],
                    f"{name} confusion accuracy")

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.8), layout="constrained")
    names = ("CAMBER", "Interaction concat.", "Equal branch weights", "Broad EEGNet feature fusion")
    task_ticks = ("Left-hand\n(0)", "Right-hand\n(1)", "Both-feet\n(2)")
    for index, name in enumerate(names):
        ax = axes.flat[index]
        matrix = matrices[name]
        percent = 100 * matrix / matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100, aspect="equal")
        ax.set_xticks(range(3), task_ticks, fontsize=7.4)
        ax.set_yticks(range(3), task_ticks, fontsize=7.4)
        if index >= 2:
            ax.set_xlabel("Predicted class")
        if index % 2 == 0:
            ax.set_ylabel("True class")
        panel(ax, "ABCD"[index], name)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f"{percent[row, col]:.1f}%",
                        ha="center", va="center", fontsize=10,
                        color="white" if percent[row, col] >= 50 else INK,
                        fontweight="bold" if row == col else "normal")
    bar = fig.colorbar(image, ax=axes.ravel(), shrink=0.8, pad=0.025, ticks=[0, 25, 50, 75, 100])
    bar.set_label("Within true class (%)")
    save(fig, output, "figure_4_confusion.pdf")


def figure_5(results: Path, output: Path):
    sensitivity = read_json(results / "09_summary/parameter_sensitivity_summary.json")["configs"]
    audit = read_json(results / "05_nested_selection/independent_audit_and_comparison.json")
    rows = read_csv(results / "05_nested_selection/subject_comparison.csv")
    names = [
        ("LR 0.006", "fixed_global12_lr006"),
        ("LR 0.018", "fixed_global12_lr018"),
        ("80 epochs", "fixed_global12_epochs80"),
        ("160 epochs", "fixed_global12_epochs160"),
        ("Dropout 0.15", "fixed_global12_dropout015"),
        ("Dropout 0.30", "fixed_global12_dropout030"),
        ("Decay 0.0003", "fixed_global12_wd0003"),
        ("Decay 0.03", "fixed_global12_wd003"),
    ]
    fixed = np.array([float(r["fixed_accuracy_percent"]) for r in rows])
    tuned = np.array([float(r["tuned_accuracy_percent"]) for r in rows])
    assert len(rows) == 12 and audit["audit_status"] == "PASS"
    check_close(fixed.mean(), audit["fixed_mean_subject_accuracy_percent"], "nested fixed")
    check_close(tuned.mean(), audit["tuned_mean_subject_accuracy_percent"], "nested selected")
    check_close(fixed.mean(), sensitivity["fixed_global12_baseline"]["mean_accuracy_percent"], "fixed sensitivity")

    fig = plt.figure(figsize=(8.8, 4.3), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.22, 0.7])
    ax = fig.add_subplot(gs[0, 0])
    means = [sensitivity[k]["delta_vs_fixed_baseline_pp"] for _, k in names]
    ci = [sensitivity[k]["delta_ci95_pp"] for _, k in names]
    horizontal_estimates(ax, [n for n, _ in names], means, ci, xlim=(-2.6, 3.0), color=BLUE, baseline=0)
    panel(ax, "A", "Single-factor sensitivity")
    ax.set_xlabel("Setting minus fixed (pp)")
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(12)
    ax.plot(x, fixed, "o-", color=BLUE, ms=4, lw=1, label=f"Fixed ({fixed.mean():.2f}%)")
    ax.plot(x, tuned, "o-", color=ORANGE, ms=4, lw=1, label=f"Inner-selected ({tuned.mean():.2f}%)")
    ax.set_xticks(x, [str(r["subject"])[-2:] for r in rows], rotation=90)
    ax.set_ylim(max(0, min(fixed.min(), tuned.min()) - 6), min(100, max(fixed.max(), tuned.max()) + 5))
    ax.set_xlabel("Participant ID suffix")
    ax.set_ylabel("Outer-test accuracy (%)")
    ax.grid(axis="y", color=GRID, lw=0.65)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    panel(ax, "B", "Retrospective nested selection")
    ax = fig.add_subplot(gs[0, 2])
    diff = tuned - fixed
    ax.scatter(np.zeros(12) + np.linspace(-0.08, 0.08, 12), diff, s=18, color=GRAY, zorder=2)
    mean = float(diff.mean())
    lo, hi = audit["paired_subject_bootstrap_ci95_pp"]
    ax.errorbar(0, mean, yerr=[[mean - lo], [hi - mean]], fmt="D", color=ORANGE, ms=6, capsize=4, zorder=3)
    ax.axhline(0, color=INK, ls=":", lw=1)
    ax.set_xlim(-0.27, 0.27)
    ax.set_xticks([])
    ax.set_ylabel("Nested minus fixed (pp)")
    ax.grid(axis="y", color=GRID, lw=0.65)
    ax.text(0.02, 0.02, f"Mean {mean:+.2f} pp\nExact p = {audit['paired_subject_exact_signflip_p_two_sided']:.3f}", transform=ax.transAxes, fontsize=7, va="bottom")
    panel(ax, "C", "Paired difference")
    save(fig, output, "figure_5_parameters.pdf")


def paired_correctness_matrix(ax, matrix, letter, title, net_trials, total_trials):
    """Draw a paired correct/incorrect table with semantic cell colors."""
    fills = (
        ("#edf0f2", PERFORMANCE_ORANGE),
        (PERFORMANCE_BLUE, "#d4dde2"),
    )
    for row in range(2):
        for column in range(2):
            y = 1 - row
            ax.add_patch(
                Rectangle(
                    (column, y), 1, 1,
                    facecolor=fills[row][column], edgecolor="white", linewidth=1.8,
                )
            )
            count = int(matrix[row, column])
            percent = 100 * count / total_trials
            text_color = "white" if row != column else INK
            ax.text(
                column + 0.5, y + 0.5, f"{count}\n({percent:.1f}%)",
                ha="center", va="center", color=text_color,
                fontsize=8.2, fontweight="bold",
            )
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5], ["Comparator\nincorrect", "Comparator\ncorrect"])
    ax.set_yticks([1.5, 0.5], ["CAMBER\nincorrect", "CAMBER\ncorrect"])
    ax.tick_params(length=0, labelsize=7.2, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    net_pp = 100 * net_trials / total_trials
    ax.set_xlabel(
        f"Net CAMBER advantage\n+{net_trials} trials (+{net_pp:.2f} pp)",
        fontsize=7.2, labelpad=5,
    )
    panel(ax, letter, title)


def figure_6(results: Path, output: Path):
    broad = read_json(results / "08_complementarity/broad_comparison_summary.json")
    matched = read_json(results / "08_complementarity/summary.json")
    check_close(broad["full_mean_accuracy_percent"], matched["mean_subject_accuracy_percent"]["full"], "CAMBER reference")
    assert broad["n_trials"] == matched["trials"] == 1800
    total = broad["n_trials"]
    full_correct = round(total * broad["full_mean_accuracy_percent"] / 100)
    broad_feature = next(item for item in broad["methods"] if item["method"] == "eegnet_feature_concat")
    broad_correct = round(total * broad_feature["mean_accuracy_percent"] / 100)
    matched_correct = round(total * matched["mean_subject_accuracy_percent"]["concat"] / 100)

    def contingency(comparator_correct, full_only, comparator_only):
        both_correct = full_correct - full_only
        both_incorrect = total - both_correct - full_only - comparator_only
        matrix = np.array([[both_incorrect, comparator_only], [full_only, both_correct]])
        assert matrix.sum() == total
        assert both_correct + comparator_only == comparator_correct
        return matrix

    broad_matrix = contingency(
        broad_correct,
        broad["full_correct_broad_concat_wrong_total"],
        broad["broad_concat_correct_full_wrong_total"],
    )
    matched_matrix = contingency(
        matched_correct,
        matched["total_full_correct_concat_wrong_n"],
        matched["total_concat_correct_full_wrong_n"],
    )
    assert broad_matrix.tolist() == [[324, 101], [383, 992]]
    assert matched_matrix.tolist() == [[379, 46], [91, 1284]]

    fig = plt.figure(figsize=(5.25, 3.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 2.05], hspace=0.08, wspace=0.18)
    ax_a = fig.add_subplot(gs[0, :])
    wrong = broad["both_broad_unimodal_wrong_total"]
    rescued = broad["full_rescues_both_broad_wrong_total"]
    assert wrong == 530 and rescued == 253
    remaining = wrong - rescued
    rescued_pct = 100 * rescued / wrong
    ax_a.barh(0, rescued_pct, color=PERFORMANCE_BLUE, height=0.48)
    ax_a.barh(0, 100 - rescued_pct, left=rescued_pct, color=GRAY, height=0.48)
    ax_a.text(rescued_pct / 2, 0, f"CAMBER correct\n{rescued} ({rescued_pct:.1f}%)",
              ha="center", va="center", color="white", fontsize=8, weight="bold")
    ax_a.text((100 + rescued_pct) / 2, 0, f"Still incorrect\n{remaining} ({100 - rescued_pct:.1f}%)",
              ha="center", va="center", color="white", fontsize=8, weight="bold")
    ax_a.set_yticks([])
    ax_a.set_ylim(-0.4, 0.4)
    ax_a.set_xlim(0, 100)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_xlabel("Share of shared unimodal errors (%)")
    ax_a.grid(axis="x", color=GRID, lw=0.65)
    ax_a.set_axisbelow(True)
    panel(ax_a, "A", f"Recovery of shared broadband unimodal errors (n={wrong})")

    ax_b = fig.add_subplot(gs[1, 0])
    paired_correctness_matrix(
        ax_b, broad_matrix, "B", "CAMBER vs broadband\nfeature fusion",
        int(broad_matrix[1, 0] - broad_matrix[0, 1]), total,
    )
    ax_c = fig.add_subplot(gs[1, 1])
    paired_correctness_matrix(
        ax_c, matched_matrix, "C", "CAMBER vs matched\ninteraction concatenation",
        int(matched_matrix[1, 0] - matched_matrix[0, 1]), total,
    )
    save(fig, output, "figure_6_complementarity.pdf")


def figure_7(results: Path, output: Path):
    directory = results / "06_spatial_interpretability/local_power"
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "PASS" and summary["subjects"] == 12
    rows = [item for i in range(1, 13) for item in read_json(directory / f"sub{i:02d}.json")]
    assert len(rows) == 120
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
    rng = np.random.default_rng(20261027)
    for i, mod in enumerate(("EEG", "MEG")):
        modrows = [r for r in rows if r["modality"] == mod]
        acc = np.array([np.mean([r["probe_test_accuracy_percent"] for r in modrows if r["subject"] == sid]) for sid in range(29101, 29113)])
        rho = np.array([np.mean([r["spearman_time_score_vs_weight"] for r in modrows if r["subject"] == sid]) for sid in range(29101, 29113)])
        check_close(acc.mean(), summary["modalities"][mod]["mean_probe_test_accuracy_percent"], mod + " probe")
        check_close(rho.mean(), summary["modalities"][mod]["mean_subject_spearman"], mod + " rho")
        color = BLUE if mod == "EEG" else ORANGE
        y = 1 - i
        axes[0].scatter(acc, y + rng.uniform(-0.14, 0.14, len(acc)), color=color, s=21, alpha=0.65, zorder=2)
        boots = rng.integers(0, 12, size=(10000, 12))
        lo, hi = np.quantile(acc[boots].mean(axis=1), [0.025, 0.975])
        axes[0].errorbar(acc.mean(), y, xerr=[[acc.mean() - lo], [hi - acc.mean()]], fmt="D", color=INK, ms=5, capsize=3, zorder=4)
        axes[1].scatter(rho, y + rng.uniform(-0.14, 0.14, len(rho)), color=color, s=21, alpha=0.65, zorder=2)
        lo, hi = summary["modalities"][mod]["ci95"]
        axes[1].errorbar(rho.mean(), y, xerr=[[rho.mean() - lo], [hi - rho.mean()]], fmt="D", color=INK, ms=5, capsize=3, zorder=4)
    for ax in axes:
        ax.set_yticks([1, 0], ["EEG", "MEG"])
        ax.set_ylim(-0.42, 1.42)
        ax.grid(axis="x", color=GRID, lw=0.65)
        ax.set_axisbelow(True)
    axes[0].axvline(100 / 3, color=GRAY, ls=":", lw=1.2)
    axes[0].set_xlim(30, 82)
    axes[0].set_xlabel("Held-out probe accuracy (%)")
    panel(axes[0], "A", "Local-power decoding")
    axes[1].axvline(0, color=GRAY, ls=":", lw=1.2)
    axes[1].set_xlim(-0.34, 0.38)
    axes[1].set_xlabel("Spearman $\\rho$ with temporal weights")
    panel(axes[1], "B", "Temporal concordance")
    save(fig, output, "figure_7_power.pdf")


def fitted_display_circle(xy, padding=0.012):
    """Return the vertically shifted circle enclosing projected sensors."""
    fit = minimize_scalar(
        lambda cy: np.max(np.hypot(xy[:, 0], xy[:, 1] - cy)),
        bounds=(-1, 1), method="bounded",
    )
    return fit.x, fit.fun + padding


def head(ax, *, center_y=0.0, radius=1.0):
    ax.add_patch(Circle((0, center_y), radius, fill=False, ec=INK, lw=0.8))
    ax.plot(
        np.array([-0.095, 0, 0.095]) * radius,
        center_y + np.array([1, 1.11, 1]) * radius,
        color=INK, lw=0.8,
    )
    for sign in (-1, 1):
        ax.plot(
            np.array([1, 1.07, 1.07, 1]) * radius * sign,
            center_y + np.array([0.11, 0.11, -0.11, -0.11]) * radius,
            color=INK, lw=0.8,
        )
    ax.set_xlim(-1.12 * radius, 1.12 * radius)
    ax.set_ylim(center_y - 1.1 * radius, center_y + 1.15 * radius)
    ax.set_aspect("equal")
    ax.axis("off")


def spatial_input(results: Path, data: Path):
    with np.load(results / "06_spatial_interpretability/spatial_audit_arrays.npz", allow_pickle=False) as z:
        occ = z["test_occlusion"].mean(axis=0)
        power = z["power_class"].mean(axis=0)
    pos = read_json(data / "channel_positions.json")["channels"]
    positions = {}
    for mod in ("eeg", "meg"):
        channel = [r for r in pos if r["modality"] == mod and r["has_display_position"]]
        positions[mod] = (
            np.asarray([r["array_channel_index0"] for r in channel], dtype=int),
            np.asarray([r["display_xy_head_radius"] for r in channel], dtype=float),
        )
    assert len(positions["eeg"][0]) == 62 and len(positions["meg"][0]) == 61
    return occ, power, positions


def draw_map(ax, ids, xy, values, *, vmin, vmax, cmap, title, marker_size=4,
             title_weight="normal"):
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    channel_values = np.asarray(values)[ids]
    averaged = np.array([channel_values[inverse == k].mean() for k in range(len(unique))])
    center_y, radius = fitted_display_circle(xy)
    grid = np.linspace(-1.0, 1.0, 250)
    gx, gy = np.meshgrid(grid, grid)
    interpolation = LinearNDInterpolator(unique, averaged)
    zz = interpolation(gx, gy)
    # Preserve the sensor-supported field. Extend only the annulus between its
    # convex hull and the circular display edge, continuously from the hull.
    hull = unique[ConvexHull(unique).vertices]
    outside = ~np.isfinite(zz)
    query = np.column_stack((gx[outside], gy[outside]))
    nearest = np.empty_like(query)
    best_distance2 = np.full(len(query), np.inf)
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        edge = end - start
        fraction = np.clip(((query - start) @ edge) / (edge @ edge), 0, 1)
        candidate = start + fraction[:, None] * edge
        distance2 = np.sum((query - candidate) ** 2, axis=1)
        closer = distance2 < best_distance2
        best_distance2[closer] = distance2[closer]
        nearest[closer] = candidate[closer]
    # Nudge hull-edge points inward to avoid roundoff returning NaNs.
    inner_edge = 0.999999 * nearest + 0.000001 * unique.mean(axis=0)
    boundary_values = interpolation(inner_edge[:, 0], inner_edge[:, 1])
    theta = np.arctan2(query[:, 1] - center_y, query[:, 0])
    edge_query = np.column_stack((radius * np.cos(theta),
                                  center_y + radius * np.sin(theta)))
    distances2 = np.sum((edge_query[:, None, :] - unique[None, :, :]) ** 2, axis=2)
    weights = np.exp(-0.5 * distances2 / 0.27 ** 2)
    edge_values = (weights @ averaged) / weights.sum(axis=1)
    gap = radius - np.hypot(nearest[:, 0], nearest[:, 1] - center_y)
    fraction = np.clip(np.sqrt(best_distance2) / np.maximum(gap, 0.01), 0, 1)
    zz[outside] = (1 - fraction) * boundary_values + fraction * edge_values
    inside_circle = gx ** 2 + (gy - center_y) ** 2 <= radius ** 2
    if not np.isfinite(zz[inside_circle]).all():
        raise ValueError("Circular map has an unfilled pixel")
    if (zz[inside_circle].min() < averaged.min() - 1e-6 or
            zz[inside_circle].max() > averaged.max() + 1e-6):
        raise ValueError("Circular map exceeds the displayed sensor range")
    image = ax.imshow(zz, origin="lower", extent=(-1.0, 1.0, -1.0, 1.0), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear", zorder=-1, rasterized=True)
    boundary = Circle((0, center_y), radius, fill=False, edgecolor=INK, linewidth=0.8, zorder=5)
    ax.add_patch(boundary)
    image.set_clip_path(boundary)
    ax.scatter(xy[:, 0], xy[:, 1], s=marker_size, c=INK, alpha=0.65, zorder=3)
    top = center_y + radius
    ax.annotate("Anterior", xy=(0, top + 0.20), xytext=(0, top + 0.06), ha="center", va="center", fontsize=6.5,
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8})
    ax.set_xlim(-0.92, 0.92)
    ax.set_ylim(-1.03, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8.4, fontweight=title_weight, pad=4)
    return image


def regional_subject_excess(results: Path, metric: str):
    path = (results / "06_spatial_interpretability" /
            "common_orientation_projection_assumed" / "matched_null64" /
            "regional_occlusion_fold.csv")
    grouped = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for row in read_csv(path):
        key = (row["modality"], row["region"], int(row["subject"]))
        value = float(row[metric])
        if row["kind"] == "region":
            grouped[key][0] += value
            grouped[key][1] += 1
        elif row["kind"] == "matched_random":
            grouped[key][2] += value
            grouped[key][3] += 1
        else:
            raise ValueError(f"Unexpected regional occlusion kind: {row['kind']}")
    result = defaultdict(list)
    for (modality, region, subject), (actual_sum, actual_n, random_sum, random_n) in grouped.items():
        if actual_n != 5 or random_n != 320:
            raise ValueError(
                f"Incomplete regional effects for {modality} {region} participant {subject}: "
                f"{actual_n} regional and {random_n} random observations"
            )
        result[(modality, region)].append(
            (subject, actual_sum / actual_n - random_sum / random_n)
        )
    subject_values = {}
    for key, pairs in result.items():
        pairs.sort()
        if len(pairs) != 12 or len({subject for subject, _ in pairs}) != 12:
            raise ValueError(f"Expected 12 participant effects for {key}")
        subject_values[key] = np.array([value for _, value in pairs])
    return subject_values


def regional_panel(ax, rows, subject_values, metric, letter, title, unit):
    subset = [r for r in rows if r["metric"] == metric]
    assert len(subset) == 12
    values = np.array([float(r["excess_effect"]) for r in subset])
    lower = np.array([float(r["ci95_lower"]) for r in subset])
    upper = np.array([float(r["ci95_upper"]) for r in subset])
    distributions = [subject_values[(r["modality"], r["region"])] for r in subset]
    y = np.arange(12)
    boxes = ax.boxplot(
        distributions, vert=False, positions=y, widths=0.46, whis=1.5,
        showfliers=False, patch_artist=True, manage_ticks=False, zorder=1,
    )
    for box, row in zip(boxes["boxes"], subset):
        color = BLUE if row["modality"] == "eeg" else TEAL
        box.set(facecolor=color, edgecolor=color, alpha=0.14, linewidth=0.8)
    for median in boxes["medians"]:
        median.set(color=INK, linewidth=1.0)
    for item in (*boxes["whiskers"], *boxes["caps"]):
        item.set(color=GRAY, linewidth=0.8)
    jitter = np.random.default_rng(20261104)
    for i, (r, distribution) in enumerate(zip(subset, distributions)):
        check_close(np.mean(distribution), values[i],
                    f"regional participant mean for {r['modality']} {r['region']}")
        ax.scatter(
            distribution, i + jitter.uniform(-0.11, 0.11, len(distribution)),
            s=7, color=GRAY, alpha=0.58, edgecolors="none", zorder=2,
        )
        sig = float(r["holm_p_12_regions"]) < 0.05
        ax.errorbar(values[i], y[i], xerr=[[values[i] - lower[i]], [upper[i] - values[i]]],
                    fmt="D", ms=4.2, capsize=1.8, color=ORANGE if sig else BLUE,
                    markerfacecolor=ORANGE if sig else "white", markeredgecolor=ORANGE if sig else BLUE,
                    ecolor=INK, zorder=3)
    ax.axvline(0, color=GRAY, ls=":", lw=1)
    ax.set_yticks(y, [r["modality"].upper() + " " + r["region"] for r in subset], fontsize=6.7)
    ax.set_ylim(11.4, -0.5)
    ax.set_xlabel(unit, fontsize=7.4)
    ax.grid(axis="x", color=GRID, lw=0.6)
    panel(ax, letter, title)


def figure_8(results: Path, data: Path, output: Path):
    occ, _, positions = spatial_input(results, data)
    regions = read_csv(results / "06_spatial_interpretability/regional_effects_table.csv")
    subject_effects = regional_subject_excess(results, "predicted_logprob_drop")
    summary = read_json(results / "09_summary/final_spatial_summary.json")
    assert summary["status"] == "PASS" and summary["subjects"] == 12
    fig = plt.figure(figsize=(7.2, 5.8), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.08], height_ratios=[1, 1.2])
    median = float(np.median(occ))
    centered = occ - median
    limit = float(np.max(np.abs(centered)))
    for i, mod in enumerate(("eeg", "meg")):
        ax = fig.add_subplot(gs[0, i])
        ids, xy = positions[mod]
        im = draw_map(ax, ids, xy, centered[i], vmin=-limit, vmax=limit, cmap="RdBu_r",
                      title=("(a)" if i == 0 else "(b)") + "  " + mod.upper() + " channel sensitivity",
                      marker_size=7, title_weight="bold")
    cax = fig.add_subplot(gs[0, 2])
    fig.colorbar(im, cax=cax, label="Drop relative to pooled median")
    ax = fig.add_subplot(gs[1, :])
    regional_panel(ax, regions, subject_effects, "predicted_logprob_drop", "C", "Regional excess effect",
                   "Excess drop in predicted-class log probability")
    ax.tick_params(axis="y", labelsize=8)
    ax.axhline(5.5, color=GRID, lw=0.9)
    ax.set_xlim(-0.18, 0.17)
    save(fig, output, "figure_8_spatial.pdf")


def figure_s1(results: Path, data: Path, output: Path):
    _, power, positions = spatial_input(results, data)
    # Mean over participants; channelwise range across the three analysis classes.
    ranges = power.max(axis=2) - power.min(axis=2)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.1), layout="constrained")
    titles = (("EEG", "MEG"), ("EEG", "MEG"))
    for band, band_name in enumerate(("mu", "beta")):
        median = float(np.nanmedian(ranges[:, band, :]))
        centered = ranges[:, band, :] - median
        limit = float(np.nanmax(np.abs(centered)))
        for col, mod in enumerate(("eeg", "meg")):
            ids, xy = positions[mod]
            letter = chr(ord("a") + 2 * band + col)
            image = draw_map(
                axes[band, col], ids, xy, centered[col], vmin=-limit, vmax=limit,
                cmap="RdBu_r", title=f"({letter})  {titles[band][col]} {band_name} class range",
            )
            fig.colorbar(image, ax=axes[band, col], shrink=0.7, pad=0.01, label="Centered class range (dB)")
    fig.suptitle("Task-interval power: range across analysis-class means", fontsize=10, weight="bold")
    save(fig, output, "figure_S1_task_power.pdf")


def figure_s2(results: Path, data: Path, output: Path):
    _, _, positions = spatial_input(results, data)
    fig, ax = plt.subplots(figsize=(3.8, 3.6), layout="constrained")
    combined_xy = np.vstack([positions[mod][1] for mod in ("eeg", "meg")])
    # Centre the pooled vertical extent, then enclose every open marker with
    # enough clearance for its edge as well as its coordinate.
    center_y = 0.5 * (combined_xy[:, 1].min() + combined_xy[:, 1].max())
    radius = np.max(np.hypot(combined_xy[:, 0], combined_xy[:, 1] - center_y)) + 0.045
    head(ax, center_y=center_y, radius=radius)
    for mod, marker, color in (("eeg", "o", BLUE), ("meg", "^", ORANGE)):
        _, xy = positions[mod]
        ax.scatter(xy[:, 0], xy[:, 1], s=20, marker=marker, fc="none", ec=color,
                   lw=0.85, label=mod.upper(), zorder=3)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, output, "figure_S2_sensor_projection.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository-like root containing results/ and data/channel_positions.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Destination directory for generated PDF figures.",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    results = source / "results"
    data = source / "data"
    if not results.is_dir() or not data.is_dir():
        parser.error(f"Expected results/ and data/ under {source}")
    style()
    figure_3(results, args.output)
    figure_4_confusion(results, args.output)
    figure_5(results, args.output)
    figure_6(results, args.output)
    figure_7(results, args.output)
    figure_8(results, data, args.output)
    figure_s1(results, data, args.output)
    figure_s2(results, data, args.output)


if __name__ == "__main__":
    main()
