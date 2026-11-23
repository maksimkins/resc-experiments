r"""Generate paper-ready figures from completed with-categorical benchmark results.

Run from ``resc-experiments`` with::

    .\.venv\Scripts\python.exe scripts\generate_paper_figures.py

PDF files are written directly to ``latex/figures``. PNG copies are written to
the same directory for quick visual inspection.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
METRICS_ROOT = EXPERIMENT_ROOT / "results" / "metrics"
STATISTICS_ROOT = METRICS_ROOT / "statistical_tests_with_c"
FIGURE_ROOT = PROJECT_ROOT / "latex" / "figures"

EXPECTED_BATCHES = 10
EXPECTED_DATASETS = 30
EXPECTED_FOLDS = 100
PROPOSED_METHOD = "kmeans_resc_q10"
BASELINE_METHOD = "resc"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"

CLASSIFIER_LABELS = {"mlp": "MLP", "rbf_svm": "RBF-SVM"}
METRIC_LABELS = {
    "roc_auc": "AUC",
    "recall": "Recall",
    "gmean": "G-mean",
    "f1": "F1",
}
COMPARATOR_ORDER = (
    "none",
    "random_under",
    "cluster_centroids",
    "smote",
    "smote_enn",
    "adasyn",
    "resc",
)
COMPARATOR_LABELS = {
    "none": "Original",
    "random_under": "RUS",
    "cluster_centroids": "ClusterCentroids",
    "smote": "SMOTE",
    "smote_enn": "SMOTE-ENN",
    "adasyn": "ADASYN",
    "resc": "Re-SC",
}


def configure_style() -> None:
    """Use compact, print-safe defaults suited to a two-column article."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def batch_number(path: Path) -> int:
    match = re.search(r"batch(\d+)of10", str(path))
    if match is None:
        raise ValueError(f"Cannot identify batch number from {path}")
    return int(match.group(1))


def load_batched_timings() -> pd.DataFrame:
    paths = sorted(
        METRICS_ROOT.glob(
            "benchmark_fixed_m15_with_c_10x10_batch*/with_c/resampling_timings.csv"
        ),
        key=batch_number,
    )
    if len(paths) != EXPECTED_BATCHES:
        raise FileNotFoundError(
            f"Expected {EXPECTED_BATCHES} timing files, found {len(paths)}"
        )
    observed_batches = [batch_number(path) for path in paths]
    if observed_batches != list(range(1, EXPECTED_BATCHES + 1)):
        raise ValueError(f"Missing or duplicate benchmark batches: {observed_batches}")
    timings = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    timings = timings.loc[
        timings["method"].isin((BASELINE_METHOD, PROPOSED_METHOD))
    ].copy()
    counts = timings.groupby(["dataset_key", "method"]).size()
    if timings["dataset_key"].nunique() != EXPECTED_DATASETS:
        raise ValueError(
            f"Expected {EXPECTED_DATASETS} datasets, "
            f"found {timings['dataset_key'].nunique()}"
        )
    if not counts.eq(EXPECTED_FOLDS).all():
        raise ValueError(f"Expected {EXPECTED_FOLDS} folds per dataset and method")
    return timings


def load_dataset_labels() -> dict[str, str]:
    manifest_path = METRICS_ROOT / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = {
        entry["dataset_key"]: entry.get("source_name") or entry["dataset_key"]
        for entry in manifest["datasets"]
    }
    return labels


def label_for(dataset_key: str, labels: dict[str, str]) -> str:
    return str(labels.get(dataset_key, dataset_key)).replace("_", "-")


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_ROOT / f"{stem}.pdf"
    png_path = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def plot_gmean_differences(
    dataset_means: pd.DataFrame, labels: dict[str, str]
) -> None:
    required = {"classifier", "dataset_key", "method", "gmean"}
    if not required.issubset(dataset_means.columns):
        raise ValueError(f"dataset_method_means.csv lacks {required - set(dataset_means)}")

    differences: dict[str, pd.Series] = {}
    for classifier in CLASSIFIER_LABELS:
        selected = dataset_means.loc[
            (dataset_means["classifier"] == classifier)
            & dataset_means["method"].isin((BASELINE_METHOD, PROPOSED_METHOD)),
            ["dataset_key", "method", "gmean"],
        ]
        pivot = selected.pivot(index="dataset_key", columns="method", values="gmean")
        if len(pivot) != EXPECTED_DATASETS or pivot.isna().any().any():
            raise ValueError(f"Incomplete G-mean data for {classifier}")
        differences[classifier] = pivot[PROPOSED_METHOD] - pivot[BASELINE_METHOD]

    max_abs = max(float(series.abs().max()) for series in differences.values())
    limit = max_abs * 1.08
    difference_frame = pd.DataFrame(differences)
    difference_frame["sort_value"] = difference_frame.mean(axis=1)
    difference_frame = difference_frame.sort_values("sort_value", kind="stable")

    y = np.arange(len(difference_frame))
    mlp = difference_frame["mlp"].to_numpy()
    rbf_svm = difference_frame["rbf_svm"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.hlines(y, mlp, rbf_svm, color=LIGHT_GRAY, linewidth=0.8, zorder=1)
    ax.scatter(mlp, y - 0.13, s=20, marker="o", color=BLUE, label="MLP", zorder=3)
    ax.scatter(
        rbf_svm,
        y + 0.13,
        s=20,
        marker="s",
        color=ORANGE,
        label="RBF-SVM",
        zorder=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_yticks(
        y,
        [label_for(key, labels) for key in difference_frame.index],
        fontsize=5.8,
    )
    ax.set_xlabel(r"$\Delta$ G-mean (KMeans-ReSC $-$ Re-SC)")
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.09, top=0.99)
    save_figure(fig, "gmean_dataset_differences")


def aggregate_ir(timings: pd.DataFrame) -> pd.DataFrame:
    source = timings.copy()
    source["before_ir"] = source["train_majority"] / source["train_minority"]
    source["after_ir"] = (
        source["resampled_majority"] / source["resampled_minority"]
    )
    before = source.groupby("dataset_key")["before_ir"].mean().rename("Before")
    after = (
        source.groupby(["dataset_key", "method"])["after_ir"]
        .mean()
        .unstack("method")
        .rename(columns={BASELINE_METHOD: "Re-SC", PROPOSED_METHOD: "KMeans-ReSC"})
    )
    result = before.to_frame().join(after[["Re-SC", "KMeans-ReSC"]])
    if len(result) != EXPECTED_DATASETS or result.isna().any().any():
        raise ValueError("Incomplete imbalance-ratio data")
    return result.sort_values("Before", kind="stable")


def plot_ir_comparison(ir: pd.DataFrame, labels: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    for _, row in ir.iterrows():
        ax.vlines(
            float(row["Before"]),
            float(min(row["Re-SC"], row["KMeans-ReSC"])),
            float(max(row["Re-SC"], row["KMeans-ReSC"])),
            color=LIGHT_GRAY,
            linewidth=0.7,
            zorder=1,
        )

    ax.scatter(ir["Before"], ir["Re-SC"], marker="o", s=22, color=BLUE, label="Re-SC", zorder=3)
    ax.scatter(
        ir["Before"],
        ir["KMeans-ReSC"],
        marker="s",
        s=22,
        color=ORANGE,
        label="KMeans-ReSC",
        zorder=3,
    )
    ax.axhline(1, color=GREEN, linestyle="--", linewidth=1.0, label="Parity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    x_ticks = [2, 5, 10, 20, 30]
    y_ticks = [0.5, 1, 2]
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(1.6, 32)
    ax.set_ylim(0.45, 2.4)
    ax.set_xlabel("IR before resampling")
    ax.set_ylabel("IR after resampling")
    ax.grid(which="major", color=LIGHT_GRAY, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=6)

    # Label only the stated exception and the strongest post-resampling overshoot.
    exception_keys = {
        key for key in ir.index if label_for(key, labels).lower() == "breast-w"
    }
    exception_keys.add(ir["KMeans-ReSC"].idxmin())
    for key in sorted(exception_keys):
        row = ir.loc[key]
        ax.annotate(
            label_for(key, labels),
            (row["Before"], row["KMeans-ReSC"]),
            xytext=(3, -8),
            textcoords="offset points",
            fontsize=5.5,
        )

    fig.tight_layout(pad=0.5)
    save_figure(fig, "imbalance_ratio_before_after")


def aggregate_tradeoff(timings: pd.DataFrame) -> pd.DataFrame:
    grouped = timings.groupby(["dataset_key", "method"])[
        ["resample_seconds", "representative_count"]
    ].mean()
    time = grouped["resample_seconds"].unstack("method")
    representatives = grouped["representative_count"].unstack("method")
    if (time[BASELINE_METHOD] <= 0).any():
        raise ValueError("Re-SC timing contains a non-positive value")
    if (representatives[BASELINE_METHOD] <= 0).any():
        raise ValueError("Re-SC representative count contains a non-positive value")
    result = pd.DataFrame(
        {
            "time_ratio": time[PROPOSED_METHOD] / time[BASELINE_METHOD],
            "representative_ratio": representatives[PROPOSED_METHOD]
            / representatives[BASELINE_METHOD],
        }
    )
    if len(result) != EXPECTED_DATASETS or result.isna().any().any():
        raise ValueError("Incomplete timing or representative-count data")
    return result


def plot_tradeoff(tradeoff: pd.DataFrame, labels: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.scatter(
        tradeoff["time_ratio"],
        tradeoff["representative_ratio"],
        s=35,
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.axvline(1, color="black", linestyle="--", linewidth=0.9)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.9)
    ax.grid(color=LIGHT_GRAY, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Resampling-time ratio (KMeans-ReSC / Re-SC; log scale)")
    ax.set_ylabel("Representative-count ratio (KMeans-ReSC / Re-SC)")

    # Label only three isolated extremes. Labelling all 30 datasets obscures the
    # trade-off pattern, while full dataset-level values remain in the CSV files.
    extreme_keys = set(tradeoff.nlargest(2, "time_ratio").index)
    extreme_keys.add(tradeoff["representative_ratio"].idxmax())
    for key in sorted(extreme_keys):
        row = tradeoff.loc[key]
        ax.annotate(
            label_for(key, labels),
            (row["time_ratio"], row["representative_ratio"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
        )

    fig.tight_layout()
    save_figure(fig, "resampling_time_representative_tradeoff")


def plot_wilcoxon_forest(holm: pd.DataFrame) -> None:
    required = {
        "classifier",
        "metric",
        "comparison_method",
        "mean_difference",
        "mean_difference_ci_low",
        "mean_difference_ci_high",
        "reject_at_0_05",
    }
    if not required.issubset(holm.columns):
        raise ValueError(f"holm_wilcoxon_results.csv lacks {required - set(holm)}")

    fig, axes = plt.subplots(2, 4, figsize=(6.8, 5.2), sharey=True)
    y = np.arange(len(COMPARATOR_ORDER))

    metric_limits: dict[str, float] = {}
    for metric in METRIC_LABELS:
        metric_rows = holm.loc[holm["metric"] == metric]
        extent = metric_rows[
            ["mean_difference_ci_low", "mean_difference_ci_high"]
        ].abs().to_numpy().max()
        metric_limits[metric] = max(float(extent) * 1.08, 0.01)

    for row_index, classifier in enumerate(CLASSIFIER_LABELS):
        for column_index, metric in enumerate(METRIC_LABELS):
            ax = axes[row_index, column_index]
            panel = (
                holm.loc[
                    (holm["classifier"] == classifier) & (holm["metric"] == metric)
                ]
                .set_index("comparison_method")
                .reindex(COMPARATOR_ORDER)
            )
            if panel.isna().any().any():
                raise ValueError(f"Incomplete Wilcoxon panel for {classifier}/{metric}")

            for position, (_, result) in enumerate(panel.iterrows()):
                estimate = float(result["mean_difference"])
                low = float(result["mean_difference_ci_low"])
                high = float(result["mean_difference_ci_high"])
                color = BLUE if estimate >= 0 else ORANGE
                significant = bool(result["reject_at_0_05"])
                ax.hlines(position, low, high, color=color, linewidth=1.2, zorder=2)
                ax.scatter(
                    estimate,
                    position,
                    s=23,
                    facecolor=color if significant else "white",
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=3,
                )

            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlim(-metric_limits[metric], metric_limits[metric])
            ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5)
            ax.set_axisbelow(True)
            if row_index == 0:
                ax.set_title(METRIC_LABELS[metric], fontweight="bold")
            if row_index == 1:
                ax.set_xlabel(r"Mean $\Delta$")
            if column_index == 0:
                ax.set_yticks(y, [COMPARATOR_LABELS[item] for item in COMPARATOR_ORDER])
                ax.set_ylabel(CLASSIFIER_LABELS[classifier], fontweight="bold")

    axes[0, 0].invert_yaxis()
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=GRAY,
                markeredgecolor=GRAY,
                label="Holm-significant",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=GRAY,
                label="Not Holm-significant",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.58, 1.01),
    )
    fig.text(
        0.56,
        0.005,
        r"Positive $\Delta$ favors KMeans-ReSC; intervals are 95% BCa confidence intervals.",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.11, top=0.91, wspace=0.32, hspace=0.18)
    save_figure(fig, "holm_wilcoxon_forest")


def main() -> None:
    configure_style()
    labels = load_dataset_labels()
    dataset_means = pd.read_csv(STATISTICS_ROOT / "dataset_method_means.csv")
    holm = pd.read_csv(STATISTICS_ROOT / "holm_wilcoxon_results.csv")
    timings = load_batched_timings()

    plot_gmean_differences(dataset_means, labels)
    plot_ir_comparison(aggregate_ir(timings), labels)
    plot_tradeoff(aggregate_tradeoff(timings), labels)
    plot_wilcoxon_forest(holm)
    print("Generated 4 paper figures in PDF and PNG formats.")


if __name__ == "__main__":
    main()
