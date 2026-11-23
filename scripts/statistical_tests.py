"""Run dataset-level Friedman and Holm-Wilcoxon tests for the with_c benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, friedmanchisquare, rankdata, wilcoxon


ALPHA = 0.05
METRICS = ("roc_auc", "recall", "gmean", "f1")
CLASSIFIERS = ("mlp", "rbf_svm")
METHODS = (
    "none",
    "random_under",
    "cluster_centroids",
    "smote",
    "smote_enn",
    "adasyn",
    "resc",
    "kmeans_resc_q10",
)
PROPOSED_METHOD = "kmeans_resc_q10"
BOOTSTRAP_RESAMPLES = 10_000

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = EXPERIMENT_ROOT / "results" / "metrics"
OUTPUT_DIR = METRICS_ROOT / "statistical_tests_with_c"


def load_dataset_means() -> pd.DataFrame:
    """Load with_c fold results and average the 100 folds per experimental cell."""
    paths = sorted(
        METRICS_ROOT.glob(
            "benchmark_fixed_m15_with_c_10x10_batch*/with_c/outer_fold_metrics.csv"
        )
    )
    if len(paths) != 10:
        raise ValueError(f"Expected 10 with_c result batches, found {len(paths)}.")

    fold_results = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if set(fold_results["track"].unique()) != {"with_c"}:
        raise ValueError("The input contains a preprocessing track other than with_c.")

    fold_key = ["dataset_key", "repeat", "fold", "method", "classifier"]
    if fold_results.duplicated(fold_key).any():
        raise ValueError("Duplicate fold rows found across benchmark batches.")

    expected_methods = set(METHODS)
    observed_methods = set(fold_results["method"].unique())
    if observed_methods != expected_methods:
        raise ValueError(
            f"Method mismatch: expected {sorted(expected_methods)}, "
            f"found {sorted(observed_methods)}."
        )

    expected_classifiers = set(CLASSIFIERS)
    observed_classifiers = set(fold_results["classifier"].unique())
    if observed_classifiers != expected_classifiers:
        raise ValueError(
            f"Classifier mismatch: expected {sorted(expected_classifiers)}, "
            f"found {sorted(observed_classifiers)}."
        )

    fold_counts = fold_results.groupby(
        ["dataset_key", "method", "classifier"], observed=True
    ).size()
    if not fold_counts.eq(100).all():
        invalid = fold_counts.loc[~fold_counts.eq(100)]
        raise ValueError(
            "Every dataset/method/classifier must have 100 folds. "
            f"Invalid cells: {invalid.to_dict()}"
        )

    return (
        fold_results.groupby(
            ["classifier", "dataset_key", "method"], observed=True
        )[list(METRICS)]
        .mean()
        .reset_index()
    )


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm step-down adjusted p-values."""
    order = np.argsort(p_values)
    ordered = p_values[order]
    adjusted_ordered = np.maximum.accumulate(
        (len(ordered) - np.arange(len(ordered))) * ordered
    )
    adjusted = np.empty_like(adjusted_ordered, dtype=float)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def matched_rank_biserial(differences: np.ndarray) -> float:
    """Calculate paired rank-biserial correlation; positive favors KMeans-ReSC."""
    nonzero = differences[~np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / ranks.sum())


def mean_difference_ci(differences: np.ndarray) -> tuple[float, float]:
    """Return a dataset-level 95% BCa bootstrap CI for the mean difference."""
    if np.all(np.isclose(differences, differences[0], rtol=0.0, atol=1e-12)):
        value = float(differences[0])
        return value, value
    result = bootstrap(
        (differences,),
        np.mean,
        confidence_level=0.95,
        n_resamples=BOOTSTRAP_RESAMPLES,
        method="BCa",
        rng=np.random.default_rng(42),
    )
    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def run_tests(dataset_means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one Friedman family and proposed-vs-baseline post-hoc tests per metric/classifier."""
    friedman_rows: list[dict] = []
    wilcoxon_rows: list[dict] = []

    for classifier in CLASSIFIERS:
        classifier_data = dataset_means.loc[
            dataset_means["classifier"].eq(classifier)
        ]
        for metric in METRICS:
            matrix = (
                classifier_data.pivot(
                    index="dataset_key", columns="method", values=metric
                )
                .reindex(columns=METHODS)
                .sort_index()
            )
            if matrix.isna().any().any():
                raise ValueError(f"Missing values in the {classifier}/{metric} matrix.")

            statistic, p_value = friedmanchisquare(
                *(matrix[method].to_numpy() for method in METHODS)
            )
            n_datasets, n_methods = matrix.shape
            friedman_significant = bool(p_value < ALPHA)
            friedman_rows.append(
                {
                    "classifier": classifier,
                    "metric": metric,
                    "n_datasets": n_datasets,
                    "n_methods": n_methods,
                    "friedman_statistic": statistic,
                    "p_value": p_value,
                    "kendalls_w": statistic / (n_datasets * (n_methods - 1)),
                    "significant_at_0_05": friedman_significant,
                }
            )

            proposed = matrix[PROPOSED_METHOD].to_numpy()
            family_rows: list[dict] = []
            for comparison_method in METHODS:
                if comparison_method == PROPOSED_METHOD:
                    continue
                comparison = matrix[comparison_method].to_numpy()
                differences = proposed - comparison
                if np.all(np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)):
                    test_statistic, raw_p_value = 0.0, 1.0
                else:
                    result = wilcoxon(
                        proposed,
                        comparison,
                        zero_method="pratt",
                        correction=False,
                        alternative="two-sided",
                        method="auto",
                    )
                    test_statistic = float(result.statistic)
                    raw_p_value = float(result.pvalue)

                ci_low, ci_high = mean_difference_ci(differences)

                family_rows.append(
                    {
                        "classifier": classifier,
                        "metric": metric,
                        "proposed_method": PROPOSED_METHOD,
                        "comparison_method": comparison_method,
                        "n_datasets": n_datasets,
                        "proposed_mean": proposed.mean(),
                        "comparison_mean": comparison.mean(),
                        "mean_difference": differences.mean(),
                        "mean_difference_ci_low": ci_low,
                        "mean_difference_ci_high": ci_high,
                        "wins": int((differences > 1e-12).sum()),
                        "ties": int(np.isclose(
                            differences, 0.0, rtol=0.0, atol=1e-12
                        ).sum()),
                        "losses": int((differences < -1e-12).sum()),
                        "wilcoxon_statistic": test_statistic,
                        "raw_p_value": raw_p_value,
                        "rank_biserial": matched_rank_biserial(differences),
                        "friedman_significant": friedman_significant,
                    }
                )

            adjusted = holm_adjust(
                np.asarray([row["raw_p_value"] for row in family_rows], dtype=float)
            )
            for row, adjusted_p_value in zip(family_rows, adjusted, strict=True):
                row["holm_adjusted_p_value"] = adjusted_p_value
                row["reject_at_0_05"] = bool(
                    friedman_significant and adjusted_p_value < ALPHA
                )
            wilcoxon_rows.extend(family_rows)

    return pd.DataFrame(friedman_rows), pd.DataFrame(wilcoxon_rows)


def main() -> None:
    dataset_means = load_dataset_means()
    friedman_results, wilcoxon_results = run_tests(dataset_means)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    means_path = OUTPUT_DIR / "dataset_method_means.csv"
    friedman_path = OUTPUT_DIR / "friedman_results.csv"
    wilcoxon_path = OUTPUT_DIR / "holm_wilcoxon_results.csv"
    dataset_means.to_csv(means_path, index=False)
    friedman_results.to_csv(friedman_path, index=False)
    wilcoxon_results.to_csv(wilcoxon_path, index=False)

    print(
        f"Analyzed {dataset_means['dataset_key'].nunique()} with_c datasets, "
        f"{len(METHODS)} methods, and {len(CLASSIFIERS)} classifiers."
    )
    print("\nFriedman tests:")
    print(friedman_results.to_string(index=False))
    print("\nHolm-Wilcoxon comparisons against KMeans-ReSC:")
    print(
        wilcoxon_results[
            [
                "classifier",
                "metric",
                "comparison_method",
                "mean_difference",
                "wins",
                "ties",
                "losses",
                "holm_adjusted_p_value",
                "rank_biserial",
                "reject_at_0_05",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
