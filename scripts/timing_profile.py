"""Profile Re-SC and KMeans-ReSC internal resampling phases.

Runs the paper's outer-fold preprocessing and sampler settings, without fitting
classifiers. Execution is intentionally sequential so each recorded wall-clock
measurement belongs to one sampler invocation.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

# Windows/MKL safeguard. Preserve a user-provided setting.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

import tests as benchmark


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
BATCH_COUNT = benchmark.BATCH_COUNT
METHODS = ("resc", "kmeans_resc_q10")
RE_SC_PHASES = (
    "validation_seconds",
    "representative_size_seconds",
    "normalization_seconds",
    "neighbor_fit_seconds",
    "neighbor_query_seconds",
    "weight_calculation_seconds",
    "weighted_sampling_seconds",
    "concatenation_seconds",
)
KMEANS_RESC_PHASES = (
    "validation_seconds",
    "representative_bounds_seconds",
    "safety_neighbor_fit_seconds",
    "safety_neighbor_query_seconds",
    "safety_filter_seconds",
    "candidate_grid_seconds",
    "candidate_kmeans_fit_seconds",
    "silhouette_score_seconds",
    "final_kmeans_fit_seconds",
    "concatenation_seconds",
)
INTERNAL_PHASES = tuple(dict.fromkeys((*RE_SC_PHASES, *KMEANS_RESC_PHASES)))
SUMMARY_FIELDS = (
    "preprocessing_seconds",
    "end_to_end_resampling_seconds",
    "end_to_end_overhead_seconds",
    "total_internal_seconds",
    "unattributed_seconds",
    *INTERNAL_PHASES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "benchmark"),
        default="benchmark",
        help="smoke: 2 folds once. benchmark: repeated stratified 10-fold CV with 10 repeats.",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset_key list for a test run.",
    )
    selection_group.add_argument(
        "--batch",
        type=int,
        choices=range(1, BATCH_COUNT + 1),
        help=f"One-based dataset batch from {BATCH_COUNT} fixed batches.",
    )
    parser.add_argument(
        "--track",
        choices=("with_c", "without_c"),
        default="with_c",
        help="Preprocessing track. Default: with_c.",
    )
    parser.add_argument("--run-name", default=None, help="Results subdirectory name.")
    args = parser.parse_args()
    args.datasets = (
        tuple(item.strip() for item in args.datasets.split(",") if item.strip())
        if args.datasets
        else None
    )
    if args.stage == "smoke":
        if args.batch is not None:
            parser.error("--batch is supported only for benchmark runs")
        args.outer_splits = 2
        args.outer_repeats = 1
    else:
        args.outer_splits = benchmark.BENCHMARK_OUTER_SPLITS
        args.outer_repeats = benchmark.BENCHMARK_OUTER_REPEATS
    if args.run_name is None:
        prefix = "timing_smoke" if args.stage == "smoke" else "timing_benchmark"
        suffix = "_10x10" if args.stage == "benchmark" else ""
        args.run_name = f"{prefix}_fixed_m15_{args.track}{suffix}"
    if args.batch is not None:
        args.run_name = f"{args.run_name}_batch{args.batch:02d}of{BATCH_COUNT}"
    if not args.run_name or any(
        not (character.isalnum() or character in "-_")
        for character in args.run_name
    ):
        parser.error("--run-name may contain only letters, digits, hyphens, and underscores")
    return args


def timing_row(
    entry: Mapping[str, Any],
    track: str,
    repeat: int,
    fold: int,
    method: str,
    train_rows: int,
    processed_features: int,
    preprocessing_seconds: float,
    sampler: Any,
    end_to_end_seconds: float,
    y_train: np.ndarray,
    y_resampled: np.ndarray,
) -> dict[str, Any]:
    phase_timings = getattr(sampler, "phase_timings_", None)
    if phase_timings is None:
        raise RuntimeError(
            "Installed imblearn-resc does not expose phase_timings_. "
            "Install imblearn-resc==0.3.1 or newer."
        )
    total_internal_seconds = float(phase_timings["total_internal_seconds"])
    row = {
        "dataset_key": entry["dataset_key"],
        "provider": entry["provider"],
        "track": track,
        "repeat": repeat,
        "fold": fold,
        "method": method,
        "M": benchmark.FIXED_M,
        "train_rows": train_rows,
        "processed_features": processed_features,
        "preprocessing_seconds": preprocessing_seconds,
        "end_to_end_resampling_seconds": end_to_end_seconds,
        "total_internal_seconds": total_internal_seconds,
        "end_to_end_overhead_seconds": max(
            0.0, end_to_end_seconds - total_internal_seconds
        ),
        "unattributed_seconds": float(phase_timings["unattributed_seconds"]),
        "representative_count": benchmark.representative_count(
            method, y_train, y_resampled
        ),
        "fallback_used": (
            bool(getattr(sampler, "fallback_used_", False))
            if method in benchmark.KMEANS_GRID_LIMITS
            else None
        ),
        "selection_fallback_used": (
            bool(getattr(sampler, "selection_fallback_used_", False))
            if method in benchmark.KMEANS_GRID_LIMITS
            else None
        ),
        "actual_k_candidates": (
            len(getattr(sampler, "candidate_ks_", ()))
            if method in benchmark.KMEANS_GRID_LIMITS
            else None
        ),
        "selected_k": (
            getattr(sampler, "selected_k_", None)
            if method in benchmark.KMEANS_GRID_LIMITS
            else None
        ),
    }
    for phase in INTERNAL_PHASES:
        row[phase] = phase_timings.get(phase, np.nan)
    return row


def profile_dataset(
    entry: Mapping[str, Any],
    schemas: Mapping[str, Any],
    track: str,
    outer_splits: int,
    outer_repeats: int,
    fold_bar: tqdm,
) -> list[dict[str, Any]]:
    bundle = benchmark.load_dataset(entry, schemas)
    rows = []
    benchmark_index = benchmark.BENCHMARK_DATASET_KEYS.index(entry["dataset_key"])
    splits = benchmark.build_outer_splits(bundle, outer_splits, outer_repeats)
    for repeat, fold, train_indices, _ in splits:
        preprocessing_started = time.perf_counter()
        preprocessor = benchmark.build_preprocessor(bundle, track)
        y_train = bundle.y[train_indices]
        X_train = np.asarray(
            preprocessor.fit_transform(bundle.X.iloc[train_indices], y_train),
            dtype=float,
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        for method in METHODS:
            method_index = benchmark.METHODS.index(method)
            seed = benchmark.derived_seed(benchmark_index, repeat, fold, method_index)
            sampler = benchmark.build_sampler(method, y_train, seed)
            started = time.perf_counter()
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            end_to_end_seconds = time.perf_counter() - started
            rows.append(
                timing_row(
                    entry=entry,
                    track=track,
                    repeat=repeat,
                    fold=fold,
                    method=method,
                    train_rows=len(train_indices),
                    processed_features=X_train.shape[1],
                    preprocessing_seconds=preprocessing_seconds,
                    sampler=sampler,
                    end_to_end_seconds=end_to_end_seconds,
                    y_train=y_train,
                    y_resampled=np.asarray(y_resampled),
                )
            )
        fold_bar.update(1)
        fold_bar.set_postfix(
            dataset=entry["dataset_key"], repeat=repeat + 1, fold=fold + 1
        )
    return rows


def write_summaries(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    folds = pd.DataFrame(rows).reset_index(drop=True)
    folds.to_csv(output_dir / "phase_timings_outer_folds.csv", index=False)

    id_columns = ["dataset_key", "provider", "track", "method"]
    phase_rows = folds.reset_index(names="_fold_row").melt(
        id_vars=["_fold_row", *id_columns],
        value_vars=SUMMARY_FIELDS,
        var_name="phase",
        value_name="seconds",
    ).dropna(subset=["seconds"])
    internal_phase_mask = phase_rows["phase"].isin(
        (*INTERNAL_PHASES, "unattributed_seconds", "total_internal_seconds")
    )
    internal_totals = phase_rows["_fold_row"].map(
        folds["total_internal_seconds"]
    )
    phase_rows["share_of_internal_percent"] = np.where(
        internal_phase_mask & (internal_totals > 0.0),
        100.0 * phase_rows["seconds"] / internal_totals,
        np.nan,
    )
    aggregation = {
        "mean_seconds": ("seconds", "mean"),
        "std_seconds": ("seconds", "std"),
        "median_seconds": ("seconds", "median"),
        "p95_seconds": ("seconds", lambda values: values.quantile(0.95)),
        "fold_count": ("seconds", "size"),
        "mean_share_of_internal_percent": ("share_of_internal_percent", "mean"),
    }
    dataset_summary = (
        phase_rows.groupby([*id_columns, "phase"], sort=False)
        .agg(**aggregation)
        .reset_index()
    )
    dataset_summary.to_csv(output_dir / "phase_timing_summary.csv", index=False)

    overall_summary = (
        phase_rows.groupby(["track", "method", "phase"], sort=False)
        .agg(**aggregation)
        .reset_index()
    )
    overall_summary.to_csv(
        output_dir / "phase_timing_overall_summary.csv", index=False
    )


def main() -> int:
    args = parse_args()
    manifest = benchmark.load_json(benchmark.MANIFEST_PATH)
    schemas = benchmark.load_json(benchmark.SCHEMA_PATH)
    entries = benchmark.select_entries(
        manifest,
        schemas,
        args.datasets,
        args.stage,
        (args.track,),
        args.batch,
    )
    if args.batch is not None:
        selected_keys = ", ".join(entry["dataset_key"] for entry in entries)
        print(f"Batch {args.batch}/{BATCH_COUNT}: {selected_keys}")
    benchmark.run_preflight(entries, schemas, args.outer_splits, (args.track,))

    total_folds = len(entries) * args.outer_splits * args.outer_repeats
    print(
        f"Profiling {len(entries)} datasets sequentially; "
        f"{total_folds} folds and {len(METHODS)} samplers per fold."
    )
    rows = []
    with tqdm(
        total=total_folds,
        desc="Timing folds complete",
        unit="fold",
        mininterval=0.5,
        smoothing=0,
    ) as fold_bar:
        for entry in entries:
            rows.extend(
                profile_dataset(
                    entry,
                    schemas,
                    args.track,
                    args.outer_splits,
                    args.outer_repeats,
                    fold_bar,
                )
            )

    output_dir = METRICS_DIR / args.run_name / args.track
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summaries(output_dir, rows)
    print(f"Completed. Timing results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
