"""Measure training-side overlap and representative geometry for Re-SC methods.

Only the paper's categorical-preserving (``with_c``) preprocessing track is
used.  Transformed kDN and N1 are evaluated on a deterministic stratified
sample matching the original training-fold size; ``Or`` uses every row.
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import tests as benchmark


TRACK = "with_c"
METHODS = ("resc", "kmeans_resc_q10")
KDN_VALUES = (3, 5, 7, 9)
OUTER_SPLITS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--batch",
        type=int,
        choices=range(1, benchmark.BATCH_COUNT + 1),
        help="One-based batch; each full batch contains three datasets.",
    )
    selection.add_argument(
        "--smoke",
        action="store_true",
        help="Run one dataset with two folds.",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def overlap_ratio(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return Re-SC's marginal-product overlap ratio and its log."""
    factors = []
    for column in range(X.shape[1]):
        values = X[:, column]
        full_range = float(np.max(values) - np.min(values))
        if full_range == 0.0:
            factors.append(1.0)
            continue
        class_zero = values[y == 0]
        class_one = values[y == 1]
        overlap = max(
            0.0,
            min(float(np.max(class_zero)), float(np.max(class_one)))
            - max(float(np.min(class_zero)), float(np.min(class_one))),
        )
        factors.append(overlap / full_range)
    if any(factor == 0.0 for factor in factors):
        return 0.0, -math.inf
    log_ratio = float(np.sum(np.log(np.asarray(factors, dtype=float))))
    return float(math.exp(log_ratio)), log_ratio


def kdn_mean(X: np.ndarray, y: np.ndarray, k: int) -> float:
    """Mean self-excluding k-disagreeing-neighbours score."""
    if len(X) <= k:
        raise ValueError(f"kDN k={k} requires more than {k} samples.")
    neighbours = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    indices = neighbours.kneighbors(X, return_distance=False)
    scores = np.empty(len(X), dtype=float)
    for row, candidates in enumerate(indices):
        candidates = candidates[candidates != row][:k]
        if len(candidates) != k:
            raise RuntimeError("Could not construct a self-excluding kDN neighborhood.")
        scores[row] = np.mean(y[candidates] != y[row])
    return float(np.mean(scores))


def n1_boundary_fraction(X: np.ndarray, y: np.ndarray) -> float:
    """N1: fraction of samples incident to cross-class Euclidean MST edges."""
    distances = pairwise_distances(X, metric="euclidean")
    positive = distances[distances > 0]
    if positive.size:
        zero_edge = float(np.min(positive)) * np.finfo(float).eps
    else:
        zero_edge = np.finfo(float).tiny
    duplicate_mask = (distances == 0.0) & ~np.eye(len(X), dtype=bool)
    distances[duplicate_mask] = zero_edge
    tree = minimum_spanning_tree(distances).tocoo()
    boundary = set()
    for left, right in zip(tree.row, tree.col):
        if y[left] != y[right]:
            boundary.add(int(left))
            boundary.add(int(right))
    return len(boundary) / len(X)


def matched_stratified_sample(
    X: np.ndarray, y: np.ndarray, sample_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a deterministic class-stratified diagnostic sample."""
    if sample_size >= len(X):
        return X, y
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    ideal = sample_size * counts / len(y)
    allocations = np.floor(ideal).astype(int)
    allocations = np.maximum(allocations, 1)
    while allocations.sum() > sample_size:
        candidates = np.flatnonzero(allocations > 1)
        allocations[candidates[np.argmax(allocations[candidates] - ideal[candidates])]] -= 1
    while allocations.sum() < sample_size:
        room = np.flatnonzero(allocations < counts)
        allocations[room[np.argmax(ideal[room] - allocations[room])]] += 1
    chosen = np.concatenate(
        [
            rng.choice(np.flatnonzero(y == label), size=count, replace=False)
            for label, count in zip(labels, allocations)
        ]
    )
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


def extract_representatives(
    X_resampled: np.ndarray, y_resampled: np.ndarray, original_features: int
) -> np.ndarray:
    """Recover unique majority representatives from concatenated rows."""
    second_halves = X_resampled[y_resampled == 0, original_features:]
    return np.unique(second_halves, axis=0)


def representative_neighbour_scores(
    representatives: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    exclude_matching_sample: bool,
    k: int = 5,
) -> np.ndarray:
    requested = k + 1 if exclude_matching_sample else k
    requested = min(requested, len(X_train))
    model = NearestNeighbors(n_neighbors=requested, metric="euclidean").fit(X_train)
    distances, indices = model.kneighbors(representatives)
    scores = np.empty(len(representatives), dtype=float)
    for row in range(len(representatives)):
        row_distances = distances[row]
        row_indices = indices[row]
        if exclude_matching_sample:
            exact = np.flatnonzero(np.isclose(row_distances, 0.0, atol=1e-12, rtol=0.0))
            if exact.size:
                keep = np.ones(len(row_indices), dtype=bool)
                keep[exact[0]] = False
                row_indices = row_indices[keep]
        row_indices = row_indices[: min(k, len(row_indices))]
        scores[row] = np.mean(y_train[row_indices] != 0)
    return scores


def representative_diagnostics(
    representatives: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
) -> dict[str, Any]:
    scores = representative_neighbour_scores(
        representatives,
        X_train,
        y_train,
        exclude_matching_sample=method == "resc",
    )
    majority = X_train[y_train == 0]
    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(representatives)
    coverage = nearest.kneighbors(majority, return_distance=True)[0][:, 0]
    return {
        "representative_count": len(representatives),
        "representative_kdn_mean": float(np.mean(scores)),
        "representative_kdn_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "representative_kdn_q25": float(np.quantile(scores, 0.25)),
        "representative_kdn_median": float(np.median(scores)),
        "representative_kdn_q75": float(np.quantile(scores, 0.75)),
        "representative_kdn_q95": float(np.quantile(scores, 0.95)),
        "empirical_safe_proportion": float(np.mean((1.0 - scores) >= 0.90)),
        "coverage_distance_mean": float(np.mean(coverage)),
        "coverage_distance_q95": float(np.quantile(coverage, 0.95)),
        "quantization_mse": (
            float(np.mean(np.square(coverage)))
            if method == "kmeans_resc_q10"
            else np.nan
        ),
    }


def geometry_row(
    dataset_key: str,
    repeat: int,
    fold: int,
    representation: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    diagnostic_size: int,
    seed: int,
) -> dict[str, Any]:
    X_diagnostic, y_diagnostic = matched_stratified_sample(
        X_full, y_full, diagnostic_size, seed
    )
    ratio, log_ratio = overlap_ratio(X_full, y_full)
    row: dict[str, Any] = {
        "dataset_key": dataset_key,
        "track": TRACK,
        "repeat": repeat,
        "fold": fold,
        "representation": representation,
        "full_rows": len(X_full),
        "diagnostic_rows": len(X_diagnostic),
        "features": X_full.shape[1],
        "or_ratio": ratio,
        "log_or_ratio": log_ratio,
        "n1": n1_boundary_fraction(X_diagnostic, y_diagnostic),
    }
    for k in KDN_VALUES:
        row[f"kdn_k{k}"] = kdn_mean(X_diagnostic, y_diagnostic, k)
    return row


def evaluate_dataset(
    entry: Mapping[str, Any],
    schemas: Mapping[str, Any],
    dataset_index: int,
    outer_splits: int,
    progress_queue: Any | None,
) -> dict[str, Any]:
    bundle = benchmark.load_dataset(entry, schemas)
    splits = benchmark.build_outer_splits(bundle, outer_splits, 1)
    geometry_rows = []
    representative_rows = []
    for repeat, fold, train_indices, _ in splits:
        preprocessor = benchmark.build_preprocessor(bundle, TRACK)
        y_train = bundle.y[train_indices]
        X_train = np.asarray(
            preprocessor.fit_transform(bundle.X.iloc[train_indices], y_train), dtype=float
        )
        common = {
            "dataset_key": entry["dataset_key"],
            "provider": entry["provider"],
            "track": TRACK,
            "repeat": repeat,
            "fold": fold,
        }
        geometry_rows.append(
            geometry_row(
                entry["dataset_key"],
                repeat,
                fold,
                "original",
                X_train,
                y_train,
                len(X_train),
                benchmark.derived_seed(dataset_index, repeat, fold, 100),
            )
        )
        for method_index, method in enumerate(METHODS):
            seed = benchmark.derived_seed(dataset_index, repeat, fold, method_index, 200)
            X_resampled, y_resampled, sampler, seconds = benchmark.resample(
                method, X_train, y_train, seed
            )
            representatives = extract_representatives(
                X_resampled, y_resampled, X_train.shape[1]
            )
            geometry_rows.append(
                geometry_row(
                    entry["dataset_key"],
                    repeat,
                    fold,
                    method,
                    X_resampled,
                    y_resampled,
                    len(X_train),
                    benchmark.derived_seed(dataset_index, repeat, fold, method_index, 300),
                )
            )
            diagnostics = representative_diagnostics(
                representatives, X_train, y_train, method
            )
            diagnostics.update(common)
            diagnostics.update(
                {
                    "method": method,
                    "resampling_seconds": seconds,
                    "fallback_used": getattr(sampler, "fallback_used_", False),
                    "selection_fallback_used": getattr(
                        sampler, "selection_fallback_used_", False
                    ),
                    "safe_majority_count": getattr(
                        sampler, "safe_majority_count_", np.nan
                    ),
                    "selected_k": getattr(sampler, "selected_k_", len(representatives)),
                }
            )
            representative_rows.append(diagnostics)
        if progress_queue is not None:
            progress_queue.put({"kind": "fold_done", "dataset_index": dataset_index})
    return {
        "dataset_index": dataset_index,
        "geometry": geometry_rows,
        "representatives": representative_rows,
    }


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def write_results(run_dir: Path, evaluations: Sequence[Mapping[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    geometry = pd.DataFrame(
        [row for evaluation in evaluations for row in evaluation["geometry"]]
    )
    representatives = pd.DataFrame(
        [row for evaluation in evaluations for row in evaluation["representatives"]]
    )
    geometry.to_csv(run_dir / "geometry_fold_metrics.csv", index=False)
    representatives.to_csv(run_dir / "representative_fold_diagnostics.csv", index=False)

    geometry_metrics = [
        "or_ratio",
        "log_or_ratio",
        "n1",
        *(f"kdn_k{k}" for k in KDN_VALUES),
    ]
    geometry_summary = flatten_columns(
        geometry.groupby(["dataset_key", "track", "representation"], sort=False)[
            geometry_metrics
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    geometry_summary.to_csv(run_dir / "geometry_dataset_summary.csv", index=False)

    representative_metrics = [
        "representative_count",
        "representative_kdn_mean",
        "representative_kdn_q95",
        "empirical_safe_proportion",
        "coverage_distance_mean",
        "coverage_distance_q95",
        "quantization_mse",
        "resampling_seconds",
    ]
    representative_summary = flatten_columns(
        representatives.groupby(["dataset_key", "track", "method"], sort=False)[
            representative_metrics
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    representative_summary.to_csv(
        run_dir / "representative_dataset_summary.csv", index=False
    )


def main() -> int:
    args = parse_args()
    manifest = benchmark.load_json(benchmark.MANIFEST_PATH)
    schemas = benchmark.load_json(benchmark.SCHEMA_PATH)
    stage = "smoke" if args.smoke else "benchmark"
    entries = benchmark.select_entries(
        manifest,
        schemas,
        None,
        stage,
        (TRACK,),
        args.batch,
    )
    outer_splits = 2 if args.smoke else OUTER_SPLITS
    benchmark.run_preflight(entries, schemas, outer_splits, (TRACK,))
    if args.batch is not None:
        print(
            f"Batch {args.batch}/{benchmark.BATCH_COUNT}: "
            + ", ".join(entry["dataset_key"] for entry in entries)
        )
    worker_count = min(args.workers, len(entries))
    print(f"Running geometry diagnostics for {len(entries)} datasets with {worker_count} worker(s).")
    evaluations = []
    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    evaluate_dataset,
                    entry,
                    schemas,
                    benchmark.BENCHMARK_DATASET_KEYS.index(entry["dataset_key"]),
                    outer_splits,
                    progress_queue,
                ): entry["dataset_key"]
                for entry in entries
            }
            pending = set(futures)
            dataset_bar = tqdm(total=len(entries), desc="Datasets complete", unit="dataset")
            fold_bar = tqdm(
                total=len(entries) * outer_splits,
                desc="Geometry folds complete",
                unit="fold",
            )
            while pending:
                while True:
                    try:
                        event = progress_queue.get_nowait()
                    except Empty:
                        break
                    if event["kind"] == "fold_done":
                        fold_bar.update(1)
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    evaluations.append(future.result())
                    dataset_bar.set_postfix_str(f"dataset={futures[future]}")
                    dataset_bar.update(1)
            while True:
                try:
                    event = progress_queue.get_nowait()
                except Empty:
                    break
                if event["kind"] == "fold_done":
                    fold_bar.update(1)
            dataset_bar.close()
            fold_bar.close()

    evaluations.sort(key=lambda result: result["dataset_index"])
    if args.smoke:
        run_name = "smoke_geometry_with_c"
    elif args.batch is None:
        run_name = "geometry_with_c"
    else:
        run_name = f"geometry_with_c_batch{args.batch:02d}of{benchmark.BATCH_COUNT}"
    run_dir = benchmark.METRICS_DIR / run_name
    write_results(run_dir, evaluations)
    print(f"Saved geometry results to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
