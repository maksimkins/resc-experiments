"""Run KMeans-ReSC component ablations or seed-stability diagnostics.

Both stages use only categorical-preserving (``with_c``) preprocessing and
fixed ``M=1.5``.  Ablations use repeated stratified 10-fold CV.  Stability uses
one fixed training fold per dataset, ten resampler seeds, and fixed classifier
seeds so resampler variation is measured separately from classifier variation.
"""

from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import tests as benchmark
from geometry_tests import extract_representatives, representative_diagnostics
from imblearn_resc.oversampling.KMeansReSC.utils._resc_kmeans_utils import (
    get_safe_majority_samples_knn,
    kmeans_re_sc_concatenation,
)


TRACK = "with_c"
ABLATION_VARIANTS = (
    "full_kmeans_resc",
    "no_safety_kmeans",
    "safety_random_representatives",
)
STABILITY_METHODS = ("resc", "kmeans_resc_q10")
OUTER_SPLITS = 10
OUTER_REPEATS = 10
STABILITY_SEEDS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("ablation", "stability"))
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
        help="Run one dataset with reduced folds or seeds.",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_resampled: np.ndarray,
    y_resampled: np.ndarray,
    classifier_name: str,
    classifier_seed: int,
) -> tuple[dict[str, Any], float]:
    transformer = benchmark.fit_resc_transformer("kmeans_resc_q10", X_resampled)
    X_test_transformed = benchmark.transform_for_prediction(transformer, X_test)
    classifier = benchmark.build_classifier(classifier_name, classifier_seed)
    started = time.perf_counter()
    classifier.fit(X_resampled, y_resampled)
    seconds = time.perf_counter() - started
    prediction = classifier.predict(X_test_transformed)
    score = benchmark.prediction_score(classifier, X_test_transformed)
    return benchmark.predictive_metrics(y_test, prediction, score), seconds


def concatenate(
    X_train: np.ndarray, y_train: np.ndarray, representatives: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return kmeans_re_sc_concatenation(
        X_min=X_train[y_train == 1],
        X_maj=X_train[y_train == 0],
        X_set_n=representatives,
        min_label=1,
        maj_label=0,
    )


def build_ablation_outputs(
    X_train: np.ndarray, y_train: np.ndarray, seed: int
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]:
    started = time.perf_counter()
    X_full, y_full, sampler, _ = benchmark.resample(
        "kmeans_resc_q10", X_train, y_train, seed
    )
    full_seconds = time.perf_counter() - started
    full_representatives = extract_representatives(X_full, y_full, X_train.shape[1])
    selected_k = int(sampler.selected_k_)
    if len(full_representatives) != selected_k:
        raise ValueError("Recovered full-method representatives differ from selected_k_.")

    majority = X_train[y_train == 0]
    started = time.perf_counter()
    no_safety_model = KMeans(
        n_clusters=selected_k,
        random_state=seed,
        **benchmark.KMEANS_PARAMS,
    ).fit(majority)
    no_safety_representatives = no_safety_model.cluster_centers_
    X_no_safety, y_no_safety = concatenate(
        X_train, y_train, no_safety_representatives
    )
    no_safety_seconds = time.perf_counter() - started

    started = time.perf_counter()
    safe_majority, safety_fallback = get_safe_majority_samples_knn(
        X=X_train,
        y=y_train,
        X_maj=majority,
        maj_label=0,
        n_neighbors=5,
        threshold=0.90,
        knn_params=None,
    )
    if selected_k > len(safe_majority):
        raise ValueError("Full method selected more representatives than safe candidates.")
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(len(safe_majority), size=selected_k, replace=False)
    random_representatives = safe_majority[random_indices]
    X_random, y_random = concatenate(X_train, y_train, random_representatives)
    random_seconds = time.perf_counter() - started

    common = {
        "selected_k": selected_k,
        "full_safety_fallback_used": bool(sampler.fallback_used_),
        "full_selection_fallback_used": bool(sampler.selection_fallback_used_),
        "safe_majority_count": int(sampler.safe_majority_count_),
    }
    return {
        "full_kmeans_resc": (
            X_full,
            y_full,
            full_representatives,
            {**common, "resampling_seconds": full_seconds},
        ),
        "no_safety_kmeans": (
            X_no_safety,
            y_no_safety,
            no_safety_representatives,
            {**common, "resampling_seconds": no_safety_seconds},
        ),
        "safety_random_representatives": (
            X_random,
            y_random,
            random_representatives,
            {
                **common,
                "resampling_seconds": random_seconds,
                "ablation_safety_fallback_used": bool(safety_fallback),
            },
        ),
    }


def evaluate_ablation_dataset(
    entry: Mapping[str, Any],
    schemas: Mapping[str, Any],
    dataset_index: int,
    outer_splits: int,
    outer_repeats: int,
    progress_queue: Any | None,
) -> dict[str, Any]:
    bundle = benchmark.load_dataset(entry, schemas)
    rows = []
    splits = benchmark.build_outer_splits(bundle, outer_splits, outer_repeats)
    for repeat, fold, train_indices, test_indices in splits:
        preprocessor = benchmark.build_preprocessor(bundle, TRACK)
        y_train = bundle.y[train_indices]
        y_test = bundle.y[test_indices]
        X_train = np.asarray(
            preprocessor.fit_transform(bundle.X.iloc[train_indices], y_train), dtype=float
        )
        X_test = np.asarray(preprocessor.transform(bundle.X.iloc[test_indices]), dtype=float)
        seed = benchmark.derived_seed(dataset_index, repeat, fold, 510)
        outputs = build_ablation_outputs(X_train, y_train, seed)
        for variant_index, variant in enumerate(ABLATION_VARIANTS):
            X_resampled, y_resampled, representatives, diagnostics = outputs[variant]
            for classifier_index, classifier_name in enumerate(benchmark.CLASSIFIERS):
                classifier_seed = benchmark.derived_seed(
                    dataset_index, repeat, fold, classifier_index, 520
                )
                metrics, classifier_seconds = fit_and_score(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    X_resampled,
                    y_resampled,
                    classifier_name,
                    classifier_seed,
                )
                rows.append(
                    {
                        "dataset_key": entry["dataset_key"],
                        "provider": entry["provider"],
                        "track": TRACK,
                        "repeat": repeat,
                        "fold": fold,
                        "variant": variant,
                        "classifier": classifier_name,
                        "train_rows_before": len(X_train),
                        "train_rows_after": len(X_resampled),
                        "representative_count": len(representatives),
                        "classifier_seconds": classifier_seconds,
                        **diagnostics,
                        **metrics,
                    }
                )
        if progress_queue is not None:
            progress_queue.put({"kind": "fold_done", "dataset_index": dataset_index})
    return {"dataset_index": dataset_index, "rows": rows}


def canonical_rows(X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return X
    order = np.lexsort(tuple(X[:, column] for column in reversed(range(X.shape[1]))))
    return X[order]


def symmetric_set_distances(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_to_right = NearestNeighbors(n_neighbors=1).fit(right).kneighbors(left)[0][:, 0]
    right_to_left = NearestNeighbors(n_neighbors=1).fit(left).kneighbors(right)[0][:, 0]
    chamfer = 0.5 * (float(np.mean(left_to_right)) + float(np.mean(right_to_left)))
    hausdorff = max(float(np.max(left_to_right)), float(np.max(right_to_left)))
    return chamfer, hausdorff


def exact_row_set(X: np.ndarray) -> set[bytes]:
    contiguous = np.ascontiguousarray(X)
    return {row.tobytes() for row in contiguous}


def evaluate_stability_dataset(
    entry: Mapping[str, Any],
    schemas: Mapping[str, Any],
    dataset_index: int,
    seed_count: int,
    progress_queue: Any | None,
) -> dict[str, Any]:
    bundle = benchmark.load_dataset(entry, schemas)
    _, _, train_indices, test_indices = benchmark.build_outer_splits(bundle, 10, 1)[0]
    preprocessor = benchmark.build_preprocessor(bundle, TRACK)
    y_train = bundle.y[train_indices]
    y_test = bundle.y[test_indices]
    X_train = np.asarray(
        preprocessor.fit_transform(bundle.X.iloc[train_indices], y_train), dtype=float
    )
    X_test = np.asarray(preprocessor.transform(bundle.X.iloc[test_indices]), dtype=float)

    prediction_rows = []
    representative_rows = []
    representative_sets: dict[str, list[np.ndarray]] = {
        method: [] for method in STABILITY_METHODS
    }
    first_kmeans_seed = None
    first_kmeans_representatives = None
    for seed_index in range(seed_count):
        for method_index, method in enumerate(STABILITY_METHODS):
            seed = benchmark.derived_seed(dataset_index, seed_index, method_index, 610)
            X_resampled, y_resampled, sampler, seconds = benchmark.resample(
                method, X_train, y_train, seed
            )
            representatives = extract_representatives(
                X_resampled, y_resampled, X_train.shape[1]
            )
            representative_sets[method].append(representatives)
            diagnostics = representative_diagnostics(
                representatives, X_train, y_train, method
            )
            representative_rows.append(
                {
                    "dataset_key": entry["dataset_key"],
                    "provider": entry["provider"],
                    "track": TRACK,
                    "seed_index": seed_index,
                    "method": method,
                    "resampling_seconds": seconds,
                    "fallback_used": getattr(sampler, "fallback_used_", False),
                    "selection_fallback_used": getattr(
                        sampler, "selection_fallback_used_", False
                    ),
                    **diagnostics,
                }
            )
            if method == "kmeans_resc_q10" and seed_index == 0:
                first_kmeans_seed = seed
                first_kmeans_representatives = representatives
            for classifier_index, classifier_name in enumerate(benchmark.CLASSIFIERS):
                classifier_seed = benchmark.derived_seed(
                    dataset_index, classifier_index, 620
                )
                metrics, classifier_seconds = fit_and_score(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    X_resampled,
                    y_resampled,
                    classifier_name,
                    classifier_seed,
                )
                prediction_rows.append(
                    {
                        "dataset_key": entry["dataset_key"],
                        "provider": entry["provider"],
                        "track": TRACK,
                        "seed_index": seed_index,
                        "method": method,
                        "classifier": classifier_name,
                        "classifier_seconds": classifier_seconds,
                        **metrics,
                    }
                )
        if progress_queue is not None:
            progress_queue.put({"kind": "seed_done", "dataset_index": dataset_index})

    X_repeat, y_repeat, _, _ = benchmark.resample(
        "kmeans_resc_q10", X_train, y_train, int(first_kmeans_seed)
    )
    repeated_representatives = extract_representatives(
        X_repeat, y_repeat, X_train.shape[1]
    )
    same_seed_identical = np.array_equal(
        canonical_rows(first_kmeans_representatives),
        canonical_rows(repeated_representatives),
    )

    pair_rows = []
    for method, sets in representative_sets.items():
        for left_index, right_index in itertools.combinations(range(seed_count), 2):
            left = sets[left_index]
            right = sets[right_index]
            chamfer, hausdorff = symmetric_set_distances(left, right)
            row = {
                "dataset_key": entry["dataset_key"],
                "track": TRACK,
                "method": method,
                "seed_index_left": left_index,
                "seed_index_right": right_index,
                "symmetric_chamfer": chamfer,
                "hausdorff": hausdorff,
                "exact_jaccard": np.nan,
            }
            if method == "resc":
                left_set = exact_row_set(left)
                right_set = exact_row_set(right)
                row["exact_jaccard"] = len(left_set & right_set) / len(left_set | right_set)
            pair_rows.append(row)
    determinism_row = {
        "dataset_key": entry["dataset_key"],
        "track": TRACK,
        "method": "kmeans_resc_q10",
        "seed_index": 0,
        "same_seed_identical_representatives": bool(same_seed_identical),
    }
    return {
        "dataset_index": dataset_index,
        "predictions": prediction_rows,
        "representatives": representative_rows,
        "pairs": pair_rows,
        "determinism": determinism_row,
    }


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def write_ablation_results(run_dir: Path, evaluations: Sequence[Mapping[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame([row for evaluation in evaluations for row in evaluation["rows"]])
    folds.to_csv(run_dir / "ablation_fold_metrics.csv", index=False)
    metrics = ["gmean", "recall", "specificity", "f1", "roc_auc"]
    summary = flatten_columns(
        folds.groupby(
            ["dataset_key", "track", "variant", "classifier"], sort=False
        )[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(run_dir / "ablation_dataset_summary.csv", index=False)


def write_stability_results(run_dir: Path, evaluations: Sequence[Mapping[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        [row for evaluation in evaluations for row in evaluation["predictions"]]
    )
    representatives = pd.DataFrame(
        [row for evaluation in evaluations for row in evaluation["representatives"]]
    )
    pairs = pd.DataFrame([row for evaluation in evaluations for row in evaluation["pairs"]])
    determinism = pd.DataFrame([evaluation["determinism"] for evaluation in evaluations])
    predictions.to_csv(run_dir / "stability_seed_metrics.csv", index=False)
    representatives.to_csv(run_dir / "stability_representative_metrics.csv", index=False)
    pairs.to_csv(run_dir / "stability_pairwise_representatives.csv", index=False)
    determinism.to_csv(run_dir / "same_seed_determinism.csv", index=False)

    metric_summary = flatten_columns(
        predictions.groupby(
            ["dataset_key", "track", "method", "classifier"], sort=False
        )[["gmean", "recall", "specificity", "f1", "roc_auc"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    metric_summary.to_csv(run_dir / "stability_predictive_summary.csv", index=False)
    pair_summary = flatten_columns(
        pairs.groupby(["dataset_key", "track", "method"], sort=False)[
            ["symmetric_chamfer", "hausdorff", "exact_jaccard"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    pair_summary.to_csv(run_dir / "stability_representative_summary.csv", index=False)


def main() -> int:
    args = parse_args()
    manifest = benchmark.load_json(benchmark.MANIFEST_PATH)
    schemas = benchmark.load_json(benchmark.SCHEMA_PATH)
    selection_stage = "smoke" if args.smoke else "benchmark"
    entries = benchmark.select_entries(
        manifest, schemas, None, selection_stage, (TRACK,), args.batch
    )
    outer_splits = 2 if args.smoke and args.stage == "ablation" else OUTER_SPLITS
    outer_repeats = 1 if args.smoke else OUTER_REPEATS
    seed_count = 2 if args.smoke else STABILITY_SEEDS
    benchmark.run_preflight(entries, schemas, outer_splits, (TRACK,))
    if args.batch is not None:
        print(
            f"Batch {args.batch}/{benchmark.BATCH_COUNT}: "
            + ", ".join(entry["dataset_key"] for entry in entries)
        )
    worker_count = min(args.workers, len(entries))
    unit_total = (
        len(entries) * outer_splits * outer_repeats
        if args.stage == "ablation"
        else len(entries) * seed_count
    )
    print(f"Running {args.stage} tests for {len(entries)} datasets with {worker_count} worker(s).")
    evaluations = []
    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            for entry in entries:
                dataset_index = benchmark.BENCHMARK_DATASET_KEYS.index(entry["dataset_key"])
                if args.stage == "ablation":
                    future = executor.submit(
                        evaluate_ablation_dataset,
                        entry,
                        schemas,
                        dataset_index,
                        outer_splits,
                        outer_repeats,
                        progress_queue,
                    )
                else:
                    future = executor.submit(
                        evaluate_stability_dataset,
                        entry,
                        schemas,
                        dataset_index,
                        seed_count,
                        progress_queue,
                    )
                futures[future] = entry["dataset_key"]
            pending = set(futures)
            dataset_bar = tqdm(total=len(entries), desc="Datasets complete", unit="dataset")
            unit_bar = tqdm(
                total=unit_total,
                desc=("Ablation folds complete" if args.stage == "ablation" else "Seed runs complete"),
                unit=("fold" if args.stage == "ablation" else "seed"),
            )
            while pending:
                while True:
                    try:
                        event = progress_queue.get_nowait()
                    except Empty:
                        break
                    if event["kind"] in {"fold_done", "seed_done"}:
                        unit_bar.update(1)
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
                if event["kind"] in {"fold_done", "seed_done"}:
                    unit_bar.update(1)
            dataset_bar.close()
            unit_bar.close()

    evaluations.sort(key=lambda result: result["dataset_index"])
    if args.smoke:
        run_name = f"smoke_{args.stage}_with_c"
    elif args.batch is None:
        run_name = f"{args.stage}_with_c"
    else:
        suffix = "_10x10" if args.stage == "ablation" else ""
        run_name = (
            f"{args.stage}_with_c{suffix}_batch{args.batch:02d}of{benchmark.BATCH_COUNT}"
        )
    run_dir = benchmark.METRICS_DIR / run_name
    if args.stage == "ablation":
        write_ablation_results(run_dir, evaluations)
    else:
        write_stability_results(run_dir, evaluations)
    print(f"Saved {args.stage} results to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
