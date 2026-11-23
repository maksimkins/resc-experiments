"""Run fixed Re-SC paper benchmark from ``resc-experiments``.

Raw data are processed inside each outer training fold. Results remain separate
for tracks with and without categorical features.
"""

from __future__ import annotations

import os

# Windows/MKL safeguard. Preserve a user-provided setting.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

try:
    from imblearn_resc.oversampling import KMeansReSC, ReSC
    from imblearn_resc.preprocessing import ReSCTransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Install local package first: python -m pip install -e ../resampling-methods"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
METRICS_DIR = ROOT / "results" / "metrics"
MANIFEST_PATH = METRICS_DIR / "dataset_manifest.json"
SCHEMA_PATH = METRICS_DIR / "feature_schema.json"

MASTER_SEED = 20260829
FIXED_M = 1.5
DEFAULT_WORKERS = 6
BATCH_COUNT = 10
BENCHMARK_OUTER_SPLITS = 10
BENCHMARK_OUTER_REPEATS = 10
BENCHMARK_DATASET_KEYS = (
    "imblearn_arrhythmia",
    "imblearn_car_eval_34",
    "imblearn_ecoli",
    "imblearn_libras_move",
    "imblearn_oil",
    "imblearn_solar_flare_m0",
    "imblearn_spectrometer",
    "imblearn_us_crime",
    "imblearn_yeast_me2",
    "openml_task_15_breast-w",
    "keel_glass0",
    "uci_wine3",
    "openml_task_50_heart-h",
    "openml_task_57_ionosphere",
    "keel_vehicle0",
    "keel_vowel0",
    "keel_cleveland_0vs4",
    "openml_task_9969_thoracic-surgery",
    "uci_abalone9_18",
    "uci_glass6",
    "uci_haberman",
    "uci_transfusion",
    "keel_new_thyroid1",
    "uci_page_blocks_13vs4",
    "keel_ecoli1",
    "uci_seeds",
    "uci_spect",
    "uci_vertebral",
    "uci_wpbc",
    "uci_yeast_1vs7",
)
METHODS = (
    "none",
    "resc",
    "kmeans_resc_q10",
    "random_under",
    "cluster_centroids",
    "smote",
    "smote_enn",
    "adasyn",
)
CLASSIFIERS = ("mlp", "rbf_svm")
TRACKS = ("with_c", "without_c")
KMEANS_GRID_LIMITS = {
    "kmeans_resc_q10": 10,
}
TIMED_METHODS = frozenset(("resc", *KMEANS_GRID_LIMITS))

KMEANS_PARAMS = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "tol": 1e-4,
    "algorithm": "lloyd",
    "copy_x": True,
}


@dataclass(frozen=True)
class DatasetBundle:
    entry: Mapping[str, Any]
    X: pd.DataFrame
    y: np.ndarray
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


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
        choices=("both", *TRACKS),
        default="both",
        help="Feature track: both (default), with_c, or without_c.",
    )
    parser.add_argument("--run-name", default=None, help="Results subdirectory name.")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help="Dataset workers. Default: 6."
    )
    args = parser.parse_args()
    args.datasets = (
        tuple(item.strip() for item in args.datasets.split(",") if item.strip())
        if args.datasets
        else None
    )
    args.tracks = TRACKS if args.track == "both" else (args.track,)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage == "smoke":
        if args.batch is not None:
            parser.error("--batch is supported only for benchmark runs")
        args.outer_splits = 2
        args.outer_repeats = 1
        args.workers = 1
    else:
        args.outer_splits = BENCHMARK_OUTER_SPLITS
        args.outer_repeats = BENCHMARK_OUTER_REPEATS
    if args.run_name is None:
        prefix = "smoke" if args.stage == "smoke" else "benchmark"
        suffix = "_10x10" if args.stage == "benchmark" else ""
        args.run_name = f"{prefix}_fixed_m15_{args.track}{suffix}"
    if args.batch is not None:
        args.run_name = f"{args.run_name}_batch{args.batch:02d}of{BATCH_COUNT}"
    if not args.run_name or any(
        not (character.isalnum() or character in "-_") for character in args.run_name
    ):
        parser.error("--run-name may contain only letters, digits, hyphens, and underscores")
    return args


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_seed(*coordinates: int) -> int:
    sequence = np.random.SeedSequence([MASTER_SEED, *coordinates])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def feature_roles(schema_entry: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = tuple(
        feature["name"]
        for feature in schema_entry["features"]
        if feature["semantic_type"] == "numeric"
    )
    categorical = tuple(
        feature["name"]
        for feature in schema_entry["features"]
        if feature["semantic_type"] == "categorical"
    )
    declared = numeric + categorical
    expected = tuple(schema_entry["attribute_names"])
    if set(declared) != set(expected) or len(declared) != len(expected):
        raise ValueError("Feature schema roles do not cover each predictor exactly once.")
    return numeric, categorical


def map_binary_target(series: pd.Series, target: Mapping[str, Any]) -> np.ndarray:
    minority = target["minority_label"]
    majority = target["majority_label"]
    if series.isna().any():
        raise ValueError("Target contains missing values.")
    minority_mask = series.eq(minority).to_numpy()
    majority_mask = series.eq(majority).to_numpy()
    if not np.all(minority_mask | majority_mask):
        normalized = series.astype(str).str.strip().str.casefold()
        minority_mask = normalized.eq(str(minority).strip().casefold()).to_numpy()
        majority_mask = normalized.eq(str(majority).strip().casefold()).to_numpy()
    if not np.all(minority_mask | majority_mask):
        observed = sorted(series.astype(str).unique().tolist())
        raise ValueError(f"Observed target labels do not match manifest: {observed}")
    if np.any(minority_mask & majority_mask):
        raise ValueError("Manifest minority and majority labels are not distinguishable.")
    y = minority_mask.astype(np.int8)
    expected_minority = int(str(target["counts"][str(minority)]))
    expected_majority = int(str(target["counts"][str(majority)]))
    observed = Counter(y.tolist())
    if observed[1] != expected_minority or observed[0] != expected_majority:
        raise ValueError(
            "Class counts differ from manifest: "
            f"observed majority/minority={observed[0]}/{observed[1]}, "
            f"expected={expected_majority}/{expected_minority}."
        )
    return y


def load_dataset(entry: Mapping[str, Any], schemas: Mapping[str, Any]) -> DatasetBundle:
    path = RAW_DIR / entry["raw_filename"]
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset is missing: {path}")
    actual_hash = sha256(path)
    if actual_hash != entry["sha256"]:
        raise ValueError(
            f"Checksum mismatch for {entry['dataset_key']}: "
            f"manifest={entry['sha256']}, actual={actual_hash}"
        )
    schema = schemas[entry["feature_schema_ref"]]
    numeric, categorical = feature_roles(schema)
    frame = pd.read_csv(path)
    target_name = schema["target_name"]
    expected_columns = list(schema["attribute_names"]) + [target_name]
    missing = sorted(set(expected_columns).difference(frame.columns))
    extra = sorted(set(frame.columns).difference(expected_columns))
    if missing or extra:
        raise ValueError(f"Column mismatch; missing={missing}, extra={extra}")
    if len(frame) != int(entry["n_samples"]):
        raise ValueError(f"Row count differs from manifest: {len(frame)} != {entry['n_samples']}")

    X = frame.loc[:, schema["attribute_names"]].copy()
    for column in numeric:
        original = X[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            examples = original.loc[invalid].astype(str).head(3).tolist()
            raise ValueError(f"Non-numeric values in {column}: {examples}")
        X[column] = converted
    for column in categorical:
        values = X[column]
        X[column] = values.where(values.isna(), values.astype(str)).astype(object)
    y = map_binary_target(frame[target_name], entry["target"])
    if len(np.unique(y)) != 2 or int(np.sum(y == 0)) == int(np.sum(y == 1)):
        raise ValueError("Paper requires strictly imbalanced binary targets.")
    return DatasetBundle(entry, X, y, numeric, categorical)


def track_eligible(entry: Mapping[str, Any], track: str) -> bool:
    return track == "with_c" or bool(entry.get("categorical_drop_eligible"))


def build_preprocessor(bundle: DatasetBundle, track: str) -> ColumnTransformer:
    numeric = list(bundle.numeric_columns)
    categorical = list(bundle.categorical_columns) if track == "with_c" else []
    if not numeric:
        raise ValueError(f"{bundle.entry['dataset_key']} has no numeric predictors for {track}.")
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    transformers: list[tuple[str, Any, list[str]]] = [("numeric", numeric_pipeline, numeric)]
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def validate_cv_feasibility(y: np.ndarray, outer_splits: int) -> None:
    minority = int(np.bincount(y, minlength=2).min())
    if minority < outer_splits:
        raise ValueError(f"Minority count {minority} cannot support {outer_splits} folds.")


def class_counts(y: np.ndarray) -> tuple[int, int]:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    return int(counts[0]), int(counts[1])


def build_classifier(name: str, seed: int) -> Any:
    if name == "mlp":
        return MLPClassifier(
            activation="relu",
            solver="adam",
            hidden_layer_sizes=(100,),
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=5000,
            random_state=seed,
        )
    if name == "rbf_svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale")
    raise ValueError(f"Unknown classifier: {name}")


def build_sampler(method: str, y: np.ndarray, seed: int) -> Any | None:
    majority, minority = class_counts(y)
    if majority <= minority:
        raise ValueError("Training fold must have class 0 as strict majority.")
    if method == "none":
        return None
    if method == "resc":
        return ReSC(
            M=FIXED_M,
            k=5,
            alpha=0.05,
            epsilon=0.05,
            random_state=seed,
            knn_params=None,
        )
    if method in KMEANS_GRID_LIMITS:
        return KMeansReSC(
            M=FIXED_M,
            random_state=seed,
            n_neighbors=5,
            safe_threshold=0.90,
            knn_params=None,
            kmeans_params=KMEANS_PARAMS,
            max_k_candidates=KMEANS_GRID_LIMITS[method],
        )
    if method == "random_under":
        return RandomUnderSampler(random_state=seed)
    if method == "cluster_centroids":
        return ClusterCentroids(random_state=seed)
    if method == "smote":
        return SMOTE(random_state=seed)
    if method == "smote_enn":
        return SMOTEENN(random_state=seed, n_jobs=1)
    if method == "adasyn":
        return ADASYN(random_state=seed)
    raise ValueError(f"Unknown method: {method}")


def resample(
    method: str, X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, Any | None, float]:
    sampler = build_sampler(method, y, seed)
    if sampler is None:
        return X, y, None, 0.0
    started = time.perf_counter()
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return np.asarray(X_resampled), np.asarray(y_resampled), sampler, time.perf_counter() - started


def fit_resc_transformer(method: str, X_resampled: np.ndarray) -> ReSCTransformer | None:
    if method not in TIMED_METHODS:
        return None
    transformer = ReSCTransformer()
    transformer.fit(X_resampled)
    return transformer


def transform_for_prediction(transformer: ReSCTransformer | None, X: np.ndarray) -> np.ndarray:
    return transformer.transform(X) if transformer is not None else X


def predictive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=(0, 1)).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "gmean": math.sqrt(recall * specificity),
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def prediction_score(classifier: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        classes = classifier.classes_.tolist()
        return np.asarray(classifier.predict_proba(X)[:, classes.index(1)], dtype=float)
    return np.asarray(classifier.decision_function(X), dtype=float)


def representative_count(method: str, y_before: np.ndarray, y_after: np.ndarray) -> int | None:
    if method not in TIMED_METHODS:
        return None
    majority_before, _ = class_counts(y_before)
    majority_after, _ = class_counts(y_after)
    if majority_before == 0 or majority_after % majority_before:
        raise ValueError("Concatenated majority count does not encode an integer representative count.")
    return majority_after // majority_before


def select_entries(
    manifest: Mapping[str, Any],
    schemas: Mapping[str, Any],
    dataset_keys: Sequence[str] | None,
    stage: str,
    tracks: Sequence[str],
    batch: int | None,
) -> list[Mapping[str, Any]]:
    by_key = {entry["dataset_key"]: entry for entry in manifest["datasets"]}
    missing = [key for key in BENCHMARK_DATASET_KEYS if key not in by_key]
    if missing:
        raise ValueError(
            "Frozen benchmark manifest is incomplete. Rerun "
            "notebooks/01_initial_EDA.ipynb acquisition cell. Missing: "
            f"{missing}"
        )
    benchmark_entries = [by_key[key] for key in BENCHMARK_DATASET_KEYS]
    compatible = [
        entry
        for entry in benchmark_entries
        if any(track_eligible(entry, track) for track in tracks)
    ]
    incompatible = [
        entry["dataset_key"]
        for entry in benchmark_entries
        if entry not in compatible
    ]
    if incompatible:
        raise ValueError(
            "Frozen benchmark datasets incompatible with requested track: "
            f"{incompatible}"
        )
    by_key = {entry["dataset_key"]: entry for entry in compatible}
    if dataset_keys:
        unknown = sorted(set(dataset_keys).difference(by_key))
        if unknown:
            raise ValueError(
                "Datasets absent from frozen benchmark or incompatible with requested track: "
                f"{unknown}"
            )
        return [by_key[key] for key in dataset_keys]
    if stage == "smoke":
        mixed = []
        for entry in benchmark_entries:
            if entry["provider"] != "OpenML":
                continue
            numeric, categorical = feature_roles(schemas[entry["feature_schema_ref"]])
            if numeric and categorical:
                mixed.append(entry)
        if not mixed:
            raise ValueError("Smoke run needs a mixed-feature OpenML benchmark dataset.")
        return [min(mixed, key=lambda item: int(item["n_samples"]))]
    if batch is not None:
        selected = select_batch(compatible, batch)
        if not selected:
            raise ValueError(f"Batch {batch}/{BATCH_COUNT} has no benchmark datasets.")
        return selected
    return compatible


def select_batch(
    entries: Sequence[Mapping[str, Any]], batch: int | None
) -> list[Mapping[str, Any]]:
    if batch is None:
        return list(entries)
    start = (batch - 1) * len(entries) // BATCH_COUNT
    stop = batch * len(entries) // BATCH_COUNT
    selected = list(entries[start:stop])
    if not selected:
        raise ValueError(f"Batch {batch}/{BATCH_COUNT} has no benchmark datasets.")
    return selected


def build_outer_splits(
    bundle: DatasetBundle, outer_splits: int, outer_repeats: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=MASTER_SEED,
    )
    result = []
    for split_index, (train_indices, test_indices) in enumerate(splitter.split(bundle.X, bundle.y)):
        result.append(
            (split_index // outer_splits, split_index % outer_splits, train_indices, test_indices)
        )
    return result


def evaluate_dataset(
    entry: Mapping[str, Any],
    schemas: Mapping[str, Any],
    dataset_index: int,
    outer_splits: int,
    outer_repeats: int,
    tracks: Sequence[str],
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    bundle = load_dataset(entry, schemas)
    splits = build_outer_splits(bundle, outer_splits, outer_repeats)
    result = {track: {"folds": [], "timings": []} for track in tracks}
    for track in tracks:
        if not track_eligible(entry, track):
            continue
        for repeat, fold, train_indices, test_indices in splits:
            if progress_queue is not None:
                progress_queue.put(
                    {
                        "kind": "fold_start",
                        "dataset_index": dataset_index,
                        "track": track,
                        "repeat": repeat,
                        "fold": fold,
                    }
                )
            preprocessing_started = time.perf_counter()
            preprocessor = build_preprocessor(bundle, track)
            y_train = bundle.y[train_indices]
            y_test = bundle.y[test_indices]
            X_train = np.asarray(
                preprocessor.fit_transform(bundle.X.iloc[train_indices], y_train), dtype=float
            )
            X_test = np.asarray(preprocessor.transform(bundle.X.iloc[test_indices]), dtype=float)
            preprocessing_seconds = time.perf_counter() - preprocessing_started
            train_majority, train_minority = class_counts(y_train)

            for method_index, method in enumerate(METHODS):
                sampler_seed = derived_seed(dataset_index, repeat, fold, method_index)
                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "kind": "resample_start",
                            "dataset_index": dataset_index,
                            "track": track,
                            "repeat": repeat,
                            "fold": fold,
                            "method": method,
                        }
                    )
                X_resampled, y_resampled, sampler, resample_seconds = resample(
                    method, X_train, y_train, sampler_seed
                )
                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "kind": "resample_done",
                            "dataset_index": dataset_index,
                            "track": track,
                            "repeat": repeat,
                            "fold": fold,
                            "method": method,
                            "seconds": resample_seconds,
                        }
                    )
                transformer = fit_resc_transformer(method, X_resampled)
                representatives = representative_count(method, y_train, y_resampled)
                resampled_majority, resampled_minority = class_counts(y_resampled)
                train_ir = train_majority / train_minority
                resampled_ir = resampled_majority / resampled_minority
                representative_ratio = (
                    representatives / train_majority if representatives is not None else None
                )
                safety_fallback = (
                    bool(getattr(sampler, "fallback_used_", False))
                    if method in KMEANS_GRID_LIMITS
                    else None
                )
                selection_fallback = (
                    bool(getattr(sampler, "selection_fallback_used_", False))
                    if method in KMEANS_GRID_LIMITS
                    else None
                )
                common = {
                    "dataset_key": entry["dataset_key"],
                    "provider": entry["provider"],
                    "track": track,
                    "repeat": repeat,
                    "fold": fold,
                    "method": method,
                    "preprocessing_seconds": preprocessing_seconds,
                    "resample_seconds": resample_seconds,
                    "train_rows": len(train_indices),
                    "test_rows": len(test_indices),
                    "input_features": bundle.X.shape[1],
                    "processed_features": X_train.shape[1],
                    "resampled_rows": X_resampled.shape[0],
                    "resampled_features": X_resampled.shape[1],
                    "train_majority": train_majority,
                    "train_minority": train_minority,
                    "train_ir": train_ir,
                    "resampled_majority": resampled_majority,
                    "resampled_minority": resampled_minority,
                    "resampled_ir": resampled_ir,
                    "representative_count": representatives,
                    "representative_ratio": representative_ratio,
                    "fallback_used": safety_fallback,
                    "selection_fallback_used": selection_fallback,
                    "max_k_candidates": KMEANS_GRID_LIMITS.get(method),
                    "actual_k_candidates": (
                        len(getattr(sampler, "candidate_ks_", ()))
                        if method in KMEANS_GRID_LIMITS
                        else None
                    ),
                    "n1": (
                        getattr(sampler, "n1_", None)
                        if method in KMEANS_GRID_LIMITS
                        else None
                    ),
                    "k_min": (
                        getattr(sampler, "k_min_", None)
                        if method in KMEANS_GRID_LIMITS
                        else None
                    ),
                    "k_max": (
                        getattr(sampler, "k_max_", None)
                        if method in KMEANS_GRID_LIMITS
                        else None
                    ),
                    "selected_k": (
                        getattr(sampler, "selected_k_", None)
                        if method in KMEANS_GRID_LIMITS
                        else None
                    ),
                }
                if method in TIMED_METHODS:
                    result[track]["timings"].append({"M": FIXED_M, **common})

                X_test_model = transform_for_prediction(transformer, X_test)
                for classifier_index, classifier_name in enumerate(CLASSIFIERS):
                    if progress_queue is not None:
                        progress_queue.put(
                            {
                                "kind": "model_start",
                                "dataset_index": dataset_index,
                                "track": track,
                                "repeat": repeat,
                                "fold": fold,
                                "method": method,
                                "classifier": classifier_name,
                            }
                        )
                    classifier = build_classifier(
                        classifier_name,
                        derived_seed(dataset_index, repeat, fold, method_index, classifier_index, 1),
                    )
                    fit_started = time.perf_counter()
                    classifier.fit(X_resampled, y_resampled)
                    classifier_fit_seconds = time.perf_counter() - fit_started
                    y_pred = np.asarray(classifier.predict(X_test_model), dtype=np.int8)
                    y_score = prediction_score(classifier, X_test_model)
                    result[track]["folds"].append(
                        {
                            **common,
                            "classifier": classifier_name,
                            "classifier_fit_seconds": classifier_fit_seconds,
                            **predictive_metrics(y_test, y_pred, y_score),
                        }
                    )
                    if progress_queue is not None:
                        progress_queue.put(
                            {
                                "kind": "model_done",
                                "dataset_index": dataset_index,
                                "track": track,
                                "repeat": repeat,
                                "fold": fold,
                                "method": method,
                                "classifier": classifier_name,
                            }
                        )
            if progress_queue is not None:
                progress_queue.put(
                    {
                        "kind": "fold_done",
                        "dataset_index": dataset_index,
                        "track": track,
                        "repeat": repeat,
                        "fold": fold,
                    }
                )
    return {"dataset_index": dataset_index, "results": result}


def run_preflight(
    entries: Sequence[Mapping[str, Any]],
    schemas: Mapping[str, Any],
    outer_splits: int,
    tracks: Sequence[str],
) -> None:
    for entry in entries:
        bundle = load_dataset(entry, schemas)
        validate_cv_feasibility(bundle.y, outer_splits)
        for track in tracks:
            if track_eligible(entry, track):
                build_preprocessor(bundle, track)
    print(f"Preflight passed: {len(entries)} datasets.")


def write_results(
    run_dir: Path, evaluations: Sequence[Mapping[str, Any]], tracks: Sequence[str]
) -> None:
    numeric_metrics = ["gmean", "recall", "specificity", "f1", "roc_auc"]
    for track in tracks:
        output_dir = run_dir / track
        output_dir.mkdir(parents=True, exist_ok=True)
        fold_rows = [
            row for evaluation in evaluations for row in evaluation["results"][track]["folds"]
        ]
        timing_rows = [
            row for evaluation in evaluations for row in evaluation["results"][track]["timings"]
        ]
        folds = pd.DataFrame(fold_rows)
        pd.DataFrame(timing_rows).to_csv(output_dir / "resampling_timings.csv", index=False)
        folds.to_csv(output_dir / "outer_fold_metrics.csv", index=False)
        if not folds.empty:
            summary = folds.groupby(
                ["dataset_key", "provider", "track", "method", "classifier"], sort=False
            )[numeric_metrics].agg(["mean", "std"]).reset_index()
            summary.columns = [
                "_".join(part for part in column if part)
                if isinstance(column, tuple)
                else column
                for column in summary.columns
            ]
            summary.to_csv(output_dir / "dataset_metrics.csv", index=False)


def main() -> int:
    args = parse_args()
    manifest = load_json(MANIFEST_PATH)
    schemas = load_json(SCHEMA_PATH)
    entries = select_entries(
        manifest, schemas, args.datasets, args.stage, args.tracks, args.batch
    )
    if args.batch is not None:
        selected_keys = ", ".join(entry["dataset_key"] for entry in entries)
        print(f"Batch {args.batch}/{BATCH_COUNT}: {selected_keys}")
    run_preflight(entries, schemas, args.outer_splits, args.tracks)

    worker_count = min(args.workers, len(entries))
    evaluations = []
    total_folds = sum(
        int(track_eligible(entry, track))
        for entry in entries
        for track in args.tracks
    ) * args.outer_splits * args.outer_repeats
    total_resamples = total_folds * len(METHODS)
    total_model_fits = total_resamples * len(CLASSIFIERS)
    print(f"Running {len(entries)} datasets with {worker_count} worker(s).")
    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    evaluate_dataset,
                    entry,
                    schemas,
                    dataset_index,
                    args.outer_splits,
                    args.outer_repeats,
                    args.tracks,
                    progress_queue,
                ): dataset_index
                for dataset_index, entry in enumerate(entries)
            }
            pending = set(futures)
            dataset_bar = tqdm(
                total=len(futures),
                desc="Datasets complete",
                unit="dataset",
                position=0,
                mininterval=0.5,
                smoothing=0,
            )
            fold_bar = tqdm(
                total=total_folds,
                desc="Folds complete",
                unit="fold",
                position=1,
                mininterval=0.5,
                smoothing=0,
            )
            resample_bar = tqdm(
                total=total_resamples,
                desc="Resamples complete",
                unit="resample",
                position=2,
                mininterval=0.5,
                smoothing=0,
            )
            model_bar = tqdm(
                total=total_model_fits,
                desc="Model fits complete",
                unit="fit",
                position=3,
                mininterval=0.5,
                smoothing=0,
            )
            active: dict[int, str] = {}
            while pending:
                while True:
                    try:
                        event = progress_queue.get_nowait()
                    except Empty:
                        break
                    context = f"{event['track']} r{event['repeat'] + 1}/f{event['fold'] + 1}"
                    if event["kind"] == "fold_start":
                        active[event["dataset_index"]] = context
                    elif event["kind"] == "resample_start":
                        active[event["dataset_index"]] = f"{context} {event['method']} resampling"
                    elif event["kind"] == "resample_done":
                        resample_bar.update(1)
                        active[event["dataset_index"]] = (
                            f"{context} {event['method']} resampled "
                            f"({event['seconds']:.1f}s)"
                        )
                    elif event["kind"] == "model_start":
                        active[event["dataset_index"]] = (
                            f"{context} {event['method']}/{event['classifier']} fitting"
                        )
                    elif event["kind"] == "model_done":
                        model_bar.update(1)
                        active[event["dataset_index"]] = (
                            f"{context} {event['method']}/{event['classifier']} done"
                        )
                    elif event["kind"] == "fold_done":
                        fold_bar.update(1)
                        active[event["dataset_index"]] = f"{context} done"
                active_text = " | ".join(
                    f"{index}:{value}" for index, value in sorted(active.items())
                )
                if active_text:
                    model_bar.set_postfix_str(active_text[:180])
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    dataset_index = futures[future]
                    evaluations.append(future.result())
                    active.pop(dataset_index, None)
                    dataset_bar.update(1)
                    dataset_bar.set_postfix(dataset=entries[dataset_index]["dataset_key"])
            while True:
                try:
                    event = progress_queue.get_nowait()
                except Empty:
                    break
                if event["kind"] == "resample_done":
                    resample_bar.update(1)
                elif event["kind"] == "model_done":
                    model_bar.update(1)
                elif event["kind"] == "fold_done":
                    fold_bar.update(1)
            model_bar.close()
            resample_bar.close()
            fold_bar.close()
            dataset_bar.close()
    evaluations.sort(key=lambda item: int(item["dataset_index"]))
    run_dir = METRICS_DIR / args.run_name
    write_results(run_dir, evaluations, args.tracks)
    print(f"Completed. Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
