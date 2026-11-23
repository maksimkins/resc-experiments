# Re-SC Paper Experiments

`scripts/tests.py` runs the fixed comparison used for the paper. It is separate
from package pytest tests in `../resampling-methods`.

## Run

Run from `resc-experiments` after installing the local package:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ..\resampling-methods
```

Quick check: one dataset, two folds, all methods, and both classifiers.

```powershell
.\.venv\Scripts\python.exe scripts\tests.py --stage smoke `
  --datasets uci_iris0 `
  --track without_c `
  --run-name smoke_iris0
```

Paper run: repeated stratified 10-fold CV with 10 repeats, MLP and RBF-SVM.
Current run uses numeric features only (`without_c`).

```powershell
.\.venv\Scripts\python.exe scripts\tests.py --stage benchmark `
  --track without_c `
  --run-name benchmark_fixed_m15_without_c_10x10
```

### Ten sequential batches

`--batch` selects one fixed batch from frozen 30-dataset benchmark order. Each
batch contains three datasets. It cannot be combined with `--datasets`. Runner appends
`_batch01of10` through `_batch10of10` to the result directory automatically.

```powershell
.\.venv\Scripts\python.exe scripts\tests.py --stage benchmark --track without_c --batch 1 --workers 3 --run-name benchmark_fixed_m15_without_c_10x10
```

Run batches `2` through `10` by changing only `--batch 1`.

### Frozen benchmark

Runner uses only explicit 30-task list. No exclusion list or runtime row-count
filter exists. Thirteen retained tasks are `arrhythmia`, `car_eval_34`, `ecoli`,
`libras_move`, `oil`, `solar_flare_m0`, `spectrometer`, `us_crime`,
`yeast_me2`, `breast-w`, `heart-h`, `ionosphere`, and `thoracic-surgery`.

Seventeen added numerical binary tasks all have fewer than 1,200 rows and IR above 1.9:
`abalone9-18`, `glass6`, `haberman`, `transfusion`, `new-thyroid1`,
`page-blocks-13vs4`, `ecoli1`, `seeds`, `spect`, `vertebral`, `wpbc`, and
`yeast-1vs7`, `iris0`, `wine3`, `vowel0`, `vehicle0`, and `cleveland-0-vs-4`.

Test two datasets before the full run:

```powershell
.\.venv\Scripts\python.exe scripts\tests.py --stage benchmark `
  --datasets uci_iris0,imblearn_ecoli `
  --track without_c --workers 2 `
  --run-name check_iris0_ecoli_10x10
```

Benchmark command always uses repeated stratified 10-fold CV with 10 repeats
(100 outer evaluations per dataset). Smoke command always uses 2 folds once.
`--track` accepts `with_c`, `without_c`, or `both` (default).
`--datasets` accepts only keys from frozen benchmark and only limits tasks run.

During execution, `tqdm` displays completed datasets, folds, resamples, and
model fits. The model bar also shows each worker's latest track, repeat/fold,
method, and classifier. With parallel workers, a dataset bar can remain at zero
while its first workers are still active.

## Fixed experiment setup

- Binary target: majority class `0`, minority class `1`.
- Each outer training fold fits its own feature selection, imputation, encoding,
  scaling, resampling, and classifier. Test folds remain untouched.
- `with_c`: numeric median imputation plus `StandardScaler`; categorical
  most-frequent imputation plus one-hot encoding.
- `without_c`: schema-defined numeric columns only, then median imputation and
  `StandardScaler`. It runs for every eligible dataset whose schema retains
  numeric predictors; numeric-only datasets are unchanged by this track.
- Re-SC and KMeans-ReSC use `M=1.5`. The current reduced benchmark runs only
  `kmeans_resc_q10`, using at most 10 deterministic feasible candidate values.
- RandomUnderSampler, ClusterCentroids, SMOTE, SMOTE-ENN, and ADASYN use their
  standard balanced-class behavior.
- MLP: one hidden layer of 100 units, `alpha=1e-4`,
  `learning_rate_init=1e-3`, `max_iter=5000`.
- RBF-SVM: `C=1`, `gamma="scale"`.
- A fixed `random_state` is derived from `20260829` for every stochastic fit;
  individual seeds are not stored in outputs.

## Outputs

Each run writes only these files:

```text
results/metrics/<run-name>/
|-- with_c/
|   |-- outer_fold_metrics.csv
|   |-- resampling_timings.csv
|   `-- dataset_metrics.csv
`-- without_c/                 # eligible datasets with numeric predictors
    |-- outer_fold_metrics.csv
    |-- resampling_timings.csv
    `-- dataset_metrics.csv
```

`outer_fold_metrics.csv` contains one predictive result for every outer fold,
method, and classifier. Primary metric is G-mean; recall, specificity, F1,
ROC-AUC, confusion-matrix counts, representative count, both fallback states,
grid limit, actual candidate count, interval bounds, selected cluster count,
and fit/resampling times are included.

`resampling_timings.csv` contains one Re-SC timing and one timing for each
KMeans-ReSC grid variant per outer fold. It also records the candidate-grid
diagnostics. Time is wall-clock duration of `sampler.fit_resample`; it excludes
preprocessing and classifier fitting.

### Internal phase timing profile

`scripts/timing_profile.py` measures only Re-SC and KMeans-ReSC. It repeats
the benchmark's fold-local preprocessing and sampler configuration, but omits
classifier fitting. It runs sequentially so every `perf_counter` measurement
belongs to one sampler call. It requires `imblearn-resc==0.3.1` or newer.

```powershell
.\.venv\Scripts\python.exe scripts\timing_profile.py --stage benchmark --track with_c --batch 1 --run-name timing_fixed_m15_with_c
```

It writes three files to `results/metrics/<run-name>/<track>/`:

```text
phase_timings_outer_folds.csv       # one row per outer fold and sampler
phase_timing_summary.csv            # dataset-method-phase mean, SD, median, p95, and share
phase_timing_overall_summary.csv    # all-dataset method-phase summary
```

`dataset_metrics.csv` gives the mean and standard deviation of predictive
metrics over outer folds for each dataset, method, and classifier.

Workers process independent datasets in parallel and return results to the main
process, which writes the final CSVs. No split manifests, seed ledgers, cache
files, or run-configuration files are created.

## Data requirements

Run acquisition cell in `notebooks/01_initial_EDA.ipynb` before benchmark. It
creates 30-entry frozen manifest and corresponding raw CSVs in:

```text
data/raw/
results/metrics/dataset_manifest.json
results/metrics/feature_schema.json
```

Runner checks raw-file checksums, columns, target labels, class counts, and
outer-CV feasibility before fitting. It rejects an old or incomplete manifest.
It does not use globally preprocessed EDA datasets for scoring.

On Windows, `OMP_NUM_THREADS=1` is set before NumPy/scikit-learn import unless
you already set it. KMeans warnings are never suppressed.
