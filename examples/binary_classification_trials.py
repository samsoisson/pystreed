"""Evaluate accuracy- or F1-optimized trees on the 14-dataset runtime-reduced benchmark.

Run from an installed source checkout:
    python examples/binary_classification_trials.py
    python examples/binary_classification_trials.py --datasets anneal --max-depth 2

Input: whitespace-delimited, no header, target in the first column. The larger
sorted target label is positive. All feature types are allowed. Preprocessing is
fitted independently inside every CV fold. Times are wall-clock seconds; runtime
includes tuning, validation predictions, and final refit, excluding test scoring.
"""

import argparse
import faulthandler
import json
import platform
import os
import pickle
import subprocess
import sys
import tempfile
import zipfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from time import sleep

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


DEFAULT_DEPTHS = {"accuracy": (4, 5), "f1": (3, 4), "mcc": (3, 4)}
SEEDS = (11, 12, 13, 14, 15)
METRICS = ("accuracy", "precision", "recall", "f1", "mcc", "training_runtime_s")
# Fixed runtime-reduced benchmark: exclude the nine costly datasets discussed.
# Some depth-4 F1 runtimes are unknown; this is not a measured speed ranking.
DATASETS = (
    "anneal", "audiology", "heart-cleveland", "hepatitis", "hypothyroid",
    "kr-vs-kp", "lymph", "mushroom", "primary-tumor", "segment", "soybean",
    "tic-tac-toe", "vote", "yeast",
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/classification")
    parser.add_argument("--output-dir", type=Path, default=root / "results/binary_classification")
    parser.add_argument("--objective", choices=("accuracy", "f1", "mcc"), default="accuracy",
                        help="Training objective (default: accuracy); mcc is not implemented yet")
    parser.add_argument("--max-num-nodes", type=int, nargs="+",
                        help="F1 branching-node caps to tune (default: 0 through 2**max_depth - 1)")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        help="Optional subset of the 14 runtime-reduced benchmark datasets")
    parser.add_argument("--resume", action="store_true", help="Preserve recorded trials and resume unfinished trials")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS),
                        help="Optional subset of the five trial seeds")
    parser.add_argument("--retry-failed", action="store_true",
                        help="With --resume, retry recorded failures within the selected datasets/seeds")
    parser.add_argument("--isolate-fits", action=argparse.BooleanOptionalAction, default=None,
                        help="Run each fit in a child process (default: enabled for F1)")
    parser.add_argument("--max-depth", type=int,
                        help="Single-depth override (default: accuracy=4 then 5; f1/mcc=3 then 4)")
    parser.add_argument("--time-limit", type=float, default=600,
                        help="Solver time limit per fit, in seconds")
    parser.add_argument("--cost-complexities", type=float, nargs="+",
                        help="Accuracy penalties (default: 0 0.0001 0.0005 0.001 0.005 0.01)")
    args = parser.parse_args()
    args.depths = [args.max_depth] if args.max_depth is not None else list(DEFAULT_DEPTHS[args.objective])
    if args.objective == "mcc":
        parser.exit(message="not implemented yet\n")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    args.seeds = sorted(set(args.seeds))
    if args.isolate_fits is None:
        args.isolate_fits = args.objective == "f1"
    args.max_depth = args.depths[0]
    if not 0 <= args.max_depth <= 20 or not np.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("depth must be in [0, 20] and time limit must be finite and positive")
    if args.objective == "f1" and args.cost_complexities is not None:
        parser.error("F1 does not use --cost-complexities; use --max-num-nodes")
    if args.cost_complexities is None:
        args.cost_complexities = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    if any(not np.isfinite(x) or not 0 <= x <= 1 for x in args.cost_complexities):
        parser.error("regularization values must be finite and in [0, 1]")
    args.cost_complexities = sorted(set(args.cost_complexities))
    if args.objective == "accuracy" and args.max_num_nodes is not None:
        parser.error("--max-num-nodes is only used with --objective f1")
    if args.objective == "f1":
        largest_tree = 2 ** args.max_depth - 1
        if args.max_num_nodes is not None:
            if any(value < 0 or value > largest_tree for value in args.max_num_nodes):
                parser.error("node caps must fit every requested depth (0 through 2**min(depths) - 1)")
            args.max_num_nodes = sorted(set(args.max_num_nodes))
    return args


def tuning_parameter(args):
    """Return the effective structural regularization parameter and its grid."""
    if args.objective == "f1":
        return "max_num_nodes", (
            args.max_num_nodes if args.max_num_nodes is not None else list(range(2 ** args.max_depth))
        )
    return "cost_complexity", args.cost_complexities


def result_path(output, objective, kind, extension="csv"):
    return output / f"{kind}_{objective}.{extension}"


def solver_cache_options(args):
    # Earlier diagnostics favored disabling caches, but native crashes can also
    # occur without them. Process isolation below contains those failures.
    return {"use_dataset_caching": getattr(args, "_dataset_cache_override",
                                          not (args.objective == "f1" and args.max_depth >= 4)),
            "use_branch_caching": False}


def make_model(args, seed, value):
    from pystreed import STreeDClassifier

    parameter, _ = tuning_parameter(args)
    return STreeDClassifier(
        optimization_task="f1-score" if args.objective == "f1" else "cost-complex-accuracy",
        **{parameter: value},
        **solver_cache_options(args),
        max_depth=args.max_depth, time_limit=args.time_limit, random_seed=seed,
        hyper_tune=False, continuous_binarize_strategy="quantile",
    )


def fit_checked(model, X, y):
    model.fit(X, y)
    result = getattr(model, "fit_result", None)
    if result is None or not result.is_feasible():
        raise RuntimeError("Solver did not return a feasible tree")
    return bool(result.is_optimal())


def fit_and_predict(args, seed, value, X, y, predict_X, context):
    """Return only plain data; native solver objects stay inside their process."""
    args._last_fit_execution = {"fit_attempts": 1, **solver_cache_options(args)}
    if not getattr(args, "isolate_fits", False):
        start = perf_counter()
        model = make_model(args, seed, value)
        optimal = fit_checked(model, X, y)
        elapsed = perf_counter() - start
        prediction_start = perf_counter()
        predictions = model.predict(predict_X)
        return predictions, optimal, elapsed, perf_counter() - prediction_start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / f"fit_logs_{args.objective}"
    logs.mkdir(exist_ok=True)
    log = logs / f"{context}.log"
    with tempfile.TemporaryDirectory(prefix="streed-fit-") as temporary:
        request = Path(temporary) / "request.pkl"
        response = Path(temporary) / "response.pkl"
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                   "--_fit-worker", str(request), str(response)]
        for attempt in (1, 2):
            options = {key: val for key, val in vars(args).items() if not key.startswith("_last_")}
            if attempt == 2:
                options["_dataset_cache_override"] = True
                log = logs / f"{context}_dataset_cache_retry.log"
            with request.open("wb") as stream:
                pickle.dump((options, seed, value, X, y, predict_X), stream)
            with log.open("w", encoding="utf-8") as stream:
                worker = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if worker.returncode == 0 and response.exists():
                args._last_fit_execution = {
                    "fit_attempts": attempt,
                    "use_dataset_caching": True if attempt == 2 else solver_cache_options(args)["use_dataset_caching"],
                    "use_branch_caching": False,
                }
                with response.open("rb") as stream:
                    return pickle.load(stream)
            # Only the reproduced F1 access-violation failure qualifies. Never
            # change the objective, seed, node cap, depth, data, or time limit.
            if (attempt == 1 and args.objective == "f1"
                    and not solver_cache_options(args)["use_dataset_caching"]
                    and worker.returncode & 0xffffffff == 0xc0000005):
                print("    Access violation: retrying this fit once with dataset caching enabled", flush=True)
                continue
            code = f"0x{worker.returncode & 0xffffffff:08X}"
            raise RuntimeError(f"Native fit worker failed (exit {code}); see {log}")


def fit_worker(request, response):
    # Prevent Windows error dialogs from blocking worker exit after a native fault.
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    faulthandler.enable()
    with Path(request).open("rb") as stream:
        options, seed, value, X, y, predict_X = pickle.load(stream)
    args = argparse.Namespace(**options)
    args.isolate_fits = False
    result = fit_and_predict(args, seed, value, X, y, predict_X, "worker")
    with Path(response).open("wb") as stream:
        pickle.dump(result, stream)


def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    # OneDrive/Windows may briefly lock the destination during synchronization.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            sleep(0.2 * 2 ** attempt)


def read_records(path):
    if not path.exists() or path.stat().st_size <= 1:
        return []
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The 'float_precision' argument is deprecated.*")
            return pd.read_csv(path, float_precision="round_trip").to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def load_checkpoint(args, metadata):
    files = {kind: result_path(args.output_dir, args.objective, kind)
             for kind in ("trials", "cv_results", "datasets")}
    config = result_path(args.output_dir, args.objective, "config", "json")
    existing = config.exists() or any(path.exists() for path in files.values())
    if existing and not args.resume:
        raise ValueError(f"Existing {args.objective} outputs in {args.output_dir}; use --resume or another --output-dir")
    if not existing:
        config.write_text(json.dumps(metadata, default=str, indent=2), encoding="utf-8")
        return [], [], []
    if not config.exists():
        raise ValueError("Cannot resume without the original configuration JSON")
    original = json.loads(config.read_text(encoding="utf-8"))
    for key in ("objective", "max_depth", "time_limit", "tuning_parameter", "tuning_grid", "seeds",
                "test_size", "cv_folds", "cv_shuffle", "use_dataset_caching", "use_branch_caching"):
        expected = list(metadata[key]) if isinstance(metadata[key], tuple) else metadata[key]
        if original.get(key, expected) != expected:
            raise ValueError(f"Cannot resume with different {key}")
    trials, cv_rows, datasets = (read_records(files[k]) for k in ("trials", "cv_results", "datasets"))
    for kind, records, keys in (("trials", trials, ["dataset", "seed"]),
                                ("datasets", datasets, ["dataset"])):
        if records and pd.DataFrame(records).duplicated(keys).any():
            raise ValueError(f"Duplicate records in {kind}; refusing to resume")
    for records in (trials, cv_rows, datasets):
        if any(row.get("objective", args.objective) != args.objective or
               row.get("max_depth", args.max_depth) != args.max_depth for row in records):
            raise ValueError("Checkpoint contains a different objective or depth")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    # A ZIP backup preserves original bytes without confusing summary discovery.
    with zipfile.ZipFile(args.output_dir / f"resume_backup_{args.objective}_{stamp}.zip", "x") as backup:
        for path in [*files.values(), result_path(args.output_dir, args.objective, "summary"), config]:
            if path.exists():
                backup.write(path, arcname=path.name)
    with (args.output_dir / f"resume_events_{args.objective}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": stamp, "isolate_fits": args.isolate_fits,
                                 "preserved_trials": len(trials),
                                 "selected_datasets": metadata.get("selected_datasets", []),
                                 "selected_seeds": metadata.get("selected_seeds", list(SEEDS)),
                                 "retry_failed": getattr(args, "retry_failed", False)}) + "\n")
    print(f"Resuming with {len(trials)} recorded trials preserved", flush=True)
    return trials, cv_rows, datasets


def run_trial(args, dataset, X, y, seed, cv_rows, checkpoint=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )
    if np.unique(y_test).size != 2 or np.bincount(y_train, minlength=2).min() < 5:
        raise ValueError("Need both test classes and at least five training samples per class")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train))
    start = perf_counter()
    candidates = []
    # Explicit CV records solver status and fits preprocessing on training folds only.
    parameter, values = tuning_parameter(args)
    total_fits = len(values) * len(folds)
    print(f"  Tuning: {total_fits} CV fits, then one final refit", flush=True)
    fit_number = 0
    saved_fold_runtime = sum(item.get("fit_runtime_s", 0) for item in cv_rows
                             if item["dataset"] == dataset and item["seed"] == seed
                             and item["status"] == "ok" and item.get(parameter) in values)
    safe_dataset = dataset.replace("/", "_").replace("\\", "_")
    for value in values:
        scores = []
        for fold, (train, valid) in enumerate(folds, start=1):
            row = {"dataset": dataset, "seed": seed, "objective": args.objective,
                   "max_depth": args.max_depth, "tuning_parameter": parameter, parameter: value, "fold": fold}
            fit_number += 1
            previous = next((item for item in cv_rows if item["dataset"] == dataset and item["seed"] == seed
                             and item.get(parameter) == value and item["fold"] == fold
                             and item["status"] == "ok"), None)
            if previous is not None:
                scores.append(previous["validation_mcc"])
                print(f"  [CV {fit_number}/{total_fits}] using saved fold result", flush=True)
                continue
            # Failed folds of an unfinished trial are retried on resume; the ZIP
            # backup retains the original records.
            cv_rows[:] = [item for item in cv_rows if not (item["dataset"] == dataset and item["seed"] == seed
                          and item.get(parameter) == value and item["fold"] == fold)]
            print(f"  [CV {fit_number}/{total_fits}] {parameter}={value}, fold {fold}/5: fitting", flush=True)
            fold_start = perf_counter()
            try:
                predictions, row["is_optimal"], solver_time, _ = fit_and_predict(
                    args, seed, value, X_train.iloc[train], y_train[train], X_train.iloc[valid],
                    f"{safe_dataset}_seed{seed}_{parameter}{value}_fold{fold}")
                row["fit_runtime_s"] = perf_counter() - fold_start
                row["solver_fit_runtime_s"] = solver_time
                row.update(getattr(args, "_last_fit_execution", {}))
                score = float(matthews_corrcoef(y_train[valid], predictions))
                row.update(validation_mcc=score, status="ok")
                scores.append(score)
                print(
                    f"    Finished in {perf_counter() - fold_start:.2f}s; "
                    f"validation MCC={score:.4f}; optimal={row['is_optimal']}",
                    flush=True,
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"    Failed after {perf_counter() - fold_start:.2f}s: {row['error']}", flush=True)
            cv_rows.append(row)
            if checkpoint is not None:
                checkpoint()
        # Do not compare candidates using an incomplete collection of folds.
        if len(scores) != 5:
            raise RuntimeError(
                f"Incomplete CV for {parameter}={value}; see cv_results_{args.objective}.csv"
            )
        candidates.append((float(np.mean(scores)), value))
    # Stronger regularization: larger accuracy penalty or smaller F1 node cap.
    direction = -1 if args.objective == "f1" else 1
    validation_mcc, value = max(candidates, key=lambda candidate: (candidate[0], direction * candidate[1]))
    print(f"  [Refit] selected {parameter}={value}; fitting all training data", flush=True)
    refit_start = perf_counter()
    pred, optimal, solver_refit_time, prediction_time = fit_and_predict(
        args, seed, value, X_train, y_train, X_test,
        f"{safe_dataset}_seed{seed}_{parameter}{value}_refit")
    refit_runtime = perf_counter() - refit_start - prediction_time
    runtime = perf_counter() - start + saved_fold_runtime - prediction_time
    print(f"    Refit finished in {refit_runtime:.2f}s; optimal={optimal}", flush=True)
    return {
        "dataset": dataset, "seed": seed, "status": "ok", "objective": args.objective,
        "max_depth": args.max_depth, "n_train": len(y_train), "n_test": len(y_test),
        "tuning_parameter": parameter, f"best_{parameter}": value, "validation_mcc": validation_mcc,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, pred),
        "training_runtime_s": runtime, "refit_runtime_s": refit_runtime,
        "solver_refit_runtime_s": solver_refit_time,
        "isolated_fits": getattr(args, "isolate_fits", False),
        "refit_execution": json.dumps(getattr(args, "_last_fit_execution", {})),
        "is_optimal": optimal,
    }


def summarize(trials):
    rows = []
    if trials.empty:
        return pd.DataFrame()
    for dataset, group in trials.groupby("dataset", sort=True):
        ok = group[group.status == "ok"]
        complete = len(ok) == 5 and set(ok.seed) == set(SEEDS)
        row = {"dataset": dataset, "successful_trials": len(ok), "complete": complete}
        # Never silently present fewer than five trials as a five-seed estimate.
        for metric in METRICS:
            row[f"{metric}_mean"] = ok[metric].mean() if complete else np.nan
            row[f"{metric}_std"] = ok[metric].std(ddof=1) if complete else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(output, trials, cv_rows, datasets, objective):
    frame = pd.DataFrame(trials)
    atomic_csv(frame, result_path(output, objective, "trials"))
    atomic_csv(pd.DataFrame(cv_rows), result_path(output, objective, "cv_results"))
    atomic_csv(pd.DataFrame(datasets), result_path(output, objective, "datasets"))
    summary = summarize(frame)
    if not frame.empty and "max_depth" in frame:
        summary.insert(0, "max_depth", frame.max_depth.iloc[0])
    summary.insert(0, "objective", objective)
    atomic_csv(summary, result_path(output, objective, "summary"))
    return summary


def experiment_configs(args):
    """Keep each depth independent, including its tuning grid and outputs."""
    for depth in args.depths:
        run = argparse.Namespace(**vars(args))
        run.max_depth = depth
        run.output_dir = args.output_dir / f"depth{depth}"
        yield run


def main():
    args = parse_args()
    failed = False
    for run in experiment_configs(args):
        print(f"Starting {run.objective} with max depth {run.max_depth}", flush=True)
        # Failed trials at the first depth must not prevent the second-depth experiment.
        failed = run_experiment(run) or failed
    if failed:
        raise SystemExit("Some datasets/trials failed; inspect the objective CSV files in each output directory")


def select_dataset_paths(args):
    paths = sorted(args.data_dir.rglob("*.csv"))
    requested = set(args.datasets) if args.datasets else set(DATASETS)
    if not requested.issubset(DATASETS):
        raise SystemExit(f"Datasets outside the 14-dataset benchmark: {sorted(requested - set(DATASETS))}")
    paths = [p for p in paths if p.stem in requested]
    if args.datasets:
        missing = requested - {p.stem for p in paths}
        if missing:
            raise SystemExit(f"Datasets not found: {sorted(missing)}")
    if not paths:
        raise SystemExit(f"No selected benchmark datasets found under {args.data_dir}")
    return paths


def run_experiment(args):
    paths = select_dataset_paths(args)
    selected_names = {p.relative_to(args.data_dir).with_suffix("").as_posix() for p in paths}
    selected_seeds = getattr(args, "seeds", SEEDS)
    print(f"Selected {len(paths)} datasets from the 14-dataset runtime-reduced benchmark", flush=True)
    # Fail before starting a lengthy run if the compiled package is unavailable.
    import pystreed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Solver time limit: {args.time_limit:g}s per fit (not per trial)", flush=True)
    if args.objective == "f1" and args.max_depth >= 4:
        print("F1 depth >= 4: solution caches disabled; native fits isolated by default", flush=True)
    parameter, values = tuning_parameter(args)
    metadata = {
        **vars(args), **solver_cache_options(args), "optimization_task": "f1-score" if args.objective == "f1" else "cost-complex-accuracy",
        "dataset_policy": "runtime_reduced_14", "selected_datasets": sorted(selected_names),
        "selected_seeds": list(selected_seeds),
        "tuning_parameter": parameter, "tuning_grid": values, "selection_metric": "mcc",
        "tie_break": "smaller node cap" if args.objective == "f1" else "larger cost complexity",
        "seeds": SEEDS, "test_size": 0.2, "cv_folds": 5,
        "cv_shuffle": True, "positive_class": "larger sorted label (LabelEncoder class 1)",
        "std_ddof": 1, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "pystreed_source": pystreed.__file__,
        "runtime_definition": "CV fits and validation scoring plus final refit; excludes test scoring",
        "isolation_runtime_note": "isolated fits include worker startup and data transfer; solver timings are also recorded",
        "binarization": "quantile (5 thresholds); learned inside each fit",
        "incomplete_policy": "flag datasets without five successful seeds; omit their summary estimates",
    }
    trials, cv_rows, datasets = load_checkpoint(args, metadata)
    recorded = {(row["dataset"], row["seed"]) for row in trials
                if not (getattr(args, "retry_failed", False) and row["status"] == "failed"
                        and row["dataset"] in selected_names and row["seed"] in selected_seeds)}
    checkpoint = lambda: save_results(args.output_dir, trials, cv_rows, datasets, args.objective)
    for path in paths:
        name = path.relative_to(args.data_dir).with_suffix("").as_posix()
        if all((name, seed) in recorded for seed in selected_seeds):
            print(f"Skipping recorded dataset: {name}", flush=True)
            continue
        datasets[:] = [row for row in datasets if row["dataset"] != name]
        info = {"dataset": name, "path": str(path), "objective": args.objective, "max_depth": args.max_depth}
        try:
            data = pd.read_csv(path, sep=r"\s+", header=None)
            if data.shape[1] < 2 or data.isna().any().any():
                raise ValueError("Expected a target and features with no missing values")
            encoder = LabelEncoder().fit(data.iloc[:, 0])
            info.update(n_samples=len(data), n_features=data.shape[1] - 1,
                        n_classes=len(encoder.classes_))
            if len(encoder.classes_) != 2:
                info.update(status="skipped", reason="Target does not have exactly two classes")
                datasets.append(info)
                print(f"Skipping {name}: not binary", flush=True)
                continue
            X, y = data.iloc[:, 1:], encoder.transform(data.iloc[:, 0])
            info.update(status="included", negative_label=str(encoder.classes_[0]),
                        positive_label=str(encoder.classes_[1]))
        except Exception as exc:
            info.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            datasets.append(info)
            save_results(args.output_dir, trials, cv_rows, datasets, args.objective)
            continue
        datasets.append(info)
        for seed in selected_seeds:
            if (name, seed) in recorded:
                print(f"Skipping recorded trial: {name}, seed {seed}", flush=True)
                continue
            print(f"{args.objective}: depth {args.max_depth}: {name}: seed {seed}", flush=True)
            try:
                row = run_trial(args, name, X, y, seed, cv_rows, checkpoint)
            except Exception as exc:
                row = {"dataset": name, "seed": seed, "status": "failed", "objective": args.objective,
                       "max_depth": args.max_depth, "error": f"{type(exc).__name__}: {exc}"}
                print(f"  Failed: {row['error']}", flush=True)
            existing = next((index for index, saved in enumerate(trials)
                             if saved["dataset"] == name and saved["seed"] == seed), None)
            if existing is None:
                trials.append(row)
            else:
                # Replace only the selected failed attempt; keep it in place.
                trials[existing] = row
            summary = save_results(args.output_dir, trials, cv_rows, datasets, args.objective)
    summary = save_results(args.output_dir, trials, cv_rows, datasets, args.objective)
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"Results saved to {args.output_dir.resolve()}")
    return any(row["status"] == "failed" for row in trials + datasets
               if row["dataset"] in selected_names)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--_fit-worker":
        fit_worker(sys.argv[2], sys.argv[3])
    else:
        faulthandler.enable()
        main()
