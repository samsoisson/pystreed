"""Test preprocessing and experiment orchestration without the C++ extension."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("classification", [True, False])
def test_preprocessing_passes_seed_to_sklearn(monkeypatch, classification):
    b = load("tested_binarizer", ROOT / "src/pystreed/binarizer.py")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    seen = []
    classes = {
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
    }
    for name, cls in classes.items():
        def factory(_cls=cls, **kwargs):
            seen.append((_cls.__name__, kwargs["random_state"]))
            return _cls(**kwargs)
        monkeypatch.setattr(b, name, factory)
    X = np.arange(120, dtype=float).reshape(40, 3)
    y = np.arange(40) % 2 if classification else np.linspace(0, 2, 40)
    outputs = []
    for _ in range(2):
        transformer = b.Binarizer("tree", 3, 5, None, 2, random_state=13)
        transformer.fit(X, y)
        outputs.append(transformer.transform(X))
    np.testing.assert_array_equal(*outputs)
    assert len(seen) == 8  # Three threshold trees and one forest, twice.
    assert all(seed == 13 for _, seed in seen)
    assert clone(b.KThresholdBinarizer("tree", 3, random_state=13)).random_state == 13
    assert clone(b.RandomForestFeatureSelector(2, random_state=13)).random_state == 13


def experiment():
    return load("tested_trials", ROOT / "examples/binary_classification_trials.py")


@pytest.mark.parametrize("objective, best_parameter, expected", [
    ("accuracy", "best_cost_complexity", 0.01), ("f1", "best_max_num_nodes", 0),
])
def test_trial_uses_disjoint_folds_refits_and_prefers_stronger_tie(monkeypatch, capsys, objective, best_parameter, expected):
    module = experiment()
    fitted_sets = []
    seeds = []

    class FeasibleTree:
        def fit(self, X, y):
            self.indices = set(X.index)
            fitted_sets.append(self.indices)
            self.fit_result = SimpleNamespace(is_feasible=lambda: True, is_optimal=lambda: True)

        def predict(self, X):
            assert not self.indices.intersection(X.index)
            return X.iloc[:, 0].to_numpy(dtype=int)

    def factory(args, seed, alpha):
        seeds.append(seed)
        return FeasibleTree()

    monkeypatch.setattr(module, "make_model", factory)
    args = SimpleNamespace(objective=objective, cost_complexities=[0, 0.01], max_num_nodes=[0, 1], max_depth=3)
    y = np.tile([0, 1], 50)
    X = pd.DataFrame({"predictor": y})
    cv_rows = []
    row = module.run_trial(args, "synthetic", X, y, 11, cv_rows)
    assert len(cv_rows) == 10 and len(fitted_sets) == 11
    assert all(indices <= fitted_sets[-1] for indices in fitted_sets[:-1])
    assert seeds == [11] * 11
    assert row["n_train"] == 80 and row["n_test"] == 20
    assert row[best_parameter] == expected
    assert row["objective"] == objective
    assert all(row[m] == 1 for m in module.METRICS[:-1])
    assert row["training_runtime_s"] >= row["refit_runtime_s"] >= 0
    output = capsys.readouterr().out
    assert "[CV 1/10]" in output and "[CV 10/10]" in output
    assert "validation MCC=" in output
    assert "[Refit] selected" in output and "Refit finished" in output


def test_summary_requires_all_five_seeds_and_uses_sample_std():
    module = experiment()
    rows = [{"dataset": "a", "seed": seed, "status": "ok",
             **{metric: i for metric in module.METRICS}}
            for i, seed in enumerate(module.SEEDS)]
    summary = module.summarize(pd.DataFrame(rows)).iloc[0]
    assert summary.complete
    assert summary.mcc_mean == 2
    assert summary.mcc_std == pytest.approx(np.std(range(5), ddof=1))
    incomplete = module.summarize(pd.DataFrame(rows[:-1])).iloc[0]
    assert not incomplete.complete and np.isnan(incomplete.mcc_mean)


def test_rare_class_fails_before_training():
    module = experiment()
    with pytest.raises(ValueError, match="five training samples"):
        module.run_trial(None, "rare", pd.DataFrame({"x": range(20)}),
                         np.array([0] * 16 + [1] * 4), 11, [])


def test_mcc_exits_before_training_or_creating_output(monkeypatch, capsys, tmp_path):
    import sys
    module = experiment()
    assert module.DEFAULT_DEPTHS["mcc"] == (3, 4)
    output = tmp_path / "unused"
    monkeypatch.setattr(sys, "argv", ["trials", "--objective", "mcc", "--output-dir", str(output)])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    assert capsys.readouterr().err.strip() == "not implemented yet"
    assert not output.exists()


@pytest.mark.parametrize("objective, task, key, value", [
    ("accuracy", "cost-complex-accuracy", "cost_complexity", 0.01),
    ("f1", "f1-score", "max_num_nodes", 2),
])
def test_model_routes_objective_and_regularization(monkeypatch, objective, task, key, value):
    import sys
    module = experiment()
    monkeypatch.setitem(sys.modules, "pystreed", SimpleNamespace(STreeDClassifier=lambda **kwargs: kwargs))
    args = SimpleNamespace(objective=objective, cost_complexities=[0.01], max_num_nodes=[2],
                           max_depth=3, time_limit=600)
    model = module.make_model(args, 13, value)
    assert model["optimization_task"] == task
    assert model[key] == value
    assert model["random_seed"] == 13
    assert not model["hyper_tune"]
    assert ("cost_complexity" in model) == (objective == "accuracy")


def test_objective_outputs_do_not_overwrite_each_other_or_legacy_files(tmp_path):
    module = experiment()
    legacy = tmp_path / "trials.csv"
    legacy.write_text("existing results")
    for objective in ("accuracy", "f1"):
        trials = [{"dataset": "a", "seed": seed, "status": "ok", "objective": objective,
                   **{metric: 0.5 for metric in module.METRICS}} for seed in module.SEEDS]
        module.save_results(tmp_path, trials, [], [], objective)
    for objective in ("accuracy", "f1"):
        assert pd.read_csv(tmp_path / f"trials_{objective}.csv").objective.unique().tolist() == [objective]
        assert pd.read_csv(tmp_path / f"summary_{objective}.csv").objective.unique().tolist() == [objective]
        assert (tmp_path / f"cv_results_{objective}.csv").exists()
        assert (tmp_path / f"datasets_{objective}.csv").exists()
    assert legacy.read_text() == "existing results"


def test_f1_default_grid_and_incompatible_parameter_validation(monkeypatch):
    import sys
    module = experiment()
    monkeypatch.setattr(sys, "argv", ["trials", "--objective", "f1"])
    args = module.parse_args()
    runs = list(module.experiment_configs(args))
    assert [run.max_depth for run in runs] == [3, 4]
    assert [module.tuning_parameter(run)[1] for run in runs] == [list(range(8)), list(range(16))]
    assert [run.output_dir.name for run in runs] == ["depth3", "depth4"]
    monkeypatch.setattr(sys, "argv", ["trials", "--objective", "f1", "--cost-complexities", "0.01"])
    with pytest.raises(SystemExit) as exc:
        module.parse_args()
    assert exc.value.code == 2


@pytest.mark.parametrize("flags, depths, directories", [
    ([], [4, 5], ["depth4", "depth5"]),
    (["--objective", "accuracy"], [4, 5], ["depth4", "depth5"]),
    (["--objective", "accuracy", "--max-depth", "4"], [4], ["depth4"]),
    (["--objective", "f1"], [3, 4], ["depth3", "depth4"]),
    (["--objective", "f1", "--max-depth", "2"], [2], ["depth2"]),
])
def test_depth_defaults_and_explicit_override(monkeypatch, tmp_path, flags, depths, directories):
    import sys
    module = experiment()
    monkeypatch.setattr(sys, "argv", ["trials", "--output-dir", str(tmp_path / "out"), *flags])
    args = module.parse_args()
    runs = list(module.experiment_configs(args))
    assert [run.max_depth for run in runs] == depths
    assert [run.output_dir.name for run in runs] == directories
    assert args.output_dir == tmp_path / "out"


@pytest.mark.parametrize("objective", ["accuracy", "f1"])
def test_main_runs_both_depths_even_when_first_has_failed_trials(monkeypatch, objective):
    import sys
    module = experiment()
    monkeypatch.setattr(sys, "argv", ["trials", "--objective", objective])
    called = []

    def run(args):
        called.append((args.max_depth, args.output_dir.name))
        return args.max_depth == (4 if objective == "accuracy" else 3)

    monkeypatch.setattr(module, "run_experiment", run)
    with pytest.raises(SystemExit, match="Some datasets/trials failed"):
        module.main()
    depths = (4, 5) if objective == "accuracy" else (3, 4)
    assert called == [(depth, f"depth{depth}") for depth in depths]


@pytest.mark.parametrize("objective, depth, cache_enabled", [
    ("accuracy", 3, True), ("accuracy", 4, True), ("f1", 3, True),
    ("f1", 4, False), ("f1", 5, False),
])
def test_f1_deep_model_bypasses_crashing_native_cache(monkeypatch, objective, depth, cache_enabled):
    import sys
    module = experiment()
    monkeypatch.setitem(sys.modules, "pystreed", SimpleNamespace(STreeDClassifier=lambda **kwargs: kwargs))
    args = SimpleNamespace(objective=objective, max_depth=depth, max_num_nodes=[6],
                           cost_complexities=[0.01], time_limit=600)
    model = module.make_model(args, 11, 6 if objective == "f1" else 0.01)
    assert model["use_dataset_caching"] == cache_enabled
    assert not model["use_branch_caching"]
    assert model["max_depth"] == depth and model["time_limit"] == 600


def resume_metadata():
    return dict(objective="f1", max_depth=4, time_limit=600, tuning_parameter="max_num_nodes",
                tuning_grid=[0, 1], seeds=(11, 12, 13, 14, 15), test_size=.2,
                cv_folds=5, cv_shuffle=True, use_dataset_caching=False, use_branch_caching=False)


def test_resume_preserves_records_config_and_original_bytes(tmp_path):
    import json
    import zipfile
    module = experiment()
    args = SimpleNamespace(output_dir=tmp_path, objective="f1", max_depth=4,
                           resume=False, isolate_fits=True)
    metadata = resume_metadata()
    assert module.load_checkpoint(args, metadata) == ([], [], [])
    row = {"dataset": "anneal", "seed": 11, "status": "ok", "objective": "f1", "max_depth": 4,
           **{metric: 0.12345678912345678 for metric in module.METRICS}}
    module.save_results(tmp_path, [row], [], [{"dataset": "anneal", "objective": "f1", "max_depth": 4}], "f1")
    original = (tmp_path / "trials_f1.csv").read_bytes()
    config_original = (tmp_path / "config_f1.json").read_bytes()
    with pytest.raises(ValueError, match="use --resume"):
        module.load_checkpoint(args, metadata)
    args.resume = True
    trials, cv, datasets = module.load_checkpoint(args, metadata)
    assert trials == [row] and cv == [] and len(datasets) == 1
    assert (tmp_path / "config_f1.json").read_bytes() == config_original
    with zipfile.ZipFile(next(tmp_path.glob("resume_backup_*.zip"))) as backup:
        assert backup.read("trials_f1.csv") == original
    changed = {**metadata, "time_limit": 10}
    with pytest.raises(ValueError, match="different time_limit"):
        module.load_checkpoint(args, changed)


def test_isolated_worker_failure_is_caught_with_exit_code(monkeypatch, tmp_path):
    import subprocess
    import sys
    module = experiment()
    real_run = subprocess.run

    def fail(command, **kwargs):
        return real_run([sys.executable, "-c", "import os; os._exit(139)"], **kwargs)

    monkeypatch.setattr(module.subprocess, "run", fail)
    args = SimpleNamespace(isolate_fits=True, output_dir=tmp_path, objective="f1", max_depth=4)
    with pytest.raises(RuntimeError, match="Native fit worker failed"):
        module.fit_and_predict(args, 11, 6, pd.DataFrame({"x": [0, 1]}), np.array([0, 1]),
                               pd.DataFrame({"x": [0]}), "test_crash")
    assert (tmp_path / "fit_logs_f1/test_crash.log").exists()


def test_trial_reuses_saved_folds_and_checkpoints_new_folds(monkeypatch):
    module = experiment()
    args = SimpleNamespace(objective="accuracy", max_depth=3, cost_complexities=[0.01])
    cv = [{"dataset": "a", "seed": 11, "cost_complexity": .01, "fold": fold,
           "status": "ok", "validation_mcc": 1., "fit_runtime_s": 1.}
          for fold in (1, 2)]
    calls, checkpoints = [], []

    def predict(args, seed, value, X, y, predict_X, context):
        calls.append(context)
        return predict_X.iloc[:, 0].to_numpy(), True, .001, 0.

    monkeypatch.setattr(module, "fit_and_predict", predict)
    y = np.tile([0, 1], 50)
    result = module.run_trial(args, "a", pd.DataFrame({"x": y}), y, 11, cv,
                              lambda: checkpoints.append(len(cv)))
    assert len(calls) == 4  # Three missing folds and the refit.
    assert checkpoints == [3, 4, 5] and len(cv) == 5
    assert result["accuracy"] == 1.
    assert result["training_runtime_s"] >= 2.  # Includes reused fold costs.


def test_experiment_resume_skips_saved_seeds_and_preserves_other_datasets(monkeypatch, tmp_path):
    import sys
    import json
    module = experiment()
    monkeypatch.setattr(module, "DATASETS", ("a",))
    data = tmp_path / "data"
    output = tmp_path / "out"
    data.mkdir()
    output.mkdir()
    pd.DataFrame({"y": np.tile([0, 1], 50), "x": np.tile([0, 1], 50)}).to_csv(
        data / "a.csv", sep=" ", header=False, index=False)
    monkeypatch.setattr(sys, "argv", ["trials", "--objective", "f1", "--max-depth", "4",
                                     "--data-dir", str(data), "--resume"])
    args = module.parse_args()
    args.output_dir = output
    config = {**resume_metadata(), "tuning_grid": list(range(16))}
    (output / "config_f1.json").write_text(json.dumps(config))
    old = [{"dataset": name, "seed": seed, "status": "ok", "objective": "f1", "max_depth": 4,
            **{metric: .5 for metric in module.METRICS}} for name, seed in [("a", 11), ("other", 11)]]
    module.save_results(output, old, [], [{"dataset": "other", "status": "included"}], "f1")
    called = []

    def trial(args, name, X, y, seed, cv, checkpoint):
        called.append(seed)
        return {**old[0], "seed": seed}

    monkeypatch.setattr(module, "run_trial", trial)
    assert not module.run_experiment(args)
    assert called == [12, 13, 14, 15]
    result = pd.read_csv(output / "trials_f1.csv")
    assert len(result) == 6
    assert result[result.dataset == "a"].seed.tolist() == [11, 12, 13, 14, 15]
    assert len(result[result.dataset == "other"]) == 1


def test_crashed_fold_is_checkpointed_and_trial_gets_no_metrics(monkeypatch):
    module = experiment()
    args = SimpleNamespace(objective="f1", max_depth=4, max_num_nodes=[6])
    called, cv, checkpoints = [], [], []

    def predict(args, seed, value, X, y, predict_X, context):
        called.append(context)
        if "fold2" in context:
            raise RuntimeError("Native fit worker failed (exit 0xC0000005)")
        return predict_X.iloc[:, 0].to_numpy(), True, .001, 0.

    monkeypatch.setattr(module, "fit_and_predict", predict)
    y = np.tile([0, 1], 50)
    with pytest.raises(RuntimeError, match="Incomplete CV"):
        module.run_trial(args, "a", pd.DataFrame({"x": y}), y, 11, cv,
                         lambda: checkpoints.append(len(cv)))
    assert checkpoints == [1, 2, 3, 4, 5]
    assert [row["status"] for row in cv] == ["ok", "failed", "ok", "ok", "ok"]
    assert "0xC0000005" in cv[1]["error"]


def test_runtime_reduced_benchmark_selects_only_the_14_retained_datasets(tmp_path):
    module = experiment()
    excluded = {"australian-credit", "breast-wisconsin", "diabetes", "german-credit",
                "ionosphere", "letter", "pendigits", "splice-1", "vehicle"}
    assert len(module.DATASETS) == 14 and len(set(module.DATASETS)) == 14
    for name in set(module.DATASETS) | excluded:
        (tmp_path / f"{name}.csv").touch()
    args = SimpleNamespace(data_dir=tmp_path, datasets=None)
    paths = module.select_dataset_paths(args)
    assert {path.stem for path in paths} == set(module.DATASETS)
    assert not {path.stem for path in paths} & excluded


def test_explicit_subset_remains_within_runtime_reduced_benchmark(tmp_path):
    module = experiment()
    for name in ("anneal", "vote", "german-credit"):
        (tmp_path / f"{name}.csv").touch()
    args = SimpleNamespace(data_dir=tmp_path, datasets=["vote"])
    assert [path.stem for path in module.select_dataset_paths(args)] == ["vote"]
    args.datasets = ["german-credit"]
    with pytest.raises(SystemExit, match="outside the 14-dataset"):
        module.select_dataset_paths(args)


def test_cli_rejects_excluded_datasets(monkeypatch):
    import sys
    module = experiment()
    monkeypatch.setattr(sys, "argv", ["trials", "--datasets", "german-credit"])
    with pytest.raises(SystemExit) as exc:
        module.parse_args()
    assert exc.value.code == 2


def test_retry_targets_only_selected_failed_seed(monkeypatch, tmp_path):
    import json
    import sys
    module = experiment()
    monkeypatch.setattr(module, "DATASETS", ("a",))
    data, output = tmp_path / "data", tmp_path / "output"
    data.mkdir()
    output.mkdir()
    pd.DataFrame({"y": np.tile([0, 1], 50), "x": np.tile([0, 1], 50)}).to_csv(
        data / "a.csv", sep=" ", index=False, header=False)
    monkeypatch.setattr(sys, "argv", ["trials", "--objective", "f1", "--max-depth", "4",
                                     "--datasets", "a", "--seeds", "12", "--resume", "--retry-failed",
                                     "--data-dir", str(data)])
    args = module.parse_args()
    args.output_dir = output
    config = {**resume_metadata(), "tuning_grid": list(range(16))}
    (output / "config_f1.json").write_text(json.dumps(config))
    rows = [{"dataset": "a", "seed": seed, "status": "failed" if seed == 12 else "ok",
             "objective": "f1", "max_depth": 4, **{metric: seed / 100 for metric in module.METRICS}}
            for seed in module.SEEDS]
    rows.append({**rows[0], "dataset": "other", "status": "failed"})
    module.save_results(output, rows, [], [{"dataset": "a", "status": "included"}], "f1")
    called = []

    def trial(args, name, X, y, seed, cv, checkpoint):
        called.append(seed)
        assert seed == 12
        checkpoint()  # Until the retry completes, the prior failed row remains.
        assert pd.read_csv(output / "trials_f1.csv").iloc[1].status == "failed"
        return {**rows[1], "status": "ok"}

    monkeypatch.setattr(module, "run_trial", trial)
    assert not module.run_experiment(args)
    assert called == [12]
    after = module.read_records(output / "trials_f1.csv")
    assert len(after) == len(rows)
    for index, row in enumerate(rows):
        assert after[index] == ({**row, "status": "ok"} if index == 1 else row)
    summary = pd.read_csv(output / "summary_f1.csv")
    assert summary.loc[summary.dataset == "a", "complete"].item()


def test_retry_requires_resume(monkeypatch):
    import sys
    module = experiment()
    monkeypatch.setattr(sys, "argv", ["trials", "--retry-failed"])
    with pytest.raises(SystemExit) as exc:
        module.parse_args()
    assert exc.value.code == 2


def test_atomic_checkpoint_retries_transient_windows_lock(monkeypatch, tmp_path):
    module = experiment()
    path = tmp_path / "checkpoint.csv"
    path.write_text("old data")
    real_replace = module.os.replace
    attempts = []

    def replace(source, target):
        attempts.append(target)
        if len(attempts) < 3:
            assert path.read_text() == "old data"
            raise PermissionError("File temporarily locked")
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", replace)
    monkeypatch.setattr(module, "sleep", lambda _: None)
    module.atomic_csv(pd.DataFrame({"value": [42]}), path)
    assert len(attempts) == 3
    assert pd.read_csv(path).value.tolist() == [42]


def test_access_violation_retries_same_fit_with_dataset_cache(monkeypatch, tmp_path):
    import pickle
    module = experiment()
    args = SimpleNamespace(isolate_fits=True, output_dir=tmp_path, objective="f1", max_depth=4,
                           time_limit=600, max_num_nodes=[10])
    attempts = []
    X = pd.DataFrame({"x": [0, 1]})
    y = np.array([0, 1])

    def run(command, **kwargs):
        with Path(command[-2]).open("rb") as stream:
            options, seed, value, train, labels, predict = pickle.load(stream)
        attempts.append(options)
        assert seed == 13 and value == 10 and options["max_depth"] == 4 and options["time_limit"] == 600
        pd.testing.assert_frame_equal(train, X)
        np.testing.assert_array_equal(labels, y)
        if len(attempts) == 1:
            assert "_dataset_cache_override" not in options
            return SimpleNamespace(returncode=-1073741819)
        assert options["_dataset_cache_override"] is True
        with Path(command[-1]).open("wb") as stream:
            pickle.dump((np.array([0, 1]), True, .01, .001), stream)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.fit_and_predict(args, 13, 10, X, y, X, "exact_failure")
    assert len(attempts) == 2 and result[1] is True
    assert args._last_fit_execution["fit_attempts"] == 2
    assert args._last_fit_execution["use_dataset_caching"] is True
    assert module.solver_cache_options(args)["use_dataset_caching"] is False
    assert (tmp_path / "fit_logs_f1/exact_failure.log").exists()
    assert (tmp_path / "fit_logs_f1/exact_failure_dataset_cache_retry.log").exists()
