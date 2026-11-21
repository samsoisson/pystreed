import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def plotter():
    path = Path(__file__).resolve().parents[1] / "examples/plot_binary_classification_results.py"
    spec = importlib.util.spec_from_file_location("plot_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(root, objective="accuracy", depth=3, datasets=("a", "b"), legacy=False):
    root.mkdir(parents=True, exist_ok=True)
    suffix = "" if legacy else f"_{objective}"
    frame = pd.DataFrame([
        {"dataset": dataset, "complete": True, "successful_trials": 5,
         "accuracy_mean": value, "f1_mean": value, "mcc_mean": value,
         "training_runtime_s_mean": 10 * value}
        for dataset, value in zip(datasets, [0.2, 0.8, 0.5])
    ])
    frame.to_csv(root / f"summary{suffix}.csv", index=False)
    pd.DataFrame([
        {"dataset": row["dataset"], "seed": seed, "status": "ok",
         **{metric: row[f"{metric}_mean"] for metric in
            ("accuracy", "f1", "mcc", "training_runtime_s")}}
        for row in frame.to_dict("records") for seed in range(11, 16)
    ]).to_csv(root / f"trials{suffix}.csv", index=False)
    (root / f"config{suffix}.json").write_text(json.dumps({"objective": objective, "max_depth": depth}))


def test_legacy_metadata_and_new_export_precedence(tmp_path):
    module = plotter()
    export(tmp_path, legacy=True)
    frame, _ = module.read_summaries(tmp_path)
    assert set(frame.objective) == {"accuracy"} and set(frame.max_depth) == {3}
    export(tmp_path)
    frame, sources = module.read_summaries(tmp_path)
    assert len(frame) == 2 and len(sources) == 1
    assert Path(sources[0]).name == "summary_accuracy.csv"


def test_equal_dataset_weights_and_common_dataset_selection(tmp_path):
    module = plotter()
    export(tmp_path)
    export(tmp_path / "depth4", "f1", 4, ("a", "c"))
    frame, _ = module.read_summaries(tmp_path)
    common, coverage = module.aggregate_results(frame)
    assert common.n_datasets.tolist() == [1, 1]
    assert common.accuracy_mean.tolist() == [0.2, 0.2]
    assert coverage[coverage.included].dataset.tolist() == ["a", "a"]
    available, _ = module.aggregate_results(frame, "available")
    assert available.n_datasets.tolist() == [2, 2]
    assert available.accuracy_mean.tolist() == [0.5, 0.5]
    assert available.training_runtime_s_mean.tolist() == [5, 5]


def test_incomplete_dataset_excluded_across_all_configurations(tmp_path):
    module = plotter()
    export(tmp_path)
    export(tmp_path / "depth3", "f1")
    path = tmp_path / "depth3/summary_f1.csv"
    data = pd.read_csv(path)
    data.loc[1, "complete"] = False
    data.to_csv(path, index=False)
    frame, _ = module.read_summaries(tmp_path)
    aggregate, coverage = module.aggregate_results(frame)
    assert aggregate.n_datasets.tolist() == [1, 1]
    assert coverage[~coverage.included].shape[0] == 2


def test_duplicate_configuration_datasets_rejected(tmp_path):
    module = plotter()
    export(tmp_path)
    export(tmp_path / "another-run")
    with pytest.raises(ValueError, match="Duplicate configuration/dataset"):
        module.read_summaries(tmp_path)


def test_missing_depth_metadata_rejected(tmp_path):
    module = plotter()
    export(tmp_path)
    (tmp_path / "config_accuracy.json").unlink()
    with pytest.raises(ValueError, match="missing max_depth"):
        module.read_summaries(tmp_path)


def test_empty_intersection_rejected_and_four_graphs_exported(tmp_path):
    module = plotter()
    export(tmp_path)
    export(tmp_path / "depth4", "f1", 4, ("c",))
    frame, _ = module.read_summaries(tmp_path)
    with pytest.raises(ValueError, match="No complete datasets"):
        module.aggregate_results(frame)
    aggregate, coverage = module.aggregate_results(frame, "available")
    aggregate, _ = module.add_seed_statistics(aggregate, coverage, frame)
    output = tmp_path / "graphs"
    output.mkdir()
    module.plot_results(aggregate, output)
    for metric in module.METRICS:
        for extension in ("png", "svg"):
            assert (output / f"{metric}.{extension}").stat().st_size > 100


def test_depth_directory_export_supersedes_old_base_export(tmp_path):
    module = plotter()
    export(tmp_path, legacy=True)
    export(tmp_path)
    export(tmp_path / "depth3")
    export(tmp_path / "depth4", depth=4)
    frame, sources = module.read_summaries(tmp_path)
    assert len(frame) == 4 and len(sources) == 2
    assert set(frame.max_depth) == {3, 4}
    assert all(Path(path).parent.name in {"depth3", "depth4"} for path in sources)


def test_shared_training_cohort_has_14_datasets():
    module = plotter()
    cohort = module.benchmark_datasets()
    assert len(cohort) == len(set(cohort)) == 14
    assert "anneal" in cohort and "primary-tumor" in cohort
    assert not set(cohort) & {"german-credit", "ionosphere", "splice-1", "letter", "vehicle"}


def test_main_excludes_historical_datasets_from_graphs_averages_and_coverage(monkeypatch, tmp_path):
    import sys
    module = plotter()
    export(tmp_path, depth=4, datasets=("anneal", "vote", "german-credit"))
    export(tmp_path / "depth3", depth=3, datasets=("anneal", "vote", "german-credit"))
    export(tmp_path / "depth3", "f1", 3, ("anneal", "vote", "german-credit"))
    export(tmp_path / "depth5", depth=5, datasets=("anneal", "vote", "german-credit"))
    export(tmp_path / "depth4", "f1", 4, ("anneal", "vote", "german-credit"))
    output = tmp_path / "graphs"
    monkeypatch.setattr(sys, "argv", ["plot", "--results-dir", str(tmp_path),
                                     "--output-dir", str(output), "--dataset-policy", "available"])
    module.main()
    averages = pd.read_csv(output / "configuration_averages.csv")
    coverage = pd.read_csv(output / "dataset_coverage.csv")
    assert averages.n_datasets.tolist() == [2, 2, 2, 2]
    assert averages.accuracy_mean.tolist() == [.5, .5, .5, .5]
    expected = {("accuracy", 4), ("accuracy", 5), ("f1", 3), ("f1", 4)}
    assert set(zip(averages.objective, averages.max_depth)) == expected
    assert set(zip(coverage.objective, coverage.max_depth)) == expected
    assert set(coverage.dataset) == {"anneal", "vote"}
    config = json.loads((output / "plot_config.json").read_text())
    assert len(config["dataset_cohort"]) == 14
    assert config["excluded_historical_datasets"] == ["german-credit"]
    assert (output / "accuracy.png").exists()
    assert pd.read_csv(output / "seed_averages.csv").shape[0] == 20
    assert averages.accuracy_seed_std.eq(0).all()


def test_error_bars_use_five_seed_averages_not_dataset_standard_deviations(tmp_path):
    import numpy as np
    module = plotter()
    export(tmp_path, depth=4)
    summary_path = tmp_path / "summary_accuracy.csv"
    summary = pd.read_csv(summary_path)
    for metric in module.METRICS:
        summary[f"{metric}_mean"] = 5 if metric == "training_runtime_s" else .5
    summary.to_csv(summary_path, index=False)
    rows = []
    for name in ("a", "b"):
        for seed, value in zip(range(11, 16), np.linspace(0, 1, 5)):
            value = value if name == "a" else 1 - value
            rows.append({"dataset": name, "seed": seed, "status": "ok",
                         **{metric: value * (10 if metric == "training_runtime_s" else 1)
                            for metric in module.METRICS}})
    pd.DataFrame(rows).to_csv(tmp_path / "trials_accuracy.csv", index=False)
    frame, _ = module.read_summaries(tmp_path)
    aggregate, coverage = module.aggregate_results(frame)
    statistics, seed_means = module.add_seed_statistics(aggregate, coverage, frame)
    assert len(seed_means) == 5
    assert seed_means.accuracy.eq(.5).all()
    assert statistics.accuracy_seed_std.item() == 0
    # Same per-dataset SD, but correlated seed fluctuations do not cancel.
    for row in rows:
        if row["dataset"] == "b":
            for metric in module.METRICS:
                row[metric] = (row["seed"] - 11) / 4 * (10 if metric == "training_runtime_s" else 1)
    pd.DataFrame(rows).to_csv(tmp_path / "trials_accuracy.csv", index=False)
    statistics, _ = module.add_seed_statistics(aggregate, coverage, frame)
    assert statistics.accuracy_seed_std.item() == pytest.approx(np.std(np.linspace(0, 1, 5), ddof=1))


def test_seed_statistics_reject_missing_trial_seed(tmp_path):
    module = plotter()
    export(tmp_path, depth=4)
    path = tmp_path / "trials_accuracy.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    frame, _ = module.read_summaries(tmp_path)
    aggregate, coverage = module.aggregate_results(frame)
    with pytest.raises(ValueError, match="five successful seeds"):
        module.add_seed_statistics(aggregate, coverage, frame)
