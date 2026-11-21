# Compare saved experiment results

Run from the source checkout (requires pandas, NumPy, and Matplotlib):

```powershell
python examples/plot_binary_classification_results.py
```

This separate script reads the saved experiment summaries, including nested
`depth3` and `depth4` directories, and creates four bar charts: test accuracy,
test F1, test MCC, and total training runtime. Each bar is an optimization
objective / maximum-depth configuration, labeled with the number of datasets.
No training, tuning, or test-set selection occurs during plotting.

For each dataset, use its existing mean over the five seeds; then take an
unweighted arithmetic mean across datasets. Large datasets do not get extra
weight. By default, use only datasets with complete, finite, valid results from
all five trials in every discovered configuration. This keeps dataset coverage
identical across bars. The CSV coverage report lists missing/incomplete results
and datasets excluded because another configuration is incomplete. It cannot
list datasets absent from every input summary.

To use another result folder or each configuration's available complete datasets:

```powershell
python examples/plot_binary_classification_results.py --results-dir results/binary_classification --output-dir results/comparison_graphs
python examples/plot_binary_classification_results.py --dataset-policy available
```

The available policy may compare different datasets, so interpret its bars with
care. Invalid/incomplete values are excluded for all four metrics together.

The default output directory is `<results-dir>/graphs`. It contains:

- `accuracy.png` / `.svg`
- `f1.png` / `.svg`
- `mcc.png` / `.svg`
- `training_runtime_s.png` / `.svg` (seconds, linear scale)
- `configuration_averages.csv`: the plotted values, dataset counts, and sample
  standard deviations across dataset means and across seed-level averages.
- `dataset_coverage.csv`: inclusion decisions for dataset/configuration pairs.
- `plot_config.json`: input file paths and aggregation policy.

Files with the same output names are replaced on reruns. Existing experiment
results are read only. Figures are saved without opening an interactive window.
Error bars show ?1 sample standard deviation across five seed-level averages.
For each seed, average over the included datasets with equal weight; then compute
sample SD (`ddof=1`) across those five averages. `seed_averages.csv` saves those
values, and `configuration_averages.csv` includes `<metric>_seed_std`. The separate
`<metric>_dataset_std` fields describe between-dataset variation and are not used
as graph error bars. These are standard deviations, not confidence intervals.

Input discovery reads `summary.csv` and `summary_<objective>.csv` recursively.
Matching trial CSVs provide seed-level error bars; CV files are not used. New summaries contain `objective` and `max_depth`; older
files use matching `config.json` / `config_<objective>.json` metadata. Required
columns include `dataset`, `complete`, `successful_trials`, and `accuracy_mean`,
`f1_mean`, `mcc_mean`, `training_runtime_s_mean`. The legacy `summary.csv` is
superseded by `summary_accuracy.csv` in the same directory. New depth-directory exports also supersede matching older base-directory exports.
Other duplicate dataset/objective/depth records cause an error rather than double-counting runs.
Use a results directory containing only the intended experiment exports.

Open requirements / limits:

- “Published results” is interpreted as this project's saved CSV exports.
  External paper tables would need to be converted to this schema first.
- Dataset weights and missing-result handling were unspecified; equal weights
  and shared complete datasets are the defaults, with `available` as an option.
- Seed standard-deviation error bars are included; significance testing is not performed.
- Runtime includes tuning plus final refit. Accuracy and F1 have different
  tuning grids; runtime comparisons therefore measure the full experiment
  procedure, not only solver speed. Hardware and run conditions must also match
  for a meaningful timing comparison; the plotter cannot establish this.
- Only discovered configurations are plotted; missing methods/depths are not
  fabricated. If only one configuration exists, the script says so. MCC training
  remains unimplemented, though MCC evaluation plots work for other objectives.

Tests:

```powershell
python -m pytest test/test_plot_binary_classification_results.py -q
```

Runtime-reduced 14-dataset cohort:

The plotting command now uses the same `DATASETS` allowlist as the training
script. It reads that constant from the sibling source file without executing
training code. All aggregates, graphs, and the coverage CSV are filtered before
averaging; historical results from the nine excluded datasets are ignored.
`plot_config.json` records the selected cohort and excluded historical names.
Raw training results are never deleted or modified.

The cohort is: Anneal, Audiology, Heart-Cleveland, Hepatitis, Hypothyroid,
Kr-vs-kp, Lymph, Mushroom, Primary-Tumor, Segment, Soybean, Tic-Tac-Toe, Vote,
and Yeast. This filtering applies to both `common` and `available` policies.
An incomplete retained dataset is still excluded from the corresponding mean,
so bar labels may show fewer than 14 datasets. `common` keeps the same complete
subset across every configuration; `available` uses each configuration's complete
members of this cohort. The coverage report contains only retained datasets.

To regenerate using the current available-results policy:

```powershell
python examples/plot_binary_classification_results.py --dataset-policy available
```

Accuracy depth 3 is excluded from all plots, aggregate CSVs, and coverage reports.
The comparison includes accuracy depths 4 and 5, and F1 depths 3 and 4, when
available. Raw historical accuracy depth-3 results remain on disk.

Paper versus experiment runtime bar charts:

```powershell
python examples/plot_paper_runtime_comparison.py
```

This separate script saves four two-color charts under
`results/binary_classification/graphs/paper_comparison/`: accuracy depth 4,
accuracy depth 5, F1 depth 3, and F1 depth 4 (PNG and SVG). Each chart has one
paper STreeD bar and one experiment bar for each of the same 14 datasets. It
uses Table 10 for accuracy and Table 7 for F1, and reads the training cohort from
the existing plotter's shared helper. Existing aggregate metric graphs and raw
trial results are left unchanged.

Experiment bars default to the mean `refit_runtime_s` over all five successful
seeds. To compare the full tuning-plus-refit procedure instead:

```powershell
python examples/plot_paper_runtime_comparison.py --runtime-metric training_runtime_s
```

The default axis is logarithmic; `--scale linear` selects a linear runtime axis.
`--results-dir`, `--paper-dir`, and `--output-dir` select alternative paths. A
rerun overwrites this script's own output files in its chosen output directory.

Reported `< 1` times use hatched paper bars drawn at the **one-second upper
bound**, labeled `<1`. Timeout bars use the timeout threshold as a **lower-bound
marker**, labeled `>600`; they are not exact durations. A missing paper entry
has no bar and is labeled `N/R`. In particular, Table 10 omits Primary-Tumor and
Segment. Missing/failed experiment trials produce an `Incomplete` annotation,
not an average based on fewer than five seeds. Complete trial keys must be
unique. Experiment bars show ?1 sample SD (`ddof=1`) across the five seeds.
Paper bars have no error bars because no per-run variability was reported.
Lower runtime whiskers extending below the visible axis floor are clipped and
marked with downward triangles; the true standard deviations remain in the CSV.
Plot limits include upper whiskers and numeric labels sit above them.

`paper_runtime_comparison.csv` saves all 56 dataset/configuration pairs, exact
paper runtimes separately from bounds, experiment five-seed mean/std, statuses,
chosen timing metric, and source paths. `comparison_config.json` records the
cohort, timing choice, scale, and censoring convention.

These charts are a **reference comparison**. The experiment's split, selected
complexity, caching/fallback settings, and hardware may differ from the paper.
Final refit timing is closer to a single optimization call than full tuning
runtime, but neither should be presented as a matched speedup measurement.

Tests:

```powershell
python -m pytest test/test_plot_paper_runtime_comparison.py -q
```

The aggregate plotter requires matching per-trial data for every included dataset
and checks that trial means agree with saved summary means. Missing seed records
or stale summaries produce an error instead of guessed error bars. Both plotting
commands regenerate their PNG/SVG outputs with these error bars by default.
