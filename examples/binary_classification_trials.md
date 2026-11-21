# Binary classification trials

The experiment now uses a fixed **14-dataset runtime-reduced benchmark** for
both objectives and their configured default depths (accuracy: 4 and 5; F1: 3 and 4). It excludes the nine costly datasets discussed;
it does not claim an empirically measured ranking of all depth-4 F1 runtimes.

Included datasets: Anneal, Audiology, Heart-Cleveland, Hepatitis, Hypothyroid,
Kr-vs-kp, Lymph, Mushroom, Primary-Tumor, Segment, Soybean, Tic-Tac-Toe, Vote,
and Yeast.

Excluded datasets: Australian-Credit, Breast-Wisconsin, Diabetes, German-Credit,
Ionosphere, Letter, Pendigits, Splice-1, and Vehicle. `--datasets` can select a
smaller subset of the 14 but cannot re-enable excluded datasets. New configs
record the selected names; resume events record the current selection while
preserving the original configuration JSON.

To continue depth-4 F1 with this selection:

```powershell
python -u examples/binary_classification_trials.py --objective f1 --max-depth 4 --resume
```

Already-running processes retain the previous selection: stop the old run and
restart with the command above to apply the new selection. Previous recorded
results, including excluded datasets, are preserved on resume. Historical excluded-dataset results remain in raw exports, but the plotting
script now filters aggregates, graphs, and coverage to this same 14-dataset cohort.
The four-hour runtime target is not guaranteed.


Install the source checkout with `python -m pip install -e .`, then run:

```powershell
python examples/binary_classification_trials.py --objective accuracy
python examples/binary_classification_trials.py --objective f1
python examples/binary_classification_trials.py --objective mcc
```

The script recursively scans `data/classification/*.csv` (whitespace-delimited,
no header, target first) and retains only allowed benchmark datasets with exactly two target classes. Features
need not be binary. The larger sorted label is mapped to positive class 1; the
mapping is saved. Missing values and invalid files are flagged, not imputed.

Each dataset uses seeds 11, 12, 13, 14, and 15 for a stratified 80/20 split,
five shuffled stratified training folds, and the STreeD solver. Each fold fits a
fresh estimator, including its preprocessing, without access to validation/test
labels. Choose the training objective with `--objective`:

- `accuracy` (default): regularized classification accuracy (`cost-complex-accuracy`),
  tuning penalties `0, 0.0001, 0.0005, 0.001, 0.005, 0.01`. Exact validation ties
  prefer larger penalties. Override with `--cost-complexities`.
- `f1`: positive-class F1 (`f1-score`). This solver does not use `cost_complexity`,
  so tune branching-node caps instead: `0` through `2**max_depth - 1` (0 through
  7 for depth 3 and 0 through 15 for depth 4). Zero allows a constant tree. Exact validation ties
  prefer smaller node caps. Override with `--max-num-nodes`, e.g. `0 1 3 7`.
- `mcc`: prints exactly `not implemented yet` and exits successfully before
  importing the solver, training, or creating any output files.

For both implemented objectives, select the highest mean **validation MCC**.
The training objective and validation selection metric are separate choices.
Internal STreeD hyper-tuning is disabled. Incompatible tuning flags are rejected.
Accuracy runs maximum depth **4 then 5** by default. F1 runs maximum depth
**3 then 4**. MCC is reserved for depths **3 then 4** but still exits with
`not implemented yet`. Each implemented configuration uses all five seeds. An explicit `--max-depth N` runs only that depth. Every experiment
uses a 600-second solver limit **per fit**. Quantile
binarization uses five thresholds; optional feature limiting is disabled.

After selection, refit on all training data. Report held-out accuracy, binary
precision, recall, F1, MCC, and elapsed training runtime. Undefined precision,
recall, and F1 are zero. MCC follows sklearn's convention (zero for a zero
denominator). Total training runtime includes CV fitting and validation scoring
plus final refit, excluding file loading, train/test splitting, and test scoring.
Final refit runtime is also saved separately. Runtime itself can vary across runs.

Every successful candidate must have all five fold scores. A failed fold flags
the trial instead of silently comparing partial validation results. Datasets
with fewer than five minority-class training examples or missing test classes
are flagged. Feasible trees without proof of optimality are evaluated, with
their status saved for every CV fit and final refit.

Outputs under `results/binary_classification` use the objective as a filename
suffix. For example, accuracy writes `trials_accuracy.csv`, and F1 writes
`trials_f1.csv`. Both objectives separate outputs by depth:

- `results/binary_classification/depth3/trials_f1.csv` (and matching summary/CV/config files).
- `results/binary_classification/depth4/trials_f1.csv` (and matching summary/CV/config files).

The same depth subdirectory convention applies with a custom `--output-dir` or
an explicit `--max-depth`. Accuracy writes `depth4/trials_accuracy.csv` and `depth5/trials_accuracy.csv`
with matching summary/CV/config files.
Raw results and summaries include `max_depth`. Each depth has its own mean and
sample standard deviation over five seeds; results are not pooled across depths.
The second depth still runs when the first contains failed trials, and the command reports
failure after both experiments finish. Each run writes:


- `summary_<objective>.csv`: mean and sample standard deviation (`ddof=1`) of each metric,
  only for datasets with all five successful seed trials.
- `trials_<objective>.csv`: per-seed test metrics, selected regularization, validation MCC,
  runtime, final solver optimality, and failures.
- `cv_results_<objective>.csv`: validation MCC, fit time, solver optimality, and errors by fold.
- `datasets_<objective>.csv`: included/skipped/failed datasets, dimensions, and label mapping.
- `config_<objective>.json`: parameters, seeds, software versions, and runtime definition.

Every nonempty result CSV includes an `objective` column for later comparison.
Trial/CV rows identify the tuning parameter, with `best_cost_complexity` for
accuracy trials or `best_max_num_nodes` for F1 trials. Config files record the
training task, effective tuning grid, MCC selection metric, and tie-break rule.
Legacy unsuffixed CSV/JSON files are left untouched. Repeating an objective
requires `--resume` to preserve its recorded trials; use a different `--output-dir` to retain
multiple runs. No graphing or cross-objective processing is performed.

Results are checkpointed after each trial. A run exits unsuccessfully if any
dataset or trial fails. Inspect the CSV files before interpreting incomplete runs.

For a smaller smoke run or a separate output directory:

```powershell
python examples/binary_classification_trials.py --datasets anneal --max-depth 2 --time-limit 10 --output-dir results/smoke
```

Accuracy requires 155 fits per dataset at each depth (five seeds times 30 CV fits
plus one refit), or 310 fits for the default depths 4 and 5 together. F1 requires 205 fits at depth 3 and 405 at depth 4, so the default
F1 command requires 610 fits per dataset. Both objectives use the same data
splits/folds, seeds, and per-fit time limit, but depth-4 F1 explores larger trees,
and tuning grids and resulting total runtime budgets differ. A 600-second limit is per solver call, not an overall experiment limit.
Adjust `--max-depth`, `--time-limit`, or `--cost-complexities` explicitly when
needed; the chosen settings are recorded.

Randomized source preprocessing now accepts `random_state` and receives the
STreeD `random_seed`, including both threshold decision trees and the optional
random-forest feature selector. Direct preprocessing callers retain the previous
`random_state=None` default. STreeD's `random_seed=-1` retains unfixed randomness.

Run the focused tests with:

```powershell
python -m pytest test/test_binary_classification_trials.py -q
```

These tests cover real sklearn preprocessing and experiment orchestration using
a test double for the C++ solver; an installed compiled STreeD package is needed
for an actual experiment.

Remaining comparison gaps:

- F1 currently supports structural size regularization, not the accuracy solver's
  penalty. A shared regularization formulation would need additional solver work.
- MCC is available as a validation/test metric but not as a training objective.
- Validation still selects by MCC, even when training optimizes F1. Selecting by
  the training objective instead would be a separate experiment specification.
- F1 optimizes positive-class binary F1, using the saved label mapping, not macro
  or weighted F1. Time-limited feasible trees may lack proof of optimality.
- Same-objective runs now refuse to overwrite their files without `--resume`; legacy results are not automatically
  migrated. F1's exhaustive default node-cap grid grows exponentially with depth;
  use an explicit short grid for deeper trees.

To run the two default F1 experiments:

```powershell
python examples/binary_classification_trials.py --objective f1
```

To run only one depth explicitly:

```powershell
python examples/binary_classification_trials.py --objective f1 --max-depth 4
```

If specifying a shared `--max-num-nodes` grid for both default F1 depths, all
values must fit depth 3 (at most 7). Without that flag, each depth gets its own
full grid (0-7 or 0-15). Legacy F1 outputs in the base directory remain untouched.

Accuracy runs depths 4 then 5 by default; an explicit `--max-depth` overrides
this. Older accuracy exports in the base result directory are left untouched.
The plotter prefers new depth-directory exports over matching older base-directory
exports so rerunning accuracy does not double-count a configuration already exported in a depth directory.

Progress / troubleshooting:

The script prints before and after every CV fit and the final refit, including
node cap or penalty, fold number, elapsed seconds, validation MCC, and proof of
optimality. Depth-4 F1 runs 80 CV fits plus one refit per trial. The default
600-second limit applies separately to each solver fit, so a trial can take
hours. The solver checks its limit internally; this is not a hard process-level
timeout. Trials are saved only after the final refit and test evaluation.
Native fault diagnostics are enabled when running the script directly.

For a shorter diagnostic run that preserves the original experiment outputs:

```powershell
python -u examples/binary_classification_trials.py --objective f1 --max-depth 4 --datasets anneal --time-limit 10 --output-dir results/f1_diagnostic
```

Shorter limits may return feasible trees without proof of optimality; those
runs have different solver budgets and should not replace the primary results.
If the process returns to the prompt unexpectedly, capture the last progress
line, any traceback, and PowerShell's `$LASTEXITCODE`. A silent computing fit
and a terminated process require different diagnoses.

Native F1 crash workaround:

The installed native solver reproduced an access violation during a short-limit
`anneal` depth-4 trial. A short diagnostic trial completed with both solution caches disabled, but a subsequent real run still crashed.
The experiment therefore sets `use_dataset_caching=False` and
`use_branch_caching=False` for F1 at depths 4 and above. Accuracy and depth-3 F1
keep their existing settings. Effective cache settings are recorded in config
JSON. The objective, depth, tuning grid, seeds, and time limit are unchanged.
Caching affects runtime and which feasible solution is reached before a timeout;
check recorded optimality before claiming all results are globally optimal.
This cache setting alone is insufficient. Process isolation below contains remaining native crashes; it does not repair or rebuild the extension. Older depth-3 results remain untouched when running
with `--max-depth 4`.

Crash containment and resume:

F1 fits now run in child processes by default. A native access violation terminates
only the worker; the parent saves the error and proceeds. Every CV fit has a log
under `depthN/fit_logs_f1/`, named by dataset, seed, node cap, and fold (or refit).
Windows worker error dialogs are suppressed so failed children can exit.
An access violation is typically reported as exit `0xC0000005`.

This is containment, not a fix for the compiled solver. Any candidate with a
failed fold invalidates that trial; no test metrics are fabricated or based on
partial CV. Recorded failed trials remain failed unless `--retry-failed` is specified with `--resume`.
The experiment still reports failure at the end if any trial failed. Re-running
an interrupted trial can reuse its successful saved folds; failed folds of that
unrecorded trial are retried.

To continue the existing depth-4 F1 run without rerunning recorded trials:

```powershell
python -u examples/binary_classification_trials.py --objective f1 --max-depth 4 --resume
```

The parent checkpoints CV rows after **each fold** and uses atomic replacement
for each output CSV. `--resume` loads previous trial/CV/dataset records and checks
that the objective, depth, tuning grid, seed list, time limit, and cache settings
match. It skips every recorded dataset/seed pair (including recorded failed
trials), preserves the original configuration JSON, and stores an exact ZIP
backup of the prior CSV/JSON bytes before modifying results. Resume events record
whether process isolation was enabled. The ZIP backup is not rediscovered as
an extra result by the graphing script. Only one experiment may write a given
objective/depth directory at a time.

Without `--resume`, existing output files cause an error rather than being
overwritten. Use another `--output-dir` for an independent experiment. Completed
depth-3 trials and other objectives are never changed by a depth-4 F1 resume.

Isolation adds Python startup and data-transfer costs to total training runtime.
`solver_fit_runtime_s` (CV) and `solver_refit_runtime_s` (trials) record the fit
itself separately; `isolated_fits` identifies trial execution mode. Total runtime
includes worker overhead and checkpointing, plus prior successful fold costs
when resuming; final test prediction/metric calculation is excluded. Existing
pre-isolation results retain their recorded runtime, so timing comparisons across
the execution-mode change require care. Accuracy keeps in-process fitting by
default; `--isolate-fits` enables the same protection for it. `--no-isolate-fits`
is available for debugging F1, but native faults can then kill the parent again.

If a native crash persists, its worker log and last progress line identify the
fit that must be debugged in the installed C++ extension. A successful bounded
smoke run does not establish that the full 600-second run is crash-free.

To run the default accuracy depths 4 and 5 while preserving existing trials:

```powershell
python -u examples/binary_classification_trials.py --objective accuracy --resume
```

Depth 4 uses its existing checkpoint; depth 5 starts a new configuration if no
outputs exist. Historical depth-3 results are not deleted or changed. An explicit
`--max-depth` continues to select a single depth for any implemented objective.

Targeted failed-trial retries:

```powershell
python -u examples/binary_classification_trials.py --objective accuracy --max-depth 5 --datasets primary-tumor --seeds 12 --resume --retry-failed
```

`--retry-failed` requires `--resume`. It retries only recorded failures inside the
selected dataset/seed scope; successful trials are always skipped. `--seeds`
accepts a subset of 11, 12, 13, 14, 15, without changing the five-seed protocol
used for summary statistics. Successful saved folds are reused. The old failed
trial remains in checkpoints until the retry completes, then its row is replaced
in place (without creating a duplicate). The resume ZIP backup retains the old
record. Other trials and the original config remain unchanged. Summary means
and standard deviations are recomputed after the replacement. Checkpoint writes
retry transient Windows permission errors six times before reporting a failure.

Native access-violation fallback:

Audiology seed 13, depth 4, node cap 10, fold 5 reproducibly crashed with both
solution caches disabled, but completed with proof of optimality when dataset
caching was enabled. If an isolated F1 fit with dataset caching disabled exits
with Windows access violation `0xC0000005`, the parent now retries that **same
fit once** with dataset caching enabled. The objective, data split, seed, depth,
node cap, and solver time limit are unchanged. Other error types are not retried
this way; if both attempts fail, the fold/trial remains failed.

The original crash log is preserved; the fallback log ends with
`_dataset_cache_retry.log`. CV rows record `fit_attempts`, `use_dataset_caching`,
and `use_branch_caching` for successful fits; the trial's `refit_execution`
records final-refit settings. Fold and total wall times include the failed
attempt plus fallback. `solver_fit_runtime_s` records the successful fit itself.
Saved successful CV folds are not recomputed, and successful trials are skipped.
This is an execution-strategy fallback, not a repair to the compiled extension.
For runtime comparisons, disclose this policy and which folds used it.
