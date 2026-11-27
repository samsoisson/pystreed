# MCC development baseline

Inspected and built on November 25, 2026, from commit
`11cdc8d029242fda7fc0142e628c3a602c3328a9`.
The local `origin` is `samsoisson/pystreed`; `upstream` is `AlgTUDelft/pystreed`.
The MCC objective is now implemented locally on `andrew-implementation`.
Changes are uncommitted. The baseline results below predate the implementation.

## MCC implementation handoff

Use `STreeDClassifier(optimization_task="mcc")` or the CLI's `-task mcc`.
The task is implemented in `include/tasks/mcc.h` and `src/tasks/mcc.cpp`.
It keeps additive `(FN, FP)` states, recovers TP/TN using the appropriate dataset's
class totals, and compares complete candidates by MCC at the root. C++ template
instantiations, CMake entries, CLI dispatch, Python bindings, and both Python
task allowlists have been updated.

Training requires both labels 0 and 1. Evaluation allows either or both classes.
Zero-denominator MCC is defined as zero; negative MCC is preserved. Internal
invalid/worst states score negative infinity rather than being treated as real
confusion matrices. Count products and bound-distance calculations use floating
point to avoid int32 multiplication overflow.

Testing is handed back to Andrew at his request. No further tests were run after
that request. Earlier checks built the extension and CLI, passed the C++ suites,
and passed exhaustive depth-2 Python comparisons. However, the full Python suite
did not complete: a depth-3 case crashed with terminal solving and bounds enabled.
The reduced input is saved locally at `build/mcc-regression.csv`; the standalone
F1 CLI also exited abnormally on that input. This shared solver issue is unresolved.
Do not treat the implementation as fully validated or benchmark-ready yet.

November 26 source-review follow-up: Andrew's diagnostic CLI printed a tree and
then trapped before the successful-close message. The CLI deletes an
`AbstractSolver*`, whose base class lacked a virtual destructor. Added a virtual
base destructor and marked `Solver`'s destructor `override`, fixing this undefined
behavior. This change has not been rebuilt or tested by the assistant; it is a
candidate fix for the CLI shutdown trap, not confirmation that the earlier Python
fit-time crash is resolved. Rebuild both CMake trees and the Python extension
before checking the behavior again.

November 27 follow-up: Andrew confirmed successful diagnostic CLI exits with
both branch caching and dataset caching, but the Python suite still crashes.
The saved native crash report places the failure in `ADataView::SplitData`,
called by `Tree<MCC>::ComputeTrainScore` during `Solver<MCC>::Solve`.
Added release-active checks for infeasible/unmatched reconstructed nodes and
invalid split features. These checks are diagnostic safeguards, not a confirmed
fix for the underlying reconstruction failure. They have not been built or tested
by the assistant; rebuild the Python extension and rerun the suite manually.

Commands for Andrew to run from the repository root:

```sh
# Rebuild after C++ edits.
build/venv/bin/python -m pip install --no-cache-dir --no-build-isolation --no-deps -e .
build/venv/bin/cmake --build build/cmake --parallel 2

# Task arithmetic checks and existing C++ regressions.
build/venv/bin/ctest --test-dir build/cmake --output-on-failure

# Python scoring, exhaustive-tree, and pruning/caching comparisons.
build/venv/bin/python -X faulthandler -m unittest discover -s test -p 'test_mcc.py' -v

# Diagnostic build already prepared; run this to investigate the crash.
build/mcc-debug/STREED -task mcc -file build/mcc-regression.csv -max-depth 3 -max-num-nodes 4 -min-leaf-node-size 2 -time 120
```

The diagnostic executable was compiled with AddressSanitizer and
UndefinedBehaviorSanitizer but was not run before testing was handed back.
The Python crash fixture is also generated reproducibly by
`test_pruning_caching_and_terminal_solver` in `test/test_mcc.py` (NumPy seed 42,
fifth generated dataset). No new automatic CI test step is enabled.

## Local build

Run these commands from the repository root. The virtual environment, CMake
outputs, and logs live in the already ignored `build/` directory.

```sh
python3 -m venv build/venv
build/venv/bin/python -m pip install 'setuptools>=42,<72.2.0' wheel 'pybind11>=2.12.0' cmake
build/venv/bin/python -m pip install 'pandas>=1.0.0' 'numpy>=1.18.0' 'scikit-learn>=1.6.0' scikit-survival typing_extensions
build/venv/bin/python -m pip install --no-build-isolation --no-deps -e .
build/venv/bin/cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
build/venv/bin/cmake --build build/cmake --parallel 2
build/venv/bin/ctest --test-dir build/cmake --output-on-failure
```

Install runtime dependencies separately with their default build isolation:
an initial combined `--no-build-isolation -e .` install failed because the
ECOS dependency needed NumPy while preparing its source-build metadata.
The separate dependency installation resolved this without source changes.

After editing C++ code, rerun the editable-install command to rebuild the Python
extension; rebuilding CMake alone does not rebuild the Python extension.
Python-only edits are picked up by the editable installation.

### Verified environment and checks

- macOS ARM64; Python 3.13.3; Apple Clang 21.0.0; CMake 4.4.5.
- PySTreeD 1.4.0 built from this checkout, installed editable in `build/venv`.
- NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, scikit-survival 0.28.0,
  pybind11 3.1.0, setuptools 72.1.0.
- CMake Release build: passed (library, CLI, and test executable).
- CTest: 1/1 passed in 7.67 seconds. The executable exercises 24 solver
  configurations across accuracy, F1, regression, and survival cases.
- `python -m pip check`: no broken requirements.
- The existing C++ sources produce compiler warnings, including an uninitialized
  base-class access in the regression code and format-security warnings in the
  test macros. No existing solver source was changed for this baseline.

Local evidence is saved in `build/cmake-configure.log`, `build/cmake-build.log`,
`build/ctest.log`, `build/python-build.log`, and `build/environment-freeze.txt`.
The freeze records the installed environment; its editable Git entry points to
the baseline commit, which does not include these new local documentation/example files.

## F1 reproduction

The existing `examples/unbalanced_example.py` runs 20 splits and extensive
cross-validation across several models. The new `examples/f1_example.py`
isolates a single F1 fit, uses hard predictions, and checks both the solver's
training score and evaluation scores against scikit-learn (absolute tolerance
`1e-12`). It also reports MCC, accuracy, confusion counts, runtime, and optimality.

```sh
# Soybean: depth 3, at most 7 branching nodes, stratified 80/20 split, seed 11.
build/venv/bin/python examples/f1_example.py

# Reproduce the existing C++ F1 fixture on all yeast rows, depth 2 / 3 nodes.
build/venv/bin/python examples/f1_example.py --dataset yeast --depth 2 --test-size 0

# Corresponding standalone C++ run.
build/cmake/STREED -task f1-score -file data/classification/yeast.csv -max-depth 2 -max-num-nodes 3 -time 120
```

The yeast fixture expects `(FN, FP) = (77, 463)`; its F1 score is
`0.5884146341463414`. With `--test-size 0`, only training performance is reported.
These examples verify the build and score plumbing; they are not a replacement
for the team's 14-dataset evaluation.

### Observed results

Both runs returned `is_optimal = true`. The training objective and the train/test
evaluation scores matched scikit-learn F1 to absolute tolerance `1e-12`.

| Run | Rows | F1 | MCC | Accuracy | TN / FP / FN / TP |
| --- | ---: | ---: | ---: | ---: | --- |
| Yeast training, depth 2 | 1,484 | 0.5884146341 | 0.3560355910 | 0.6361185984 | 558 / 463 / 77 / 386 |
| Soybean training, depth 3 | 504 | 0.8447204969 | 0.8192149731 | 0.9503968254 | 411 / 19 / 6 / 68 |
| Soybean held-out test | 126 | 0.8333333333 | 0.8055555556 | 0.9523809524 | 105 / 3 / 3 / 15 |

Observed Python `fit` times were approximately 0.012 seconds for yeast and
0.018 seconds for soybean, excluding imports and data loading. These small
examples do not predict runtime on the larger evaluation datasets.
Machine-readable results are in `build/f1-yeast.json` and `build/f1-soybean.json`;
the C++ result is in `build/f1-cli.log`.

## Complete task-registration map

Locations below refer to the baseline commit. Add MCC alongside F1 rather than
replacing the F1 entries. Proposed public task name: `mcc`; proposed C++ type:
`MCC`. The generic solver templates themselves do not need a new search algorithm.

### Build, dispatch, and Python validation

| File and baseline line | Existing registration | MCC change |
| --- | --- | --- |
| `CMakeLists.txt:73` and `:123` | F1 header and source lists | Add `include/tasks/mcc.h` and `src/tasks/mcc.cpp` |
| `include/tasks/tasks.h:17` | F1 task include | Include `tasks/mcc.h` |
| `src/solver/define_parameters.cpp:25` | Valid `task` values | Add `mcc` |
| `src/main.cpp:75` | CLI task branch | Construct `Solver<MCC>` and call `ReadData<MCC>` |
| `src/bindings.cpp:25` | `task_type` enum | Add MCC enum value |
| `src/bindings.cpp:43` | `get_task_type_code` | Map `mcc` string to enum value |
| `src/bindings.cpp:290` | `DefineSolver<F1Score>` | Register `DefineSolver<MCC>` |
| `src/bindings.cpp:324` | Solver initialization switch | Construct `Solver<MCC>` |
| `src/pystreed/base.py:22` | Base estimator parameter constraints | Add `mcc` to `StrOptions` |
| `src/pystreed/classification.py:91` | Classifier constructor allowlist | Add `mcc` |
| `src/pystreed/classification.py:119` | F1 scalar upper-bound warning | Extend the warning behavior for the partially ordered MCC task |

`setup.py:13` already discovers `src/**/*.cpp` recursively, so a new MCC source
file needs no separate setuptools source-list entry. Reusing `STreeDClassifier`
also needs no new export in `src/pystreed/__init__.py` or new project dependency.

### All nine explicit C++ template instantiations

Copy the corresponding F1 instantiation with `MCC` as the template argument.
Missing an entry can cause link failures even when the new task header compiles.

| File and baseline line | Template |
| --- | --- |
| `src/solver/solver.cpp:1406` | `Solver<MCC>` |
| `src/solver/branch_cache.cpp:252` | `BranchCache<MCC>` |
| `src/solver/dataset_cache.cpp:264` | `DatasetCache<MCC>` |
| `src/solver/cache.cpp:97` | `Cache<MCC>` |
| `src/solver/cost_combiner.cpp:517` | `CostCalculator<MCC>` |
| `src/solver/cost_storage.cpp:78` | `CostStorage<MCC>` |
| `src/solver/terminal_solver.cpp:516` | `TerminalSolver<MCC>` |
| `src/solver/similarity_lowerbound.cpp:156` | `SimilarityLowerBoundComputer<MCC>` |
| `src/utils/file_reader.cpp:172` | `FileReader::ReadData<MCC>(...)` |

### Task implementation and validation touchpoints

- `include/tasks/f1score.h`: reference for additive `(FN, FP)` state, hashing,
  dominance, bounds, and depth-two contributions. MCC needs its own root
  comparisons in `DominatesD0` and `DominatesD0Inv`; changing only a displayed
  score does not change the selected objective.
- `src/tasks/f1score.cpp`: reference for leaf contributions and the three score
  functions. `ComputeTrainScore` and `ComputeTrainTestScore` use training totals;
  `ComputeTestTestScore` uses evaluation-data totals.
- `include/tasks/optimization_task.h:66`: score comparison defaults to maximization.
- `include/model/container.h:261`: root candidate filtering invokes the task's
  root-dominance methods. Inspect this path when validating MCC selection.
- `include/solver/result.h:59` and `:92`: worst-solution placeholders are passed
  into score functions. Handle these separately from valid confusion counts.
- `src/pystreed/classification.py`, `fit`: add a clear binary-label check for MCC
  before calling C++; establish behavior for single-class input and label encoding.
- `test/correct_solution_test.cpp:60`, `:111`, `:270`, `:299`, `:318`: existing
  F1 fixture type, runner, expected result, setup, and invocation. Add MCC cases
  and independent exhaustive-search checks; preserve F1 cases.
- `README.md:116`: document the new objective when it is implemented.

Re-audit after merging changes from Samuel:

```sh
rg -n 'F1Score|f1score|f1-score' src include test CMakeLists.txt README.md
rg -n 'template .*F1Score' src
```

The repository's `new optimization task.md` explains the generic extension
contract; use the concrete F1 implementation as the template for MCC's
partially ordered state.
