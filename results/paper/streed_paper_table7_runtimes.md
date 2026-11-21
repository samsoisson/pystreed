# Published Table 7: F1 optimization

`streed_paper_table7_runtimes.csv` transcribes the table supplied by the user:

> Table 7: Runtime (s) for minimizing biobjective misclassification score for two
> classes, with timeouts beyond 600s denoted with ‘-’. Best results are shown in
> bold. Average of five runs.

The CSV retains both MurTree and STreeD results for maximum depths 3 and 4,
plus dataset sample counts, class counts, feature counts, run count, and timeout.
Dataset names use lowercase identifiers matching the experiment CSVs. COMPAS
and Fico are retained even though they are not in the current experiment's
23-dataset collection. Class counts are copied as given in the paper table.

Runtime fields preserve the reported strings:

- `< 1`: a reported average below one second, not an exact value of zero or one.
- `-`: timeout beyond 600 seconds, not a zero runtime or an exact 600 seconds.
- Numeric values: the reported average runtime in seconds.

The supplied table does not contain per-run times or standard deviations, so
paper-runtime error bars cannot be reconstructed. Bold formatting was not
available in the supplied plain text and is not represented in the CSV.

Values: user-provided Table 7 transcription. Appendix E.2 of the paper confirms
that these experiments maximize **F1 at depths 3 and 4**; the biobjective caption
refers to the false-positive/false-negative formulation. The matching published
accuracy reference is Table 10, saved in `streed_paper_table10_runtimes.csv`.
Local total runtime includes CV tuning and refitting, and solver settings differ,
so these remain reference comparisons rather than matched speedup measurements.

Paper: *Necessary and Sufficient Conditions for Optimal Decision Trees using
Dynamic Programming*, Appendix E.2:
https://proceedings.neurips.cc/paper_files/paper/2023/file/1d5fce9627e15c84db572a66e029b1fc-Paper-Conference.pdf#page=36
