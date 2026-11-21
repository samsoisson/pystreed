# Published Table 10: accuracy optimization

`streed_paper_table10_runtimes.csv` transcribes the user's Table 10: runtime in
seconds for maximizing accuracy at maximum depths 4 and 5, averaged over five
runs. It retains all 22 reported datasets and all three methods: DL8.5, MurTree,
and STreeD. Identifiers use the same lowercase spelling as the experiment CSVs.
The published average ranks are saved separately in
`streed_paper_table10_average_ranks.csv`; they are ranks, not runtime values.

`< 1` is a reported average below one second, not exactly zero or one. `-` marks
a timeout beyond 600 seconds, not an exact 600-second measurement. Neither
marker should be silently converted into an exact runtime. The source provides
no per-run measurements or standard deviations. Bold formatting is not available
in the pasted text and is not represented.

Table 10 omits datasets for which all methods finish within one second. In the
current 14-dataset cohort, Primary-Tumor and Segment are not reported in Table 10;
their missing rows are not filled with guessed values. Twelve retained datasets
have reported entries. Fico is included in the paper CSV although absent from
the local dataset collection.

Use Table 10 for the **accuracy depth-4/depth-5** published reference. Use
`streed_paper_table7_runtimes.csv` for the **F1 depth-3/depth-4** reference, as
clarified by Appendix E.2. The local accuracy runs use regularization, MCC-based
CV selection, and an 80/20 train/test split; their total runtime includes tuning
and refitting. Published runtime values therefore remain a reference comparison
rather than a matched speedup benchmark without aligning solver settings,
training data, hardware, and timing boundaries.

Values: user-supplied Table 10 transcription. Paper context: *Necessary and
Sufficient Conditions for Optimal Decision Trees using Dynamic Programming*,
Appendix G, Table 10:
https://proceedings.neurips.cc/paper_files/paper/2023/file/1d5fce9627e15c84db572a66e029b1fc-Paper-Conference.pdf
