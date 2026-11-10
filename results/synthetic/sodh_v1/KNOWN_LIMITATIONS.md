# SODH v1 — frozen; known limitations (2026-11-11)

Frozen test fixture. Do not use its budget curves or reader rankings as evidence about memory.
Independent-context review findings (docs/18): cups are retrieval not tracking; process_count
needs the one-pressed-tick-is-one-press convention; test_ood is only longer/larger; heldout_mapping is
renamed symbols; task_interleave has 88-97 % alpha-twins between train and val/test; evidence
position was drawn from one distribution so Uniform-vs-Window rankings are generator artefacts.
`g0_report.json` is the original g0/1 result (frozen record); `g0_report.g0-2.json` is the
stronger audit on the same data (passes; reports alpha_twin_rate and the constant-answer floor).
