# Paper artifacts and Tier 1b harvesting

Install plotting dependencies with `python -m pip install -e ".[paper]"`.
For harvesting and final COCO evaluation, use `python -m pip install -e ".[harvest]"`.

From the repository root:

```powershell
python scripts/generate_paper_artifacts.py
python scripts/generate_paper_artifacts.py --report C:/Users/Wagur1/Desktop/HOI_DONG_9/results_tier1b/tier1b_val5k_final_report.json --output-dir docs/paper_artifacts
python scripts/auto_harvester.py --annotations C:/data/annotations/instances_val2017.json
```

The artifact generator defaults to `docs/reports/step3_report.json` and writes
`rd_curves.{pdf,png}`, `bd_rate_comparison.{pdf,png}`, `table_m1_m4_coco.tex`, and
`table_cross_detector.tex` into `docs/paper_artifacts`. PDF figures use embedded
TrueType fonts; PNG figures use 300 DPI. Include tables with `\input{...}` and
`\usepackage{booktabs}`. The tables use `table*` to span two-column layouts.

The RD plots show pooled AP50 and AP@[.50:.95], averaged across student seeds.
The horizontal axis is total bitstream bits over the evaluation set, with no
assumed frame rate. Table bitstream sizes are total bytes, averaged over seeds
for M2--M4. The main table includes all five CRFs; the transfer table summarizes
AP over the CRFs and gives the full-curve BD-rate for each detector.

BD-rate is computed separately for each seed against shared M1, then averaged.
All seeds must have defined BD-rates to report that mean. Recorded Tier 1b means
are used when present, including explicit nulls. For Step 3, MPEG Bjontegaard
and PAVA-PCHIP are recomputed from pooled AP; they are different analyses from
the report's strict raw-PCHIP gate. PAVA-PCHIP remains a sensitivity analysis.
Unavailable values appear as N/A in charts and em dashes in tables. A negative
BD-rate denotes savings; the main table's **BD reduction** column reverses that
sign, and reports one full-curve result per model rather than per-CRF results.
The Step 3 calibration restriction is disclosed in the generated captions.

The harvester uses the dispatcher's sorted account mapping and quota overrides.
Its defaults are `~/.kaggle_pool`,
`C:/Users/Wagur1/Desktop/HOI_DONG_9/results_tier1b`, and a 180-second interval.
Override them with `--pool-dir`, `--results-dir`, and `--poll-interval`.
`--annotations` is required to check the exact 5,000-image partition and run
COCOeval. The tool never pushes or restarts kernels.

`--once` runs one polling pass and finalizes if all shards are ready. It returns
zero for a successful check with shards still running, one for errors, and 130
when interrupted. The daemon retries failed checks, downloads, or finalization
on subsequent passes. Downloaded files live under `shard_0/` through `shard_18/`;
all Kaggle output pages are fetched. Incomplete downloads and malformed reports
are retried. A completed kernel counts toward progress only after its report
passes image, observation, CRF, seed, bitstream, and prediction validation.

Once all shards are ready, the tool verifies disjoint 5,000-image coverage and
matching provenance, invokes `merge_pool_5k.py`, then generates artifacts under
`{results_dir}/paper_artifacts`. The final report and artifacts are fingerprinted
in `{results_dir}/harvest_receipt.json`. A restart with unchanged inputs and
intact outputs reuses that receipt. A writer lock prevents simultaneous
harvesters; after a forced process termination, remove `.harvester.lock` only
after confirming that no harvester is still running.

Run the tests without contacting Kaggle:

```powershell
python -m pytest tests -v
```
