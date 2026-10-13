# Paper result mapping

`results_for_advisor/` contains convenient CSV copies of the paper tables.
The underlying evidence is retained at the paths used by the original scripts.

| Paper output | Archived evidence / regeneration |
|---|---|
| Tables 4–7 | Fixed-scale `report/all_repetitions.csv` (8,000 rows), `tables4_7_numeric.csv`, `tables4_7_paper.csv` |
| Table 8 | `report/table8_paired_instance_means.csv`, `report/table8_paired_statistics.csv`; independent `paper_artifacts/validation/verify_table8.py` |
| Figure 5 | `paper_artifacts/fixed_scale_figure5/build_figure5.py`; `tools/rebuild_saved.py --target figure5` |
| Table 9 | 120 compressed CP-SAT ablation raw records; `results_for_advisor/Table9_CP_SAT_ablation.csv` |
| Table 10 | 3,731 compressed DBD raw records; `paper_artifacts/dbd_10seeds_20260831/table9_*.csv`; independent `paper_artifacts/validation/verify_dbd_table10.py` |
| Figures 6–7 | DBD raw schedule replay and queue aggregation; `tools/rebuild_saved.py --target dbd` |

The fixed-scale report root is:

```text
cssd-fixed-multiseed-server32-w72-final/
  fixed_30_20_server32_w72_report_bundle/outputs/
    fixed_30_20_server32_w72/report/
```

Run `python tools/prepare_results.py` to restore the exact historical JSONL
records into their original paths. Their original sizes and hashes are in
`validation/RESULT_ARCHIVES.json`.

`tools/check_delivery.py` validates each formal task set and record identity;
Table 8 statistics use 80 instance means rather than treating repeated seeds
as independent instances. The DBD case contains 41 complete 91-day windows:
one deterministic FFCBSPT window plus ten seeds for each of four stochastic
algorithms. Each window contains 22,927 jobs.

Historical naming of Table 10 files as `table9_*` predates insertion of the
CP-SAT ablation table and is retained for source compatibility.
