# Release validation — 2026-10-13

Validated in a new virtual environment on Windows with Python 3.10.5 and
the dependency versions in `requirements-lock.txt` and
`requirements-constraints.txt`. `pip check` passed.

| Check | Result |
|---|---|
| Pinned upstream input downloads | PASS: all 547 SHA-256 values match the experiment inputs |
| Original copied source/configuration files | PASS: all hashes in `validation/SOURCE_COPY_MANIFEST.json` match |
| Restored historical records and schedule CSV | PASS: original byte lengths and SHA-256 values match |
| Saved evidence audit | PASS: 8,000 fixed-scale runs, 3,731 DBD daily records, 120 ablation records |
| Formal task dry runs | PASS: 8,000 / 2,730 / 1,001 tasks |
| Independent Table 8 statistics | PASS |
| Independent Table 10 statistics | PASS |
| Fixed-scale runner tests | 23 passed |
| Core IBS-CABC tests | 54 passed |
| Comparison baseline tests | 3 passed, 3 skipped |
| Three-algorithm DBD tests | 8 passed |
| FFCBSPT/Kraul-GA tests | 20 passed |
| Fixed-scale end-to-end smoke | PASS: 64/64 tasks, zero failures, report integrity PASS |
| Three-algorithm DBD end-to-end smoke | PASS: 6/6 tasks |
| FFCBSPT/Kraul-GA end-to-end smoke | PASS: 4/4 tasks |
| Rebuild archived paper reports | PASS: Figure 5, Table 10, Figures 6–7 and five-method queue consistency checks |

Total: **108 tests passed, 3 skipped**. The three skipped baseline integration
tests require legacy J25/single-seed result sets intentionally excluded from
the multi-seed paper package; they do not represent skipped current formal
task validation.

The archived DBD rebuild reproduces IBS-CABC's mean time in system of
425.53 minutes and independently checks schedule-derived queue areas.
New smoke optimization outputs are kept in ignored `reruns/` and are not
published as formal experimental evidence.

This release did not rerun the entire multi-day optimization campaign. It
validated the full historical records and executed all three short experiment
pipelines. Linux CI is configured in `.github/workflows/reproduce.yml` and is
separate from the local Windows validation. The root README documents the
wall-clock reproducibility limits and missing historical fixed-scale checkpoints.

The first Linux CI run exposed a Windows-path parsing issue during historical
schedule replay. The report code now extracts saved instance filenames with
`PureWindowsPath`; the original and released hashes are recorded in
`validation/PORTABILITY_CHANGES.json`. Solver code and raw results remain
unchanged. This fix is covered by rerunning the report checks on both platforms.
