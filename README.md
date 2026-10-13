# IBS-CABC

Research implementation of **IBS-CABC: A Bottleneck-Aware Cooperative Artificial
Bee Colony Algorithm for Integrated Two-Stage Parallel-Batch Scheduling in
Central Sterile Supply Departments**.

This release contains the paper experiment versions of IBS-CABC, ALNS, TS,
FFCBSPT and Kraul-GA; experiment configurations and splits; historical run
records; and scripts to verify and rebuild the paper results. Original solver
files are preserved byte for byte. See [中文说明](README_zh.md),
[the code map](docs/CORE_CODE.md) and [the result map](docs/PAPER_RESULTS.md).

## Install

Use 64-bit **Python 3.10 or 3.11**. Run from the repository root:

```bash
git clone https://github.com/GcccQ/IBS-CABC.git
cd IBS-CABC
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. If PowerShell blocks
activation, run the following Python commands using `.\.venv\Scripts\python.exe`.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt -c requirements-constraints.txt
python -m pip install -e ./cssd-kg-edr -e ./cssd-comparison-baselines -e ./cssd-dbd-case-study -e ./kraul-dbd-reproduction
python -m pip check
python tools/prepare_data.py
python tools/prepare_results.py
```

`prepare_data.py` fetches benchmark archives and the reference notebook directly
from a pinned upstream commit and verifies all 547 required file hashes. It
requires internet access on first use. External inputs are not redistributed
under MIT; see [third-party notices](THIRD_PARTY_NOTICES.md).
`prepare_results.py` losslessly decompresses the three historical raw-result
archives and the archived job-level schedule CSV, and checks the original hashes.
Neither command overwrites changed inputs.

## Verify the archived paper evidence

```bash
python tools/verify_files.py
python tools/check_delivery.py --tests
python tools/rebuild_saved.py --target all
```

The checks verify 8,000 fixed-scale records, 3,731 daily DBD records and 120
CP-SAT ablation records, validate task contracts, independently recalculate
Tables 8 and 10, and run the existing tests for all five component projects.
Rebuilt Figure 5, Table 10 and Figures 6–7 are written to
`reruns/rebuilt_reports/`; historical records are preserved.
This validates archived evidence without repeating the full optimization campaign.

## Run small end-to-end experiments

```bash
python tools/run_experiment.py --experiment fixed --smoke
python tools/run_experiment.py --experiment dbd-three --smoke
python tools/run_experiment.py --experiment dbd-kraul --smoke
```

These are infrastructure checks with short budgets, not paper results.
Outputs go to `reruns/`. The evaluation-budget configuration
`cssd-kg-edr/configs/smoke.yaml` also provides a small core-algorithm example.

## Repeat the full experiments

Inspect the task contracts first:

```bash
python tools/run_experiment.py --experiment fixed --dry-run
python tools/run_experiment.py --experiment dbd-three --dry-run
python tools/run_experiment.py --experiment dbd-kraul --dry-run
```

Expected task counts: **8,000 / 2,730 / 1,001**. Start complete new runs with:

```bash
python tools/run_experiment.py --experiment fixed
python tools/run_experiment.py --experiment dbd-three
python tools/run_experiment.py --experiment dbd-kraul
python tools/run_experiment.py --experiment cp
```

The portable fixed-scale launcher uses **24 outer workers**; historical paper
results used **72 outer workers**, one numerical thread per task, on a Linux
node with 32 AMD EPYC 9654 cores and 60 GB memory. The exact historical profile
is `cssd-fixed-multiseed-server32-w72-final/configs/fixed_30_20_server32.yaml`:

```bash
cd cssd-fixed-multiseed-server32-w72-final
python run.py --config configs/fixed_30_20_server32.yaml --dry-run
python run.py --config configs/fixed_30_20_server32.yaml --output ../reruns/fixed_historical72
```

Historical fixed-scale execution took about 1,746.5 minutes. Full runs are
expensive; keep smoke outputs separate. Restart fixed/DBD experiments using
`tools/run_experiment.py --experiment <name> --resume <actual-run-directory>`.
For CP-SAT ablation resume, run `cssd-kg-edr/scripts/run_cp_sat_ablation.py`
directly with the same data/output/workers and omit `--new`.

## Reproducibility boundaries

- Algorithms use wall-clock budgets. Seeds, instances, parameters and dependency
  versions are specified, but hardware, OS and concurrent load can change the
  search trajectory. Re-running does not guarantee identical objective values.
- Kraul's released reference sorting uses NumPy's default unstable `argsort`.
  Equal-key order can vary with CPU SIMD dispatch, even at the same NumPy
  version. This release preserves that baseline behavior; FFCBSPT is
  deterministic within an environment, but its tied-input result is not a
  universal cross-machine constant. Exact decoder regression tests use a
  tie-free fixture; archived results remain the source for paper-value checks.
- Tables 4–8 can be independently recalculated from all 8,000 archived metrics.
  Historical per-task fixed-scale `checkpoints/` are absent from the local
  archive, so this release cannot recover every historical fixed-scale schedule.
  A new full run produces its own checkpoints.
- The Real365 case is the public seed 233–323 window: **91 days, 22,927 jobs**.
  It is a proxy, not the unpublished hospital dataset of 22,934 jobs.
  Each day is scheduled independently; machine occupancy is reset between days.
- Table 10 retains historical filenames `table9_*` in the report code.
- Earlier component README files describe additional development experiments;
  the root instructions and formal configurations identify this release's scope.

## License and citation

Original software is released under the [MIT license](LICENSE).
External data/notebooks and dependencies retain their own terms.
When reporting results, cite the IBS-CABC manuscript by its title above and
include the exact Git commit used. Publication metadata will be added when
available. Cite [Sebastian Kraul's benchmark repository](https://github.com/SebastianKraul/A-two-stage-hybrid-flow-shop-formulation-for-sterilization-processes-in-hospitals)
and accompanying paper when using its data or baselines.
