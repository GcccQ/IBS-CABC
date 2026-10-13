# Core code map

All paths below are relative to `cssd-kg-edr/src/cssd_ibscabc/`.

| Paper mechanism | Implementation |
|---|---|
| Integrated two-stage batch structure | `representation/batch_structure.py` |
| Unified decoder and machine dispatch | `decoder/integrated.py`, `decoder/dispatch.py`, `decoder/construction.py` |
| Cooperative bee colony and BFC/BSC roles | `algorithms/ibscabc.py` |
| Batch formation / sequencing moves | `algorithms/operators.py` |
| Bottleneck diagnosis and search allocation | `algorithms/bottleneck.py` |
| Quality/diversity archive and search budgets | `algorithms/common.py` |
| Seed portfolio | `algorithms/seeding.py` |
| Critical batches and interstage alignment | `algorithms/intensification.py` |
| Restricted family-decomposed CP-SAT repair | `algorithms/assignment_repair.py` |
| Objective and feasibility checks | `core/objective.py`, `core/feasibility.py` |
| Paper budget protocols | `experiments/protocols.py` |

ALNS and TS are implemented in `cssd-comparison-baselines/src/cssd_baselines/`.
The independent FFCBSPT/Kraul-GA reimplementation is in
`kraul-dbd-reproduction/src/kraul_dbd_reproduction/`.
DBD integration is in `cssd-dbd-case-study/src/cssd_dbd_case/`.

The fixed-scale runner preferentially loads historical copies under
`cssd-fixed-multiseed-server32-w72-final/vendor/`, rather than editable sibling
packages. Modify both deliberately if developing a new algorithm; do not mix
development versions with historical paper evidence.

`validation/SOURCE_COPY_MANIFEST.json` records the original hashes of the
unchanged copied code/configuration/result files. `SHA256SUMS.txt` covers the
complete published file set. External inputs have their own hash contract in
`validation/EXTERNAL_ASSETS.json`.
