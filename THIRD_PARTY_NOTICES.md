# External benchmark and algorithm provenance

The MIT license covers the original code and documentation in this repository.
It does not relicense externally downloaded data, reference notebooks, or installed dependencies.

Benchmark instances and the reference notebook originate from Sebastian Kraul's
[public research repository](https://github.com/SebastianKraul/A-two-stage-hybrid-flow-shop-formulation-for-sterilization-processes-in-hospitals)
at commit `fc4e0c13013b100f068d338ae5b4c9bc0fe1c278`. The upstream repository did
not provide an explicit license when this release was prepared. These assets
are fetched directly from upstream by `tools/prepare_data.py`, kept outside
the Git index, and checked against the exact SHA-256 values of the paper inputs.
Consult the upstream author for permissions to redistribute those assets.

`kraul-dbd-reproduction` is the local reimplementation of the published FFCBSPT
and GA algorithms. The reference notebook is used for validation and provenance;
it is not included in this Git repository. The reproduction contract is in
`kraul-dbd-reproduction/docs/reproduction_contract.md`. The fixed experiment
bundle contains copies of the local solver implementations under `vendor/` so
that it uses the historical experiment versions.

Please cite Kraul's paper, *A two-stage hybrid flow-shop formulation for
sterilization processes in hospitals*, when using its benchmark or baselines.
Dependency packages, including OR-Tools, NumPy, SciPy and Matplotlib, retain
their respective upstream licenses.
