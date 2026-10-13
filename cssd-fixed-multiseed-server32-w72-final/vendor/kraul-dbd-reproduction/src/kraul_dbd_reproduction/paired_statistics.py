from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon


SCALES = (100, 250, 500, 1000)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_818


class PairedStatisticsError(ValueError):
    """Raised when the fixed-scale instance tables are not exactly pairable."""


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise PairedStatisticsError(f"empty instance table: {source}")
    return rows


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _instance_map(
    rows: Sequence[Mapping[str, str]], *, algorithm: str
) -> dict[tuple[int, str], dict[str, object]]:
    result: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        jobs = int(row["jobs"])
        if jobs not in SCALES:
            continue
        digest = str(row["instance_sha256"]).strip().lower()
        key = (jobs, digest)
        if key in result:
            raise PairedStatisticsError(f"duplicate {algorithm} instance: J{jobs}/{digest}")
        objective = float(row["objective"])
        if not np.isfinite(objective) or objective <= 0.0:
            raise PairedStatisticsError(f"invalid {algorithm} objective: J{jobs}/{digest}")
        if not _truthy(row["feasible"]):
            raise PairedStatisticsError(f"infeasible {algorithm} instance: J{jobs}/{digest}")
        result[key] = {
            "instance": row["instance"],
            "objective": objective,
            "seed_runs": int(row["seed_runs"]),
        }
    return result


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def rank_biserial(values: Sequence[float]) -> float:
    differences = np.asarray(values, dtype=float)
    nonzero = differences[differences != 0.0]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    return (positive - negative) / (positive + negative)


def _bootstrap_median_interval(
    values: np.ndarray, *, rng: np.random.Generator
) -> tuple[float, float]:
    sampled = values[
        rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    ]
    medians = np.median(sampled, axis=1)
    lower, upper = np.quantile(medians, (0.025, 0.975))
    return float(lower), float(upper)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_paired_statistics(
    *,
    ibscabc_instance_table: str | Path,
    kraul_instance_table: str | Path,
    output_dir: str | Path,
) -> Path:
    ibs_path = Path(ibscabc_instance_table).resolve()
    kraul_path = Path(kraul_instance_table).resolve()
    ibs = _instance_map(_read_csv(ibs_path), algorithm="IBS-CABC")
    kraul = _instance_map(_read_csv(kraul_path), algorithm="Kraul-GA")
    if set(ibs) != set(kraul):
        missing_kraul = sorted(set(ibs) - set(kraul))
        missing_ibs = sorted(set(kraul) - set(ibs))
        raise PairedStatisticsError(
            f"instance mismatch: missing Kraul-GA={len(missing_kraul)}, "
            f"missing IBS-CABC={len(missing_ibs)}"
        )
    counts = {jobs: sum(key[0] == jobs for key in ibs) for jobs in SCALES}
    if counts != {jobs: 20 for jobs in SCALES}:
        raise PairedStatisticsError(f"unexpected paired counts: {counts}")

    instance_rows: list[dict[str, object]] = []
    for jobs, digest in sorted(ibs):
        ibs_objective = float(ibs[(jobs, digest)]["objective"])
        kraul_objective = float(kraul[(jobs, digest)]["objective"])
        gain = 100.0 * (kraul_objective - ibs_objective) / kraul_objective
        instance_rows.append(
            {
                "scale": f"J{jobs}",
                "jobs": jobs,
                "instance_sha256": digest,
                "instance_ibscabc": ibs[(jobs, digest)]["instance"],
                "instance_kraul_ga": kraul[(jobs, digest)]["instance"],
                "ibscabc_seed_runs": ibs[(jobs, digest)]["seed_runs"],
                "kraul_ga_seed_runs": kraul[(jobs, digest)]["seed_runs"],
                "ibscabc_objective": ibs_objective,
                "kraul_ga_objective": kraul_objective,
                "gain_percent": gain,
                "outcome": "W" if gain > 0.0 else "L" if gain < 0.0 else "T",
            }
        )

    scopes: list[tuple[str, list[dict[str, object]]]] = [
        (f"J{jobs}", [row for row in instance_rows if int(row["jobs"]) == jobs])
        for jobs in SCALES
    ]
    scopes.append(("Overall", instance_rows))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for scale, members in scopes:
        gains = np.asarray([row["gain_percent"] for row in members], dtype=float)
        lower, upper = _bootstrap_median_interval(gains, rng=rng)
        nonzero = gains[gains != 0.0]
        p_value = (
            float(
                wilcoxon(
                    nonzero,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                ).pvalue
            )
            if len(nonzero)
            else 1.0
        )
        wins = int(np.sum(gains > 0.0))
        ties = int(np.sum(gains == 0.0))
        losses = int(np.sum(gains < 0.0))
        raw_p_values.append(p_value)
        summary_rows.append(
            {
                "scale": scale,
                "instances": len(members),
                "median_gain_percent": float(np.median(gains)),
                "bootstrap_95_ci_low_percent": lower,
                "bootstrap_95_ci_high_percent": upper,
                "wilcoxon_p_two_sided": p_value,
                "holm_p": 0.0,
                "rank_biserial": rank_biserial(gains),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wtl": f"{wins}/{ties}/{losses}",
            }
        )
    for row, adjusted in zip(summary_rows, holm_adjust(raw_p_values)):
        row["holm_p"] = adjusted

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "table8_ibscabc_vs_kraul_ga.csv", summary_rows)
    _write_csv(output / "table8_instance_level_gains.csv", instance_rows)
    audit = {
        "status": "PASS",
        "source_files": [str(ibs_path), str(kraul_path)],
        "paired_instances": len(instance_rows),
        "paired_counts": {str(key): value for key, value in counts.items()},
        "same_instance_sha256_set": True,
        "all_feasible": True,
        "statistical_unit": "instance",
        "seed_aggregation": "median objective within algorithm and instance (formula 4-2)",
        "observed_seed_runs": sorted(
            {
                int(row["ibscabc_seed_runs"])
                for row in instance_rows
            }
            | {int(row["kraul_ga_seed_runs"]) for row in instance_rows}
        ),
        "gain_formula": "100 * (Kraul-GA objective - IBS-CABC objective) / Kraul-GA objective",
        "positive_gain_favours": "IBS-CABC",
        "bootstrap": {
            "type": "paired percentile bootstrap of instance-level median Gain",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "wilcoxon": "two-sided signed-rank test on nonzero instance-level Gain",
        "rank_biserial": "signed-rank effect size on instance-level Gain; positive favours IBS-CABC",
        "holm_family": "the five displayed tests: J100, J250, J500, J1000, and Overall",
        "important_caveat": (
            "Only seed 101 is currently available, so formula (4-2) does not estimate "
            "within-instance stochastic variability; inference is across paired instances."
        ),
    }
    (output / "table8_statistics_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Table 8 paired IBS-CABC versus Kraul-GA statistics"
    )
    parser.add_argument("--ibscabc-instance-table", required=True)
    parser.add_argument("--kraul-instance-table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        build_paired_statistics(
            ibscabc_instance_table=args.ibscabc_instance_table,
            kraul_instance_table=args.kraul_instance_table,
            output_dir=args.output,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
