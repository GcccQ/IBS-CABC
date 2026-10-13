from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


SOURCE = Path(sys.argv[1])
RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260831
SCALES = [100, 250, 500, 1000]


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    previous = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        previous = max(previous, candidate)
        adjusted[index] = previous
    return adjusted.tolist()


raw = pd.read_csv(SOURCE)
key = ["jobs", "instance_sha256", "algorithm", "seed"]
print(f"rows={len(raw)} unique_task_ids={raw['task_id'].nunique()} duplicate_keys={raw.duplicated(key).sum()}")

counts = (
    raw.groupby(["jobs", "algorithm"], sort=True)
    .size()
    .unstack("algorithm")
    .sort_index()
)
print("run counts by scale and algorithm")
print(counts.to_string())

pair_raw = raw[raw["algorithm"].isin(["ibscabc", "kraul_ga"])].copy()
seed_sets = pair_raw.groupby(["jobs", "instance_sha256", "algorithm"])["seed"].agg(lambda x: tuple(sorted(x)))
seed_wide = seed_sets.unstack("algorithm")
print(f"paired_instance_groups={len(seed_wide)} mismatched_seed_sets={(seed_wide['ibscabc'] != seed_wide['kraul_ga']).sum()}")

means = (
    pair_raw.groupby(["jobs", "instance", "instance_sha256", "algorithm"], sort=True)["objective"]
    .agg(["mean", "count", "sum"])
    .reset_index()
)
pivot = means.pivot(index=["jobs", "instance", "instance_sha256"], columns="algorithm", values=["mean", "count", "sum"])
pairs = pivot.reset_index()
pairs["difference"] = pairs[("sum", "kraul_ga")] / pairs[("count", "kraul_ga")] - pairs[("sum", "ibscabc")] / pairs[("count", "ibscabc")]
pairs["gain"] = 100.0 * pairs["difference"] / pairs[("mean", "kraul_ga")]
pairs = pairs.sort_values([("jobs", ""), ("instance_sha256", "")]).reset_index(drop=True)

rng = np.random.default_rng(BOOTSTRAP_SEED)
rows: list[dict[str, float | int | str]] = []

for label, subset in [(f"J{s}", pairs[pairs[("jobs", "")] == s]) for s in SCALES] + [("Overall", pairs)]:
    differences = subset["difference"].to_numpy(dtype=float)
    gains = subset["gain"].to_numpy(dtype=float)
    n = len(subset)
    if label == "Overall":
        sampled_parts = []
        for scale in SCALES:
            scale_gains = subset.loc[subset[("jobs", "")] == scale, "gain"].to_numpy(dtype=float)
            sampled_parts.append(scale_gains[rng.integers(0, len(scale_gains), size=(RESAMPLES, len(scale_gains)))])
        sampled = np.concatenate(sampled_parts, axis=1)
    else:
        sampled = gains[rng.integers(0, n, size=(RESAMPLES, n))]
    low, high = np.quantile(np.median(sampled, axis=1), [0.025, 0.975])
    p = float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
    nonzero = differences[differences != 0]
    ranks = rankdata(np.abs(nonzero), method="average")
    effect = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    wins = int((differences > 0).sum())
    ties = int((differences == 0).sum())
    losses = int((differences < 0).sum())
    rows.append(
        {
            "scale": label,
            "n": n,
            "median_gain": float(np.median(gains)),
            "ci_low": float(low),
            "ci_high": float(high),
            "wilcoxon_p": p,
            "rank_biserial": effect,
            "W_T_L": f"{wins}/{ties}/{losses}",
        }
    )

adjusted = holm([float(row["wilcoxon_p"]) for row in rows])
for row, p_adj in zip(rows, adjusted):
    row["holm_p"] = p_adj

result = pd.DataFrame(rows)[
    ["scale", "n", "median_gain", "ci_low", "ci_high", "wilcoxon_p", "holm_p", "rank_biserial", "W_T_L"]
]
print("independent recomputation")
print(result.to_string(index=False, float_format=lambda x: f"{x:.15g}"))

print("loss instances")
losses = pairs[pairs["difference"] < 0][
    [("jobs", ""), ("instance", ""), ("difference", ""), ("gain", "")]
]
print(losses.to_string(index=False, float_format=lambda x: f"{x:.12g}"))
