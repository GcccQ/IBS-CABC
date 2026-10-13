from __future__ import annotations

import numpy as np

from kraul_dbd_reproduction.convergence_merge import RunHistory, _incumbent_curve


def test_merged_curve_uses_best_seen_incumbent_at_endpoint() -> None:
    run = RunHistory(
        algorithm="Kraul-GA",
        instance="synthetic",
        instance_sha256="abc",
        jobs=25,
        seed=101,
        machines_stage1=3,
        machines_stage2=2,
        nominal_budget_seconds=10.0,
        evaluations=200,
        history=(100.0, 95.0),
        history_evaluations=(50.0, 150.0),
        final_incumbent=90.0,
        reported_objective=92.0,
        source_path="synthetic.jsonl",
    )
    curve = _incumbent_curve(run, np.asarray([0.0, 0.5, 0.8, 1.0]))
    assert np.allclose(curve, [100.0, 100.0, 95.0, 90.0])
    assert np.all(np.diff(curve) <= 0.0)
