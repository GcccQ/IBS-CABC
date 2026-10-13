from __future__ import annotations

from ..core.entities import ProblemInstance


def published_jobwise_lower_bound(instance: ProblemInstance) -> int:
    """Jobwise lower bound used by the reference paper.

    Batching and parallel-machine conflicts are relaxed:
    LB = sum_j(r_j + p_j1 + p_j2).

    In the released instance files, the two setup columns have not yet been
    folded into ``p_jk``. Section 3 states that setup is added to each job's
    processing time (Sule's rule), and the author's released notebook does
    exactly that before decoding. Consequently, ``duration_stage*``--raw
    processing plus the corresponding setup--is the paper-compatible ``p_jk``.
    """

    return sum(
        job.release + job.duration_stage1 + job.duration_stage2
        for job in instance.jobs
    )
