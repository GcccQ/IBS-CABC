from __future__ import annotations

import hashlib
from typing import Sequence


def batch_signature(stage: int, family: int, job_ids: Sequence[int]) -> str:
    """Return the order-invariant identity of one same-family batch."""

    payload = f"{stage}|{family}|" + ",".join(map(str, sorted(job_ids)))
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=10).hexdigest()
    return f"s{stage}f{family}-{digest}"
