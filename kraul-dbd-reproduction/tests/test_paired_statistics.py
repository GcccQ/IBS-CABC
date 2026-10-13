from __future__ import annotations

import numpy as np

from kraul_dbd_reproduction.paired_statistics import holm_adjust, rank_biserial


def test_rank_biserial_sign_and_ties() -> None:
    assert rank_biserial([1.0, 2.0, 3.0]) == 1.0
    assert rank_biserial([-1.0, -2.0, -3.0]) == -1.0
    assert rank_biserial([0.0, 0.0]) == 0.0


def test_holm_adjustment_is_order_safe_and_monotone() -> None:
    raw = [0.04, 0.001, 0.02]
    adjusted = holm_adjust(raw)
    assert np.allclose(adjusted, [0.04, 0.003, 0.04])
