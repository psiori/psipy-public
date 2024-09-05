# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import numpy as np
import pytest

from psipy.core.np_utils import count_unique_values_per_row


def test_count_unique_values_per_row():
    arr = np.arange(100).reshape(10, 10)
    assert count_unique_values_per_row(arr).shape == (10,)
    assert np.all(count_unique_values_per_row(arr) == 10)
    arr = np.array([[1, 1, 3], [0, 0, 0], [1, 2, 3]])
    assert count_unique_values_per_row(arr).shape == (3,)
    assert count_unique_values_per_row(arr).tolist() == [2, 1, 3]
    with pytest.raises(ValueError):
        count_unique_values_per_row(np.random.rand(10, 10, 10))
    with pytest.raises(ValueError):
        count_unique_values_per_row(np.random.rand(10))
