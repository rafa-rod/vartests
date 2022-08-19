import random
from typing import List
import pandas as pd

import numpy as np
import sys, os, pytest

path = "../vartests/"
sys.path.append(path)

from vartests import duration_test


def get_violations(repeat: int = 10) -> List:
    return [random.randint(0, 1) for _ in range(repeat)]


@pytest.fixture
def violations():
    res = []
    for _ in range(10):
        t = get_violations(100)
        t[0] = random.randint(0, 1)
        t[-1] = random.randint(0, 1)
        res.append(t)
    return res


class TestClass:
    def test_detecttrend(self, violations):
        self.conf_level = 0.95
        for i, violate in enumerate(violations):
            duration_test(violate, self.conf_level)

