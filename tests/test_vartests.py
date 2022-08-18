import random
from typing import List
import pandas as pd

import numpy as np
import sys, os, pytest

path = "../vartests/"
sys.path.append(path)

from vartests import duration_test

def get_violations(repeat: int = 10) -> List:
    return [random.randint(0, 1) for x in range(repeat)]

class TestClass():

      def __init__(self):
        self.conf_level = 0.95

      def test_detecttrend(self):
        self.violations = get_violations(142)
        duration_test(self.violations , self.conf_level)
        self.violations = pd.DataFrame(np.array([0]*142))
        duration_test(self.violations , self.conf_level)
        self.violations = pd.DataFrame(np.array([1]+[0]*141))
        duration_test(self.violations , self.conf_level)
        self.violations = pd.DataFrame(np.array([1]+[0]*140+[1]))
        duration_test(self.violations , self.conf_level)
        self.violations = pd.DataFrame(np.array([0]*141+[1]))
        duration_test(self.violations , self.conf_level)
        self.violations = np.array([0]*141+[1])
        duration_test(self.violations , self.conf_level)

TestClass()