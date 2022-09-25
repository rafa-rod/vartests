# -*- coding: utf-8 -*-

import random
from typing import List
import pandas as pd

import numpy as np
import sys

sys.path.append("./vartests/")
sys.path.append("./vartests/vartests")

from vartests import duration_test

def get_violations(repeat: int = 10) -> List[int]:
    return [random.randint(0, 1) for _ in range(repeat)]

class TestClass():
            
    def test_duration(self):
        violations = [1,1,1,1,1]
        self.conf_level = 0.95
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [10.] and result["decision"] == "Reject H0"
        
        violations = [0,0,0,0,0]
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [2.] and result["decision"] == "Fail to Reject H0"
        
        violations = [1,0,0,0,0]
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [2.] and result["decision"] == "Fail to Reject H0"
        
        violations = [0,0,0,0,1]
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [2.] and result["decision"] == "Fail to Reject H0"
        
        violations = [0,1,0,1,0]
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [10.] and result["decision"] == "Reject H0"
        
        violations = np.array([0,1,0,1,0]) #ok
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [10.] and result["decision"] == "Reject H0"
        
        violations = np.array([[0,1,0,1,0]]) #ok
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [10.] and result["decision"] == "Reject H0"
        
        violations = pd.Series([0,1,0,1,0]) #ok
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [10.] and result["decision"] == "Reject H0"
        
        violations = pd.Series(np.array([0,1,0,1,0])) #ok
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [10.] and result["decision"] == "Reject H0"
        
        violations = pd.DataFrame(np.array([0,1,0,1,0])) #ok
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [10.] and result["decision"] == "Reject H0"

        violations = pd.DataFrame(np.array([[0,1,0,1,0]]))
        result = duration_test(violations, self.conf_level)
        assert result["weibull exponential"] == [10.] and result["decision"] == "Reject H0"
        
        for _ in range(30):
            t = get_violations(200)
            t[0] = random.randint(0, 1)
            t[-1] = random.randint(0, 1)
            _ = duration_test(t)
            
TestClass()