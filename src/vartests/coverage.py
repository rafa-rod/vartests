# -*- coding: utf-8 -*-
"""Testes de cobertura incondicional: a taxa de violação é a nominal?"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from ._validation import _as_violations, _check_level, _decision

ViolationsLike = Union[List[int], np.ndarray, pd.Series, pd.DataFrame]


def failure_rate(violations: ViolationsLike) -> Dict:
    """Proporção de violações do VaR.

    Parameters:
        violations (array-like): série de violações (1 = violação, 0 = não violação)
    Returns:
        answer (dict): número de violações, de observações e taxa de falha
    """
    hits = _as_violations(violations)
    n1, n = int(hits.sum()), hits.size
    return {"violations": n1, "observations": n, "failure rate": n1 / n}


def kupiec_test(
    violations: ViolationsLike,
    var_conf_level: float = 0.99,
    conf_level: float = 0.95,
) -> Dict:
    """Kupiec (1995), proporção de falhas (POF), bilateral.

    H0: a probabilidade de violação é 1 - var_conf_level. LR ~ qui-quadrado(1).
    Com zero violações (ou só violações) o LR é finito, porque 0 * log(0) = 0:
    o teste pode não rejeitar, a depender de N e da taxa nominal
    (ex.: 0 violações em 250 dias a 99,9% não rejeita).

    Parameters:
        violations (array-like): série de violações (1 = violação, 0 = não violação)
        var_conf_level (float):  nível de confiança do VaR
        conf_level (float):      nível de confiança do teste
    Returns:
        answer (dict): estatística, valor crítico, p-valor e decisão
    """
    var_conf_level = _check_level(var_conf_level, "var_conf_level")
    conf_level = _check_level(conf_level, "conf_level")
    hits = _as_violations(violations)

    p = 1 - var_conf_level
    n = hits.size
    n1 = int(hits.sum())
    n0 = n - n1
    pi_obs = n1 / n

    log_lik_h0 = xlogy(n1, p) + xlogy(n0, 1 - p)
    log_lik_h1 = xlogy(n1, pi_obs) + xlogy(n0, 1 - pi_obs)
    lr = max(-2 * (log_lik_h0 - log_lik_h1), 0.0)
    pvalue = float(chi2.sf(lr, 1))

    return {
        "null hypothesis": f"Probability of failure is {round(p, 6)}",
        "violations": n1,
        "expected violations": n * p,
        "statistic": float(lr),
        "critical value": float(chi2.ppf(conf_level, 1)),
        "p-value": pvalue,
        "decision": _decision(pvalue, conf_level),
    }
