# -*- coding: utf-8 -*-
"""Testes de independência das violações."""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import chi2

from ._validation import _as_violations, _check_level, _decision

ViolationsLike = Union[List[int], np.ndarray, pd.Series, pd.DataFrame]

_SHAPE_BOUNDS = (0.001, 10.0)


def _durations(hits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Durações entre violações e indicadores de censura (C = 1 censurada).

    A primeira duração é censurada se a série não começa com violação;
    a última, se não termina com violação (Christoffersen e Pelletier, 2004).
    """
    hit_days = np.flatnonzero(hits == 1) + 1  # posições 1-based
    D = np.diff(hit_days).astype(float)
    C = np.zeros(D.size)
    if hits[0] == 0:
        C = np.r_[1.0, C]
        D = np.r_[float(hit_days[0]), D]
    if hits[-1] == 0:
        C = np.r_[C, 1.0]
        D = np.r_[D, float(hits.size - hit_days[-1])]
    return D, C


def _weibull_log_likelihood(b: float, D: np.ndarray, C: np.ndarray) -> float:
    """Log-verossimilhança Weibull com escala 'a' concentrada (perfilada).

    Com b = 1 a Weibull vira a exponencial (sem memória), que é a H0.
    """
    N = D.size
    a = ((N - C[0] - C[-1]) / np.sum(D**b)) ** (1 / b)

    def log_pdf(d):
        return b * np.log(a) + np.log(b) + (b - 1) * np.log(d) - (a * d) ** b

    def log_survival(d):
        return -((a * d) ** b)

    first = C[0] * log_survival(D[0]) + (1 - C[0]) * log_pdf(D[0])
    last = C[-1] * log_survival(D[-1]) + (1 - C[-1]) * log_pdf(D[-1])
    middle = np.sum(log_pdf(D[1:-1])) if N > 2 else 0.0
    return float(first + middle + last)


def duration_test(violations: ViolationsLike, conf_level: float = 0.95) -> Dict:
    """Christoffersen e Pelletier (2004), teste de duração.

    Verifica se o tempo entre violações não tem memória: H0 é duração exponencial
    (Weibull com b = 1). b < 1 indica violações agrupadas. LR ~ qui-quadrado(1).
    Porte do VaRDurTest (pacote rugarch, R).

    Parameters:
        violations (array-like): série de violações (1 = violação, 0 = não violação)
        conf_level (float):      nível de confiança do teste
    Returns:
        answer (dict): parâmetro b, log-verossimilhanças, estatística, p-valor e decisão
    """
    conf_level = _check_level(conf_level, "conf_level")
    hits = _as_violations(violations)
    H0 = "Duration between exceedances has no memory (Weibull b = 1, exponential)"
    critical = float(chi2.ppf(conf_level, 1))

    if hits.sum() == 0:
        D, C = np.array([]), np.array([])
    else:
        D, C = _durations(hits)

    # são necessárias ao menos 2 durações e 1 delas não censurada
    if D.size < 2 or (D.size - C[0] - C[-1]) < 1:
        return {
            "null hypothesis": H0,
            "weibull shape (b)": np.nan,
            "unrestricted log-likelihood": np.nan,
            "restricted log-likelihood": np.nan,
            "statistic": np.nan,
            "critical value": critical,
            "p-value": np.nan,
            "decision": _decision(np.nan, conf_level),
        }

    def negative(b):
        value = _weibull_log_likelihood(b, D, C)
        return 1e10 if not np.isfinite(value) else -value

    solution = optimize.minimize_scalar(negative, bounds=_SHAPE_BOUNDS, method="bounded")
    b = float(solution.x)
    unrestricted = -negative(b)
    restricted = -negative(1.0)
    if restricted > unrestricted:  # o otimizador não pode ficar pior que b = 1
        b, unrestricted = 1.0, restricted
    lr = max(2 * (unrestricted - restricted), 0.0)
    pvalue = float(chi2.sf(lr, 1))

    return {
        "null hypothesis": H0,
        "weibull shape (b)": b,
        "unrestricted log-likelihood": unrestricted,
        "restricted log-likelihood": restricted,
        "statistic": lr,
        "critical value": critical,
        "p-value": pvalue,
        "decision": _decision(pvalue, conf_level),
    }
