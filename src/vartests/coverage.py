# -*- coding: utf-8 -*-
"""Testes de cobertura incondicional: a taxa de violação é a nominal?

Servem para qualquer medida de risco baseada em quantil (VaR, DaR etc.):
recebem só a série de violações e a probabilidade de violação sob H0.
"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2

from ._validation import _as_float_array, _as_violations, _check_level, _decision

ViolationsLike = Union[List[int], np.ndarray, pd.Series, pd.DataFrame]
ArrayLike = Union[List[float], np.ndarray, pd.Series, pd.DataFrame]


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


# ==================== TESTES EXATOS ====================

_ALTERNATIVES = ("greater", "less", "two-sided")

def _check_alternative(alternative: str) -> str:
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}.")
    return alternative


def _pvalues(pmf: np.ndarray, alternative: str, mid: bool = False) -> np.ndarray:
    """p-valor de cada contagem x = 0..n a partir da distribuição sob H0.

    greater: P(X >= x); less: P(X <= x); two-sided: soma das probabilidades
    menores ou iguais a P(X = x), como no scipy.stats.binomtest.
    Com mid=True, a probabilidade do valor observado entra pela metade
    (bilateral: dobro do menor mid-p unilateral).
    """
    half = 0.5 * pmf if mid else 0.0
    upper = np.cumsum(pmf[::-1])[::-1] - half  # soma a partir da cauda: preciso
    lower = np.cumsum(pmf) - half
    if alternative == "greater":
        pv = upper
    elif alternative == "less":
        pv = lower
    elif mid:
        pv = 2 * np.minimum(upper, lower)
    else:
        ordered = np.sort(pmf)
        cumulative = np.cumsum(ordered)
        position = np.searchsorted(ordered, pmf * (1 + 1e-7), side="right")
        pv = cumulative[position - 1]
    return np.clip(pv, 0.0, 1.0)


def _exact_result(
    x: int,
    pmf: np.ndarray,
    alternative: str,
    conf_level: float,
    mid: bool,
    null_hypothesis: str,
    expected: float,
) -> Dict:
    pv = _pvalues(pmf, alternative, mid)
    significance = 1 - conf_level
    accepted = np.flatnonzero(pv >= significance)
    return {
        "null hypothesis": null_hypothesis,
        "alternative": alternative,
        "violations": x,
        "observations": pmf.size - 1,
        "expected violations": expected,
        "statistic": x,
        "p-value": float(pv[x]),
        "non-rejection region": (int(accepted.min()), int(accepted.max())),
        "size": float(pmf[pv < significance].sum()),
        "decision": _decision(float(pv[x]), conf_level),
    }


def _binomial_pmf(n: int, p: float) -> np.ndarray:
    return binom.pmf(np.arange(n + 1), n, p)


def binomial_test(
    violations: ViolationsLike,
    var_conf_level: float = 0.99,
    conf_level: float = 0.95,
    alternative: str = "greater",
) -> Dict:
    """Teste binomial exato para o número de violações.

    H0: a probabilidade de violação é 1 - var_conf_level. Vale para qualquer medida
    de risco baseada em quantil (VaR, DaR etc.). O padrão "greater" testa
    subestimação do risco (violações demais); use "two-sided" para também acusar
    conservadorismo. Por ser discreto, o tamanho efetivo ("size") fica abaixo do
    nominal; ver `mid_p_test`.

    Parameters:
        violations (array-like): série de violações (1 = violação, 0 = não violação)
        var_conf_level (float):  nível de confiança da medida de risco
        conf_level (float):      nível de confiança do teste
        alternative (str):       "greater", "less" ou "two-sided"
    Returns:
        answer (dict): contagem, p-valor, região de não rejeição, tamanho efetivo e decisão
    """
    var_conf_level = _check_level(var_conf_level, "var_conf_level")
    conf_level = _check_level(conf_level, "conf_level")
    alternative = _check_alternative(alternative)
    hits = _as_violations(violations)
    p = 1 - var_conf_level
    return _exact_result(
        int(hits.sum()), _binomial_pmf(hits.size, p), alternative, conf_level, False,
        f"Probability of failure is {round(p, 6)}", hits.size * p,
    )


def mid_p_test(
    violations: ViolationsLike,
    var_conf_level: float = 0.99,
    conf_level: float = 0.95,
    alternative: str = "greater",
) -> Dict:
    """Teste binomial com mid-p (Lancaster, 1961).

    Igual ao `binomial_test`, mas a probabilidade do valor observado entra pela
    metade: P(X > x) + 0,5 P(X = x). Compensa a discretização e deixa o tamanho
    mais perto do nominal, sem garantia de não ultrapassá-lo. Use como
    complemento do teste exato.

    Parameters:
        violations (array-like): série de violações (1 = violação, 0 = não violação)
        var_conf_level (float):  nível de confiança da medida de risco
        conf_level (float):      nível de confiança do teste
        alternative (str):       "greater", "less" ou "two-sided"
    Returns:
        answer (dict): contagem, mid-p, região de não rejeição, tamanho efetivo e decisão
    """
    var_conf_level = _check_level(var_conf_level, "var_conf_level")
    conf_level = _check_level(conf_level, "conf_level")
    alternative = _check_alternative(alternative)
    hits = _as_violations(violations)
    p = 1 - var_conf_level
    return _exact_result(
        int(hits.sum()), _binomial_pmf(hits.size, p), alternative, conf_level, True,
        f"Probability of failure is {round(p, 6)}", hits.size * p,
    )


def poisson_binomial_pmf(probabilities: ArrayLike) -> np.ndarray:
    """Distribuição do número de sucessos em ensaios independentes com
    probabilidades diferentes, por convolução sucessiva (exata, O(n^2))."""
    probs = _as_float_array(probabilities, "probabilities")
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError("probabilities: values must be between 0 and 1.")
    pmf = np.zeros(probs.size + 1)
    pmf[0] = 1.0
    for k, p in enumerate(probs, start=1):
        pmf[1 : k + 1] = pmf[1 : k + 1] * (1 - p) + pmf[:k] * p
        pmf[0] *= 1 - p
    return pmf


def poisson_binomial_test(
    violations: ViolationsLike,
    probabilities: ArrayLike,
    conf_level: float = 0.95,
    alternative: str = "greater",
) -> Dict:
    """Teste exato com probabilidade de violação diferente em cada observação.

    H0: a observação i viola com probabilidade p_i, de forma independente; o
    total segue uma Poisson-binomial. Útil quando a cobertura esperada varia,
    como no DaR por episódios (p_i do estimador em cada pico) ou num VaR com
    nível variável. Com todos os p_i iguais, coincide com o `binomial_test`.

    Parameters:
        violations (array-like):    série de violações (1 = violação, 0 = não violação)
        probabilities (array-like): probabilidade de violação de cada observação
        conf_level (float):         nível de confiança do teste
        alternative (str):          "greater", "less" ou "two-sided"
    Returns:
        answer (dict): contagem, p-valor, região de não rejeição, tamanho efetivo e decisão
    """
    conf_level = _check_level(conf_level, "conf_level")
    alternative = _check_alternative(alternative)
    hits = _as_violations(violations)
    pmf = poisson_binomial_pmf(probabilities)
    if pmf.size - 1 != hits.size:
        raise ValueError("violations and probabilities must have the same length.")
    probs = _as_float_array(probabilities, "probabilities")
    return _exact_result(
        int(hits.sum()), pmf, alternative, conf_level, False,
        "Each observation fails with its own probability p_i", float(probs.sum()),
    )


def binomial_power(
    n: int,
    var_conf_level: float = 0.99,
    true_failure_rate: Union[float, List[float]] = 0.02,
    conf_level: float = 0.95,
    alternative: str = "greater",
    mid_p: bool = False,
) -> Dict:
    """Poder do teste binomial (ou mid-p) com n observações.

    Probabilidade de rejeitar H0 quando a taxa real de violação é
    `true_failure_rate`. Ajuda a ler um "não rejeita": com poucas observações,
    o teste raramente detecta desvios moderados.

    Parameters:
        n (int):                        número de observações
        var_conf_level (float):         nível de confiança da medida de risco
        true_failure_rate (float/list): taxa(s) real(is) de violação
        conf_level (float):             nível de confiança do teste
        alternative (str):              "greater", "less" ou "two-sided"
        mid_p (bool):                   usa a regra de decisão do mid-p
    Returns:
        answer (dict): região de não rejeição, tamanho efetivo e poder por taxa
    """
    var_conf_level = _check_level(var_conf_level, "var_conf_level")
    conf_level = _check_level(conf_level, "conf_level")
    alternative = _check_alternative(alternative)
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer.")
    n = int(n)
    pmf = _binomial_pmf(n, 1 - var_conf_level)
    reject = _pvalues(pmf, alternative, mid_p) < 1 - conf_level
    accepted = np.flatnonzero(~reject)
    rates = np.atleast_1d(true_failure_rate).astype(float)
    if ((rates < 0) | (rates > 1)).any():
        raise ValueError("true_failure_rate must be between 0 and 1.")
    return {
        "non-rejection region": (int(accepted.min()), int(accepted.max())),
        "size": float(pmf[reject].sum()),
        "power": {float(r): float(_binomial_pmf(n, r)[reject].sum()) for r in rates},
    }
