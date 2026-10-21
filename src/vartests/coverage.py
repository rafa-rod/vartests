# -*- coding: utf-8 -*-
"""Testes de cobertura incondicional: a taxa de violação é a nominal?

Servem para qualquer medida de risco baseada em quantil (VaR, DaR etc.):
recebem a série de violações (ou só a contagem x em n observações) e a
probabilidade de violação sob H0.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import beta, binom, chi2

from ._validation import _as_float_array, _as_violations, _check_level, _decision

ViolationsLike = Union[List[int], np.ndarray, pd.Series, pd.DataFrame]
ArrayLike = Union[List[float], np.ndarray, pd.Series, pd.DataFrame]

_DEFAULT_RISK_CONF_LEVEL = 0.99


# ==================== AUXILIARES ====================


def _resolve_risk_level(
    risk_conf_level: Optional[float], var_conf_level: Optional[float]
) -> float:
    """Aceita o nome antigo `var_conf_level`, com aviso, e devolve o nível validado."""
    if var_conf_level is not None:
        if risk_conf_level is not None:
            raise TypeError("Pass risk_conf_level or var_conf_level, not both.")
        warnings.warn(
            "var_conf_level is deprecated and will be removed; use risk_conf_level.",
            FutureWarning,
            stacklevel=3,
        )
        risk_conf_level = var_conf_level
    if risk_conf_level is None:
        risk_conf_level = _DEFAULT_RISK_CONF_LEVEL
    return _check_level(risk_conf_level, "risk_conf_level")


def _counts(
    violations: Optional[ViolationsLike], x: Optional[int], n: Optional[int]
) -> Tuple[int, int]:
    """Número de violações e de observações, a partir da série ou de x e n."""
    if violations is not None:
        if x is not None or n is not None:
            raise ValueError("Pass either violations or x and n, not both.")
        hits = _as_violations(violations)
        return int(hits.sum()), hits.size
    if x is None or n is None:
        raise ValueError("Pass violations, or both x and n.")
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer.")
    if int(x) != x or not 0 <= x <= n:
        raise ValueError("x must be an integer between 0 and n.")
    return int(x), int(n)


# ==================== TAXA DE FALHA ====================


def failure_rate(
    violations: Optional[ViolationsLike] = None,
    *,
    x: Optional[int] = None,
    n: Optional[int] = None,
) -> Dict:
    """Proporção de violações da medida de risco.

    Parameters:
        violations (array-like): série de violações (1 = violação, 0 = não violação)
        x, n (int):              alternativa à série: x violações em n observações
    Returns:
        answer (dict): número de violações, de observações e taxa de falha
    """
    n1, n = _counts(violations, x, n)
    return {"violations": n1, "observations": n, "failure rate": n1 / n}


def upper_confidence_limit(
    violations: Optional[ViolationsLike] = None,
    conf_level: float = 0.95,
    *,
    x: Optional[int] = None,
    n: Optional[int] = None,
) -> Dict:
    """Limites de confiança exatos (Clopper-Pearson) da taxa real de violação.

    Traduz o resultado do backtest sem exagero: com x violações em n
    observações, a taxa real é de no máximo "upper limit", com confiança
    `conf_level`. Unilateral:
        x < n:  Beta^-1(conf_level; x + 1, n - x);  x = n: 1.
    Com x = 0, fica 1 - (1 - conf_level)^(1/n).
    Também devolve o limite inferior unilateral e o intervalo bilateral.

    Parameters:
        violations (array-like): série de violações (1 = violação, 0 = não violação)
        conf_level (float):      nível de confiança dos limites
        x, n (int):              alternativa à série: x violações em n observações
    Returns:
        answer (dict): taxa observada, limite superior, limite inferior e intervalo bilateral
    """
    conf_level = _check_level(conf_level, "conf_level")
    x, n = _counts(violations, x, n)
    alpha = 1 - conf_level
    upper = 1.0 if x == n else float(beta.ppf(conf_level, x + 1, n - x))
    lower = 0.0 if x == 0 else float(beta.ppf(alpha, x, n - x + 1))
    interval = (
        0.0 if x == 0 else float(beta.ppf(alpha / 2, x, n - x + 1)),
        1.0 if x == n else float(beta.ppf(1 - alpha / 2, x + 1, n - x)),
    )
    return {
        "violations": x,
        "observations": n,
        "failure rate": x / n,
        "conf level": conf_level,
        "upper limit": upper,
        "lower limit": lower,
        "two-sided interval": interval,
    }


# ==================== KUPIEC ====================


def kupiec_test(
    violations: Optional[ViolationsLike] = None,
    risk_conf_level: Optional[float] = None,
    conf_level: float = 0.95,
    *,
    x: Optional[int] = None,
    n: Optional[int] = None,
    var_conf_level: Optional[float] = None,
) -> Dict:
    """Kupiec (1995), proporção de falhas (POF), bilateral.

    H0: a probabilidade de violação é 1 - risk_conf_level. LR ~ qui-quadrado(1).
    Com zero violações (ou só violações) o LR é finito, porque 0 * log(0) = 0:
    o teste pode não rejeitar, a depender de N e da taxa nominal
    (ex.: 0 violações em 250 dias a 99,9% não rejeita).

    Parameters:
        violations (array-like): série de violações (1 = violação, 0 = não violação)
        risk_conf_level (float): nível de confiança da medida de risco (padrão 0.99)
        conf_level (float):      nível de confiança do teste
        x, n (int):              alternativa à série: x violações em n observações
        var_conf_level (float):  nome antigo de risk_conf_level (descontinuado)
    Returns:
        answer (dict): estatística, valor crítico, p-valor e decisão
    """
    risk_conf_level = _resolve_risk_level(risk_conf_level, var_conf_level)
    conf_level = _check_level(conf_level, "conf_level")
    n1, n = _counts(violations, x, n)

    p = 1 - risk_conf_level
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

CriticalValue = Union[int, None, Tuple[Optional[int], Optional[int]]]


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
    As somas acumuladas são normalizadas pelo total, de modo que
    P(X >= 0) = P(X <= n) = 1 exatamente.
    """
    half = 0.5 * pmf if mid else 0.0
    upper_sum = np.cumsum(pmf[::-1])[::-1]  # soma a partir da cauda: preciso
    lower_sum = np.cumsum(pmf)
    upper = upper_sum / upper_sum[0] - half
    lower = lower_sum / lower_sum[-1] - half
    if alternative == "greater":
        pv = upper
    elif alternative == "less":
        pv = lower
    elif mid:
        pv = 2 * np.minimum(upper, lower)
    else:
        ordered = np.sort(pmf)
        cumulative = np.cumsum(ordered)
        cumulative = cumulative / cumulative[-1]
        position = np.searchsorted(ordered, pmf * (1 + 1e-7), side="right")
        pv = cumulative[position - 1]
    return np.clip(pv, 0.0, 1.0)


def _critical_value(reject: np.ndarray, alternative: str) -> CriticalValue:
    """Fronteira da região de rejeição.

    greater: menor número de violações que rejeita; less: maior número que
    rejeita; two-sided: (maior que rejeita abaixo, menor que rejeita acima).
    None quando não há como rejeitar naquele lado.
    """
    accepted = np.flatnonzero(~reject)
    below = np.flatnonzero(reject[: accepted.min()])
    above = np.flatnonzero(reject[accepted.max() + 1 :]) + accepted.max() + 1
    lower = int(below.max()) if below.size else None
    upper = int(above.min()) if above.size else None
    if alternative == "greater":
        return upper
    if alternative == "less":
        return lower
    return (lower, upper)


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
    reject = pv < significance
    accepted = np.flatnonzero(~reject)
    return {
        "null hypothesis": null_hypothesis,
        "alternative": alternative,
        "violations": x,
        "observations": pmf.size - 1,
        "expected violations": expected,
        "statistic": x,
        "p-value": float(pv[x]),
        "critical value": _critical_value(reject, alternative),
        "non-rejection region": (int(accepted.min()), int(accepted.max())),
        "size": float(pmf[reject].sum()),
        "decision": _decision(float(pv[x]), conf_level),
    }


def _binomial_pmf(n: int, p: float) -> np.ndarray:
    return binom.pmf(np.arange(n + 1), n, p)


def binomial_test(
    violations: Optional[ViolationsLike] = None,
    risk_conf_level: Optional[float] = None,
    conf_level: float = 0.95,
    alternative: str = "greater",
    *,
    x: Optional[int] = None,
    n: Optional[int] = None,
    var_conf_level: Optional[float] = None,
) -> Dict:
    """Teste binomial exato para o número de violações.

    H0: a probabilidade de violação é 1 - risk_conf_level. Vale para qualquer
    medida de risco baseada em quantil (VaR, DaR etc.). O padrão "greater" testa
    subestimação do risco (violações demais); use "two-sided" para também acusar
    conservadorismo. Por ser discreto, o tamanho efetivo ("size") fica abaixo do
    nominal; ver `mid_p_test`.

    Parameters:
        violations (array-like): série de violações (1 = violação, 0 = não violação)
        risk_conf_level (float): nível de confiança da medida de risco (padrão 0.99)
        conf_level (float):      nível de confiança do teste
        alternative (str):       "greater", "less" ou "two-sided"
        x, n (int):              alternativa à série: x violações em n observações
        var_conf_level (float):  nome antigo de risk_conf_level (descontinuado)
    Returns:
        answer (dict): contagem, p-valor, valor crítico, região de não rejeição,
        tamanho efetivo e decisão
    """
    risk_conf_level = _resolve_risk_level(risk_conf_level, var_conf_level)
    conf_level = _check_level(conf_level, "conf_level")
    alternative = _check_alternative(alternative)
    n1, n = _counts(violations, x, n)
    p = 1 - risk_conf_level
    return _exact_result(
        n1, _binomial_pmf(n, p), alternative, conf_level, False,
        f"Probability of failure is {round(p, 6)}", n * p,
    )


def mid_p_test(
    violations: Optional[ViolationsLike] = None,
    risk_conf_level: Optional[float] = None,
    conf_level: float = 0.95,
    alternative: str = "greater",
    *,
    x: Optional[int] = None,
    n: Optional[int] = None,
    var_conf_level: Optional[float] = None,
) -> Dict:
    """Teste binomial com mid-p (Lancaster, 1961).

    Igual ao `binomial_test`, mas a probabilidade do valor observado entra pela
    metade: P(X > x) + 0,5 P(X = x). Compensa a discretização e deixa o tamanho
    mais perto do nominal, sem garantia de não ultrapassá-lo. Use como
    complemento do teste exato.

    Parameters:
        violations (array-like): série de violações (1 = violação, 0 = não violação)
        risk_conf_level (float): nível de confiança da medida de risco (padrão 0.99)
        conf_level (float):      nível de confiança do teste
        alternative (str):       "greater", "less" ou "two-sided"
        x, n (int):              alternativa à série: x violações em n observações
        var_conf_level (float):  nome antigo de risk_conf_level (descontinuado)
    Returns:
        answer (dict): contagem, mid-p, valor crítico, região de não rejeição,
        tamanho efetivo e decisão
    """
    risk_conf_level = _resolve_risk_level(risk_conf_level, var_conf_level)
    conf_level = _check_level(conf_level, "conf_level")
    alternative = _check_alternative(alternative)
    n1, n = _counts(violations, x, n)
    p = 1 - risk_conf_level
    return _exact_result(
        n1, _binomial_pmf(n, p), alternative, conf_level, True,
        f"Probability of failure is {round(p, 6)}", n * p,
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
    violations: Optional[ViolationsLike] = None,
    probabilities: Optional[ArrayLike] = None,
    conf_level: float = 0.95,
    alternative: str = "greater",
    *,
    x: Optional[int] = None,
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
        x (int):                    alternativa à série: número de violações
                                    (n é o número de probabilidades)
    Returns:
        answer (dict): contagem, p-valor, valor crítico, região de não rejeição,
        tamanho efetivo e decisão
    """
    conf_level = _check_level(conf_level, "conf_level")
    alternative = _check_alternative(alternative)
    if probabilities is None:
        raise ValueError("probabilities is required.")
    pmf = poisson_binomial_pmf(probabilities)
    n_probs = pmf.size - 1
    if violations is not None:
        n1, n = _counts(violations, x, None)
        if n != n_probs:
            raise ValueError("violations and probabilities must have the same length.")
    else:
        n1, _ = _counts(None, x, n_probs)
    probs = _as_float_array(probabilities, "probabilities")
    return _exact_result(
        n1, pmf, alternative, conf_level, False,
        "Each observation fails with its own probability p_i", float(probs.sum()),
    )


def binomial_power(
    n: int,
    risk_conf_level: Optional[float] = None,
    true_failure_rate: Union[float, List[float]] = 0.02,
    conf_level: float = 0.95,
    alternative: str = "greater",
    mid_p: bool = False,
    *,
    var_conf_level: Optional[float] = None,
) -> Dict:
    """Poder do teste binomial (ou mid-p) com n observações.

    Probabilidade de rejeitar H0 quando a taxa real de violação é
    `true_failure_rate`. Ajuda a ler um "não rejeita": com poucas observações,
    o teste raramente detecta desvios moderados.

    As chaves de "power" são as taxas arredondadas em 10 casas decimais, para
    que power[0.075] funcione mesmo quando a taxa foi calculada (3 * 0.025).

    Parameters:
        n (int):                        número de observações
        risk_conf_level (float):        nível de confiança da medida de risco (padrão 0.99)
        true_failure_rate (float/list): taxa(s) real(is) de violação
        conf_level (float):             nível de confiança do teste
        alternative (str):              "greater", "less" ou "two-sided"
        mid_p (bool):                   usa a regra de decisão do mid-p
        var_conf_level (float):         nome antigo de risk_conf_level (descontinuado)
    Returns:
        answer (dict): valor crítico, região de não rejeição, tamanho efetivo e poder por taxa
    """
    risk_conf_level = _resolve_risk_level(risk_conf_level, var_conf_level)
    conf_level = _check_level(conf_level, "conf_level")
    alternative = _check_alternative(alternative)
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer.")
    n = int(n)
    pmf = _binomial_pmf(n, 1 - risk_conf_level)
    reject = _pvalues(pmf, alternative, mid_p) < 1 - conf_level
    accepted = np.flatnonzero(~reject)
    rates = np.atleast_1d(true_failure_rate).astype(float)
    if ((rates < 0) | (rates > 1)).any():
        raise ValueError("true_failure_rate must be between 0 and 1.")
    return {
        "critical value": _critical_value(reject, alternative),
        "non-rejection region": (int(accepted.min()), int(accepted.max())),
        "size": float(pmf[reject].sum()),
        "power": {
            round(float(r), 10): float(_binomial_pmf(n, r)[reject].sum()) for r in rates
        },
    }
