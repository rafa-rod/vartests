# -*- coding: utf-8 -*-
"""Testes sobre a distribuição prevista pelo modelo."""

import warnings
from typing import Dict, List, Union

import arch
import numpy as np
import pandas as pd
from scipy import optimize, stats
from tqdm import tqdm

from ._validation import _as_float_array, _check_level, _decision

SeriesLike = Union[List[float], np.ndarray, pd.Series, pd.DataFrame]


def zero_mean_test(data: SeriesLike, true_mu: float = 0, conf_level: float = 0.95) -> Dict:
    """Teste t bilateral para a média da distribuição.

    H0: média = true_mu. Premissa comum de VaR paramétricos (EWMA, GARCH).

    Parameters:
        data (array-like):  PnL ou retornos
        true_mu (float):    média sob H0
        conf_level (float): nível de confiança do teste
    Returns:
        answer (dict): estatística t, valor crítico, p-valor e decisão
    """
    conf_level = _check_level(conf_level, "conf_level")
    values = _as_float_array(data, "data")
    n = values.size
    if n < 2:
        raise ValueError("data: at least 2 observations are required.")

    result = stats.ttest_1samp(values, popmean=true_mu, alternative="two-sided")
    pvalue = float(result.pvalue)
    return {
        "null hypothesis": f"Mean of distribution = {true_mu}",
        "mean": float(values.mean()),
        "standard deviation": float(values.std(ddof=1)),
        "statistic": float(result.statistic),
        "critical value": float(stats.t.ppf(1 - (1 - conf_level) / 2, n - 1)),
        "p-value": pvalue,
        "decision": _decision(pvalue, conf_level),
    }


def garch_pit(
    pnl: SeriesLike, volatility_window: int = 252, verbose: bool = True
) -> pd.Series:
    """PIT de um GARCH(1,1) normal reestimado em janela móvel.

    Para cada t, ajusta o modelo nas `volatility_window` observações anteriores e
    calcula u_t = Phi((r_t - mu_t) / sigma_t) com a previsão de 1 passo. Não usa
    dados futuros. Use apenas se o seu modelo de VaR for um GARCH(1,1) normal;
    útil para outros testes de distribuição sobre o PIT.

    Parameters:
        pnl (array-like):        PnL ou retornos
        volatility_window (int): tamanho da janela de estimação
        verbose (bool):          mostra barra de progresso
    Returns:
        pit (Series): valores u_t em (0, 1), um para cada observação após a janela
    """
    values = _as_float_array(pnl, "pnl")
    index = pnl.index if isinstance(pnl, (pd.Series, pd.DataFrame)) else pd.RangeIndex(values.size)
    if values.size <= volatility_window:
        raise ValueError("pnl must be longer than volatility_window.")

    pit = np.empty(values.size - volatility_window)
    failures = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in tqdm(range(pit.size), disable=not verbose, desc="GARCH(1,1)"):
            window = values[t : volatility_window + t]
            # escala só da janela: o PIT não muda com a escala e o otimizador
            # do arch converge melhor com dados de ordem de grandeza 1
            scale = window.std(ddof=1)
            if not scale > 0:
                raise ValueError("pnl: a window has zero variance.")
            fit = arch.arch_model(
                window / scale, vol="GARCH", dist="normal", rescale=False
            ).fit(disp="off", show_warning=False)
            failures += fit.convergence_flag != 0
            forecast = fit.forecast(horizon=1, reindex=False)
            mu = forecast.mean.to_numpy()[-1, 0]
            sigma = np.sqrt(forecast.variance.to_numpy()[-1, 0])
            pit[t] = stats.norm.cdf((values[volatility_window + t] / scale - mu) / sigma)

    if failures:
        warnings.warn(
            f"{failures} of {pit.size} GARCH(1,1) fits did not converge.", RuntimeWarning,
            stacklevel=2,
        )
    return pd.Series(pit, index=index[volatility_window:], name="pit")


def _standardize_by_var(pnl: np.ndarray, var: np.ndarray, var_conf_level: float) -> np.ndarray:
    """z_t = PnL_t / sigma_t, com sigma_t = |VaR_t| / Phi^-1(var_conf_level) (VaR normal, média zero).

    Equivale a Phi^-1(u_t), em que u_t é o PIT; calcular direto evita perder
    precisão nas caudas.
    """
    if pnl.size != var.size:
        raise ValueError("pnl and var must have the same length.")
    if (var == 0).any():
        raise ValueError("var: values must be different from zero.")
    if (var > 0).any() and (var < 0).any():
        raise ValueError("var: all values must have the same sign (losses as positive or negative).")
    sigma = np.abs(var) / stats.norm.ppf(var_conf_level)
    return pnl / sigma


def berkowitz_tail_test(
    pnl: SeriesLike,
    var: SeriesLike,
    var_conf_level: float = 0.99,
    conf_level: float = 0.95,
) -> Dict:
    """Berkowitz (2001), teste de cauda (likelihood censurada), para VaR normal.

    Recebe o PnL realizado e o VaR previsto para cada dia. Supõe que o modelo
    de VaR é normal com média zero, de modo que o desvio previsto é
    sigma_t = |VaR_t| / Phi^-1(var_conf_level) e o PIT é u_t = Phi(PnL_t / sigma_t).
    Sob H0, z_t = Phi^-1(u_t) ~ N(0, 1). Só a cauda abaixo de
    Phi^-1(1 - var_conf_level), ou seja, as perdas além do VaR, é observada;
    o resto é censurado. LR ~ qui-quadrado(2) (média e desvio-padrão).

    O VaR pode vir positivo (perda) ou negativo (retorno), desde que todos os
    valores tenham o mesmo sinal.

    Parameters:
        pnl (array-like):       PnL ou retornos realizados
        var (array-like):       VaR previsto para cada dia, na mesma unidade do PnL
        var_conf_level (float): nível de confiança do VaR
        conf_level (float):     nível de confiança do teste
    Returns:
        answer (dict): parâmetros estimados, log-verossimilhanças, estatística, p-valor e decisão
    """
    var_conf_level = _check_level(var_conf_level, "var_conf_level")
    conf_level = _check_level(conf_level, "conf_level")
    z = _standardize_by_var(
        _as_float_array(pnl, "pnl"), _as_float_array(var, "var"), var_conf_level
    )
    cut = stats.norm.ppf(1 - var_conf_level)
    tail = z[z < cut]
    n_censored = z.size - tail.size

    def log_likelihood(mu: float, sigma: float) -> float:
        return float(
            np.sum(stats.norm.logpdf(tail, mu, sigma))
            + n_censored * stats.norm.logsf(cut, mu, sigma)
        )

    restricted = log_likelihood(0.0, 1.0)

    if tail.size == 0:
        # sem observações na cauda, o supremo da verossimilhança é 0 (mu -> -inf)
        mu_hat, sigma_hat, unrestricted = -np.inf, np.nan, 0.0
    else:
        def negative(x):
            value = log_likelihood(x[0], np.exp(x[1]))
            return 1e10 if not np.isfinite(value) else -value

        starts = [(0.0, 0.0), (float(z.mean()), float(np.log(max(z.std(), 1e-3))))]
        best = min(
            (optimize.minimize(negative, x0=s, method="Nelder-Mead",
                               options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
             for s in starts),
            key=lambda r: r.fun,
        )
        mu_hat, sigma_hat, unrestricted = float(best.x[0]), float(np.exp(best.x[1])), -best.fun
        if restricted > unrestricted:
            mu_hat, sigma_hat, unrestricted = 0.0, 1.0, restricted

    lr = max(2 * (unrestricted - restricted), 0.0)
    pvalue = float(stats.chi2.sf(lr, 2))

    return {
        "null hypothesis": "Losses beyond the VaR follow the normal tail implied by the VaR",
        "tail observations": int(tail.size),
        "mu": mu_hat,
        "sigma": sigma_hat,
        "unrestricted log-likelihood": unrestricted,
        "restricted log-likelihood": restricted,
        "statistic": lr,
        "critical value": float(stats.chi2.ppf(conf_level, 2)),
        "p-value": pvalue,
        "decision": _decision(pvalue, conf_level),
    }

