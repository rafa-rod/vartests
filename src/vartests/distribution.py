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

_PIT_EPS = 1e-12


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
    caso contrário, informe ao `berkowitz_tail_test` o PIT do seu próprio modelo.

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in tqdm(range(pit.size), disable=not verbose, desc="GARCH(1,1)"):
            fit = arch.arch_model(
                values[t : volatility_window + t], vol="GARCH", dist="normal", rescale=False
            ).fit(disp="off")
            forecast = fit.forecast(horizon=1, reindex=False)
            mu = forecast.mean.to_numpy()[-1, 0]
            sigma = np.sqrt(forecast.variance.to_numpy()[-1, 0])
            pit[t] = stats.norm.cdf((values[volatility_window + t] - mu) / sigma)

    return pd.Series(pit, index=index[volatility_window:], name="pit")


def berkowitz_tail_test(
    pit: SeriesLike, var_conf_level: float = 0.99, conf_level: float = 0.95
) -> Dict:
    """Berkowitz (2001), teste de cauda (likelihood censurada).

    Recebe o PIT do modelo avaliado, u_t = F_t(r_t), em que F_t é a distribuição
    prevista pelo modelo de VaR para o dia t. Sob H0, z_t = Phi^-1(u_t) ~ N(0, 1).
    Só a cauda abaixo de Phi^-1(1 - var_conf_level) é observada; o resto é
    censurado. LR ~ qui-quadrado(2) (média e desvio-padrão).

    Parameters:
        pit (array-like):       valores u_t em (0, 1) previstos pelo modelo
        var_conf_level (float): nível de confiança do VaR
        conf_level (float):     nível de confiança do teste
    Returns:
        answer (dict): parâmetros estimados, log-verossimilhanças, estatística, p-valor e decisão
    """
    var_conf_level = _check_level(var_conf_level, "var_conf_level")
    conf_level = _check_level(conf_level, "conf_level")
    u = _as_float_array(pit, "pit")
    if ((u < 0) | (u > 1)).any():
        raise ValueError("pit: values must be between 0 and 1.")

    z = stats.norm.ppf(np.clip(u, _PIT_EPS, 1 - _PIT_EPS))
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
        "null hypothesis": "Tail of the PIT-transformed distribution is Normal(0, 1)",
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


def berkowtiz_tail_test(
    pnl: SeriesLike,
    volatility_window: int = 252,
    var_conf_level: float = 0.99,
    conf_level: float = 0.95,
    random_seed: int = None,
) -> Dict:
    """Descontinuada: use `berkowitz_tail_test(garch_pit(pnl), ...)`.

    Mantida por compatibilidade com versões <= 0.2.x. `random_seed` é ignorado.
    """
    warnings.warn(
        "berkowtiz_tail_test is deprecated and will be removed; use "
        "berkowitz_tail_test(garch_pit(pnl, volatility_window), ...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return berkowitz_tail_test(
        garch_pit(pnl, volatility_window), var_conf_level=var_conf_level, conf_level=conf_level
    )
