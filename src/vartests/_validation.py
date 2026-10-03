# -*- coding: utf-8 -*-
"""Validação e conversão de entradas compartilhadas pelos testes."""

from typing import Any

import numpy as np
import pandas as pd

REJECT = "Reject H0"
FAIL_TO_REJECT = "Fail to reject H0"
NOT_ENOUGH_DATA = "Not enough data"


def _to_1d_array(data: Any, name: str) -> np.ndarray:
    """Converte list, tuple, array, Series ou DataFrame de uma coluna em array 1-D."""
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1 and data.shape[0] != 1:
            raise ValueError(f"{name}: DataFrame must have a single column.")
        values = data.to_numpy().ravel()
    elif isinstance(data, (pd.Series, np.ndarray, list, tuple)):
        values = np.asarray(data)
        if values.ndim > 1:
            if sum(s > 1 for s in values.shape) > 1:
                raise ValueError(f"{name}: input must be one-dimensional.")
            values = values.ravel()
    else:
        raise TypeError(f"{name}: input must be list, tuple, array, Series or DataFrame.")
    if values.size == 0:
        raise ValueError(f"{name}: input is empty.")
    return values


def _as_violations(violations: Any) -> np.ndarray:
    """Série de violações como array de inteiros 0/1."""
    values = _to_1d_array(violations, "violations")
    if values.dtype == bool:
        return values.astype(int)
    values = values.astype(float)
    if np.isnan(values).any():
        raise ValueError("violations: input contains NaN.")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("violations: values must be 0 or 1.")
    return values.astype(int)


def _as_float_array(data: Any, name: str) -> np.ndarray:
    """Série numérica (PnL, retornos, PIT) como array float sem NaN."""
    values = _to_1d_array(data, name).astype(float)
    if np.isnan(values).any():
        raise ValueError(f"{name}: input contains NaN.")
    return values


def _check_level(level: float, name: str) -> float:
    """Nível de confiança estritamente entre 0 e 1."""
    if not 0 < level < 1:
        raise ValueError(f"{name} must be between 0 and 1 (exclusive).")
    return float(level)


def _decision(pvalue: float, conf_level: float) -> str:
    if np.isnan(pvalue):
        return NOT_ENOUGH_DATA
    return REJECT if pvalue < 1 - conf_level else FAIL_TO_REJECT
