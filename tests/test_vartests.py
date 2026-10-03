# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import vartests
from vartests import (
    berkowitz_tail_test,
    binomial_power,
    binomial_test,
    mid_p_test,
    poisson_binomial_pmf,
    poisson_binomial_test,
    berkowtiz_tail_test,
    duration_test,
    failure_rate,
    garch_pit,
    kupiec_test,
    zero_mean_test,
)
from vartests.independence import _durations


def _hits(x, n):
    """x violações em n observações."""
    return [1] * x + [0] * (n - x)


def _clustered_violations(rng, T=1000, p01=0.012, p11=0.3):
    """Cadeia de Markov: violações tendem a vir em sequência."""
    x = np.zeros(T, dtype=int)
    for i in range(1, T):
        x[i] = rng.random() < (p11 if x[i - 1] else p01)
    return x


class TestClass:
    def setup_method(self):
        """Setup configurations executed before each test"""
        self.rng = np.random.default_rng(42)
        self.conf_level = 0.95
        self.hits = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]

    # ==================== INPUT VALIDATION ====================
    def test_input_types(self):
        expected = kupiec_test(self.hits, 0.9)
        inputs = [
            np.array(self.hits),
            np.array([self.hits]),
            pd.Series(self.hits),
            pd.DataFrame(self.hits),
            pd.DataFrame([self.hits]),
            np.array(self.hits, dtype=bool),
            tuple(self.hits),
        ]
        for data in inputs:
            assert kupiec_test(data, 0.9) == expected
            assert duration_test(data) == duration_test(self.hits)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            kupiec_test([0, 1, 2])
        with pytest.raises(ValueError):
            kupiec_test([0, np.nan, 1])
        with pytest.raises(ValueError):
            kupiec_test([])
        with pytest.raises(ValueError):
            kupiec_test(pd.DataFrame(np.zeros((5, 2))))
        with pytest.raises(TypeError):
            kupiec_test("0101")
        with pytest.raises(ValueError):
            kupiec_test(self.hits, var_conf_level=1.0)
        with pytest.raises(ValueError):
            berkowitz_tail_test([0.1, 1.2])

    # ==================== FAILURE RATE ====================
    def test_failure_rate(self):
        result = failure_rate(pd.DataFrame(self.hits))
        assert result == {"violations": 3, "observations": 10, "failure rate": 0.3}

    # ==================== KUPIEC ====================
    @pytest.mark.parametrize(
        "n, var_conf_level, lower, upper",
        [  # Tabela 10.1 de Giambiagi (cap. 10): intervalos de não rejeição a 5%
            (250, 0.99, 1, 6),
            (250, 0.999, 0, 1),
            (1000, 0.999, 0, 3),
            (2000, 0.99, 12, 29),
        ],
    )
    def test_kupiec_giambiagi_table(self, n, var_conf_level, lower, upper):
        accepted = [
            x
            for x in range(n + 1)
            if kupiec_test([1] * x + [0] * (n - x), var_conf_level)["decision"]
            == "Fail to reject H0"
        ]
        assert (accepted[0], accepted[-1]) == (lower, upper)

    def test_kupiec_zero_violations(self):
        result = kupiec_test([0] * 100, var_conf_level=0.99)
        assert result["statistic"] == pytest.approx(-2 * 100 * np.log(0.99))
        assert result["decision"] == "Fail to reject H0"
        assert kupiec_test([0] * 250, var_conf_level=0.99)["decision"] == "Reject H0"
        assert kupiec_test([1] * 10, var_conf_level=0.99)["decision"] == "Reject H0"

    # ==================== BINOMIAL / MID-P ====================
    @pytest.mark.parametrize(
        "x, binomial, mid_p",
        [  # 60 episódios a 97,5%: P(X >= x) e mid-p
            (0, 1.000, 0.891),
            (1, 0.781, 0.613),
            (2, 0.444, 0.317),
            (3, 0.190, 0.126),
            (4, 0.063, 0.040),
            (5, 0.017, 0.010),
        ],
    )
    def test_binomial_and_mid_p_table(self, x, binomial, mid_p):
        assert binomial_test(_hits(x, 60), 0.975)["p-value"] == pytest.approx(binomial, abs=5e-4)
        assert mid_p_test(_hits(x, 60), 0.975)["p-value"] == pytest.approx(mid_p, abs=5e-4)

    def test_binomial_matches_scipy(self):
        for _ in range(200):
            n = int(self.rng.integers(5, 400))
            p = float(self.rng.uniform(0.001, 0.3))
            x = int(self.rng.integers(0, n + 1))
            for alternative in ("greater", "less", "two-sided"):
                result = binomial_test(_hits(x, n), 1 - p, alternative=alternative)
                expected = stats.binomtest(x, n, p, alternative=alternative).pvalue
                assert result["p-value"] == pytest.approx(expected, rel=1e-9, abs=1e-12)

    def test_binomial_size_and_region(self):
        result = binomial_test(_hits(4, 58), 0.975)
        assert result["p-value"] == pytest.approx(0.057, abs=5e-4)
        assert result["size"] == pytest.approx(0.0149, abs=5e-5)
        assert result["non-rejection region"] == (0, 4)
        assert result["decision"] == "Fail to reject H0"
        assert mid_p_test(_hits(4, 58), 0.975)["decision"] == "Reject H0"
        # o tamanho efetivo nunca passa do nominal no teste exato
        for n in (20, 60, 250, 1000):
            assert binomial_test(_hits(0, n), 0.99)["size"] <= 0.05

    def test_binomial_alternatives(self):
        hits = _hits(0, 500)  # zero violações a 99%: conservador
        assert binomial_test(hits, 0.99)["decision"] == "Fail to reject H0"
        assert binomial_test(hits, 0.99, alternative="less")["decision"] == "Reject H0"
        assert binomial_test(hits, 0.99, alternative="two-sided")["decision"] == "Reject H0"
        with pytest.raises(ValueError):
            binomial_test(hits, 0.99, alternative="bigger")

    def test_binomial_power(self):
        result = binomial_power(60, 0.975, [0.05, 0.075, 0.10])
        assert result["non-rejection region"] == (0, 4)
        assert result["size"] == pytest.approx(0.0171, abs=5e-5)
        powers = [result["power"][r] for r in (0.05, 0.075, 0.10)]
        assert powers == pytest.approx([0.18, 0.47, 0.73], abs=5e-3)
        assert binomial_power(60, 0.975, 0.025)["power"][0.025] == pytest.approx(result["size"])
        assert binomial_power(60, 0.975, mid_p=True)["size"] > result["size"]

    # ==================== POISSON-BINOMIAL ====================
    def test_poisson_binomial_pmf(self):
        probs = self.rng.uniform(0, 0.2, 300)
        pmf = poisson_binomial_pmf(probs)
        assert pmf.sum() == pytest.approx(1.0)
        assert (pmf >= 0).all()
        assert np.dot(np.arange(pmf.size), pmf) == pytest.approx(probs.sum())
        # comparação com simulação
        draws = (self.rng.random((20000, probs.size)) < probs).sum(axis=1)
        assert np.mean(draws <= 30) == pytest.approx(pmf[:31].sum(), abs=0.01)

    def test_poisson_binomial_equals_binomial(self):
        for x in range(6):
            for alternative in ("greater", "less", "two-sided"):
                pb = poisson_binomial_test(_hits(x, 60), [0.025] * 60, alternative=alternative)
                b = binomial_test(_hits(x, 60), 0.975, alternative=alternative)
                assert pb["p-value"] == pytest.approx(b["p-value"])
                assert pb["non-rejection region"] == b["non-rejection region"]

    def test_poisson_binomial_heterogeneous(self):
        # coberturas acima da nominal explicam mais violações
        hits = _hits(4, 58)
        probs = np.full(58, 0.039)
        assert binomial_test(hits, 0.975)["p-value"] < poisson_binomial_test(hits, probs)["p-value"]
        assert poisson_binomial_test(hits, probs)["expected violations"] == pytest.approx(58 * 0.039)
        with pytest.raises(ValueError):
            poisson_binomial_test(hits, probs[:-1])
        with pytest.raises(ValueError):
            poisson_binomial_test(hits, np.full(58, 1.5))

    # ==================== DURATION ====================
    def test_duration_restricted_is_censored_exponential(self):
        hits = (self.rng.random(1000) < 0.02).astype(int)
        hits[0], hits[-1] = 0, 0
        D, C = _durations(hits)
        k = D.size - C[0] - C[-1]
        lam = k / D.sum()
        expected = k * np.log(lam) - lam * D.sum()
        result = duration_test(hits)
        assert result["restricted log-likelihood"] == pytest.approx(expected)

    def test_duration_not_enough_data(self):
        for hits in ([0] * 50, [0] * 20 + [1] + [0] * 20, [1] + [0] * 10):
            result = duration_test(hits)
            assert np.isnan(result["p-value"])
            assert result["decision"] == "Not enough data"

    def test_duration_size_and_power(self):
        size = [
            duration_test((self.rng.random(1000) < 0.02).astype(int))["p-value"] < 0.05
            for _ in range(300)
        ]
        assert 0.02 <= np.mean(size) <= 0.10

        power = [duration_test(_clustered_violations(self.rng))["p-value"] < 0.05 for _ in range(100)]
        assert np.mean(power) >= 0.5
        assert duration_test(_clustered_violations(self.rng))["weibull shape (b)"] < 1

    # ==================== ZERO MEAN ====================
    def test_zero_mean(self):
        data = self.rng.normal(0, 1, 500)
        expected = stats.ttest_1samp(data, 0)
        for x in (data, pd.Series(data), pd.DataFrame(data), list(data)):
            result = zero_mean_test(x)
            assert result["statistic"] == pytest.approx(expected.statistic)
            assert result["p-value"] == pytest.approx(expected.pvalue)
        assert zero_mean_test(data + 1)["decision"] == "Reject H0"
        assert zero_mean_test(data + 1, true_mu=1)["decision"] == "Fail to reject H0"

    # ==================== BERKOWITZ ====================
    def test_berkowitz_size_and_power(self):
        size = [berkowitz_tail_test(self.rng.random(1000))["p-value"] < 0.05 for _ in range(200)]
        assert 0.01 <= np.mean(size) <= 0.10

        def fat_tail_pit():  # t-Student(3) avaliado como se fosse normal
            x = stats.t.rvs(3, size=1000, random_state=self.rng) / np.sqrt(3)
            return stats.norm.cdf(x)

        power = [berkowitz_tail_test(fat_tail_pit())["p-value"] < 0.05 for _ in range(50)]
        assert np.mean(power) >= 0.8

    def test_berkowitz_no_tail_observations(self):
        result = berkowitz_tail_test(np.full(100, 0.5), var_conf_level=0.99)
        assert result["statistic"] == pytest.approx(-2 * 100 * np.log(0.99))
        assert result["decision"] == "Fail to reject H0"

    def test_garch_pit(self):
        pnl = pd.Series(self.rng.normal(0, 1, 280))
        pit = garch_pit(pnl, volatility_window=250, verbose=False)
        assert len(pit) == 30
        assert pit.index.equals(pnl.index[250:])
        assert ((pit > 0) & (pit < 1)).all()

        # o PIT não depende da escala dos dados (reais, % ou fração)
        for scale in (0.01, 100):
            scaled = garch_pit(pnl * scale, volatility_window=250, verbose=False)
            assert np.allclose(scaled, pit, atol=1e-3)  # tolerância do otimizador

        with pytest.warns(DeprecationWarning):
            result = berkowtiz_tail_test(pd.DataFrame(pnl.to_numpy()), volatility_window=250)
        assert result["decision"] in ("Reject H0", "Fail to reject H0")

    def test_garch_pit_berkowitz_normal_data(self):
        """Ponta a ponta: retornos normais, GARCH(1,1) normal, Berkowitz não rejeita."""
        pnl = pd.DataFrame(self.rng.normal(0, 0.01, 500))
        pit = garch_pit(pnl, volatility_window=250, verbose=False)
        for var_conf_level in (0.95, 0.99):
            result = berkowitz_tail_test(pit, var_conf_level=var_conf_level)
            assert result["decision"] == "Fail to reject H0"

    def test_version(self):
        assert vartests.__version__ == "0.3.0"
