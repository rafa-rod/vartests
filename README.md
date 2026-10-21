<!-- buttons -->
<p align="center">
    <a href="https://www.linkedin.com/in/rafaelrod/">
        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white"
            alt="LinkedIn"></a> &nbsp;
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
            alt="MIT license"></a> &nbsp;
    <a href="https://github.com/rafa-rod/vartests/actions/workflows/pipeline.yml">
        <img src="https://github.com/rafa-rod/vartests/actions/workflows/pipeline.yml/badge.svg"
            alt="CI"></a> &nbsp;
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg"
            alt="Code style: black"></a> &nbsp;
    <a href="http://mypy-lang.org/">
        <img src="http://www.mypy-lang.org/static/mypy_badge.svg"
            alt="Checked with mypy"></a> &nbsp;
</p>

<!-- content -->

**vartests** is a Python library with statistical tests to backtest Value at Risk (VaR) models and other quantile-based risk measures (e.g. Drawdown at Risk). Coverage tests only need the series of violations and the probability of violation, so they work for any risk measure that is a quantile.

| Group | Test | Question |
|---|---|---|
| Coverage | `kupiec_test` (Kupiec, 1995) | Is the failure rate the nominal one? (chi-square approximation, two-sided) |
| Coverage | `binomial_test` | Same question, exact binomial distribution (one- or two-sided) |
| Coverage | `mid_p_test` (Lancaster, 1961) | Exact binomial with mid-p correction, closer to the nominal size |
| Coverage | `poisson_binomial_test` | Exact test when each observation has its own probability of violation |
| Coverage | `binomial_power` | Size and power of the binomial test for a given sample |
| Coverage | `upper_confidence_limit` | Highest true failure rate compatible with the result (Clopper-Pearson) |
| Independence | `duration_test` (Christoffersen and Pelletier, 2004) | Do violations have no memory, i.e. no clusters? |
| Distribution | `berkowitz_tail_test` (Berkowitz, 2001) | Is the tail of the forecast distribution right? |
| Distribution | `zero_mean_test` | Is the mean of the PnL zero? (assumption of parametric VaR) |

## Installation

```sh
pip install vartests
```

Or the latest version from GitHub:

```sh
pip install https://github.com/rafa-rod/vartests/archive/refs/heads/main.zip
```

Requires Python 3.10 or later.

## Results

Every test returns a dictionary with the same main keys:

| Key | Content |
|---|---|
| `null hypothesis` | H0 being tested |
| `statistic` | test statistic (for exact tests, the number of violations) |
| `p-value` | p-value |
| `decision` | `"Reject H0"`, `"Fail to reject H0"` or `"Not enough data"` |

Some tests add specific keys, such as `critical value`, `non-rejection region` and `size` (exact tests) or the estimated parameters.

## Example

The file [media/Example.xlsx](media/Example.xlsx) has the PnL of a portfolio, its VaR at 99% and the violations (1 when the loss exceeds the VaR). Reading Excel files requires `openpyxl`:

```python
import pandas as pd
import vartests

data = pd.read_excel("media/Example.xlsx", index_col=0)
violations = data["Violations"]
pnl = data["PnL"]
```

All functions accept a list, NumPy array, Series or single-column DataFrame. The coverage tests also accept the count directly, as `x` violations in `n` observations, which is handy to reproduce tables or third-party results:

```python
vartests.binomial_test(x=4, n=58, risk_conf_level=0.975)
```

### Coverage

```python
vartests.failure_rate(violations)

vartests.kupiec_test(violations, risk_conf_level=0.99, conf_level=0.95)

vartests.binomial_test(violations, risk_conf_level=0.99, conf_level=0.95, alternative="greater")
vartests.mid_p_test(violations, risk_conf_level=0.99, conf_level=0.95, alternative="greater")
```

`risk_conf_level` is the confidence level of the risk measure (VaR, DaR or another quantile). The old name `var_conf_level` still works in the coverage tests, with a `FutureWarning`.

In the exact tests, `critical value` is the first number of violations that rejects H0 (for `"less"`, the largest one; for `"two-sided"`, a pair with the bound on each side, or `None` where rejection is impossible).

`alternative="greater"` (default of the exact tests) tests underestimation of risk, i.e. too many violations. Use `"two-sided"` to also detect a conservative model, or `"less"` for that alone.

With zero violations, the Kupiec and binomial tests do not reject automatically: the decision depends on the number of observations and on the nominal rate. For example, at a 5% significance level, Kupiec does not reject 0 to 1 violations in 250 days for a 99.9% VaR, but requires 1 to 6 for a 99% VaR (Giambiagi, ch. 10, Table 10.1).

Because the binomial distribution is discrete, the exact test rejects a correct model less often than the nominal level. The `size` key reports the actual probability; `mid_p_test` brings it closer to the nominal level, with no guarantee of staying below it. A common practice is to decide with the exact test and report the mid-p as a complement.

A non-rejection is only informative if the test has power. `binomial_power` shows the chance of detecting a wrong model:

```python
vartests.binomial_power(n=60, risk_conf_level=0.975, true_failure_rate=[0.05, 0.075, 0.10])
# {'critical value': 5, 'non-rejection region': (0, 4), 'size': 0.017,
#  'power': {0.05: 0.18, 0.075: 0.47, 0.1: 0.73}}
```

The rates used as keys of `power` are rounded to 10 decimals, so `power[0.075]` works even when the rate was computed as `3 * 0.025`.

`upper_confidence_limit` states what the result guarantees, without overstating it: the highest true failure rate compatible with the observed count (one-sided Clopper-Pearson). It also returns the one-sided lower limit and the two-sided interval:

```python
vartests.upper_confidence_limit(x=0, n=60, conf_level=0.95)["upper limit"]
# 0.049: with no violations in 60 episodes, the true rate is at most 4.9%
```

When the probability of violation differs across observations (a VaR with a varying level, or the expected coverage of an estimator at each episode of a Drawdown at Risk), use the Poisson-binomial test with one probability per observation:

```python
vartests.poisson_binomial_test(violations, probabilities, conf_level=0.95, alternative="greater")
```

### Independence

```python
vartests.duration_test(violations, conf_level=0.95)
```

Under H0 the time between violations is exponential (Weibull shape `b = 1`); `b < 1` indicates clusters. At least two durations, one of them uncensored, are required; otherwise the decision is `"Not enough data"`.

### Distribution

The Berkowitz test checks whether losses beyond the VaR have the size a normal VaR model predicts, not only how often they happen. It takes the realized PnL and the VaR forecast for each day, in the same unit:

```python
vartests.berkowitz_tail_test(pnl, var, var_conf_level=0.99, conf_level=0.95)
```

The test assumes a normal VaR with zero mean, so the forecast standard deviation is `|VaR_t| / z`, where `z` is the normal quantile of `var_conf_level`. The VaR can be reported as a positive loss or a negative return, as long as all values have the same sign. In the example file, the test rejects H0: the largest loss is about 3.3 times the VaR, much more than a normal tail allows.

Parametric models such as EWMA and GARCH often assume zero mean:

```python
vartests.zero_mean_test(pnl, conf_level=0.95)
```

## Changes in 0.4.1

- New `upper_confidence_limit`: Clopper-Pearson limits of the true failure rate.
- `binomial_test`, `mid_p_test`, `poisson_binomial_test` and `binomial_power` return `critical value`.
- Coverage tests accept the count (`x`, `n`) instead of the series of violations.
- `risk_conf_level` replaces `var_conf_level` in the coverage tests; the old name still works, with a `FutureWarning`.
- Keys of `binomial_power(...)["power"]` are rounded to 10 decimals.
- Exact p-values of the extreme counts are exactly 1 (previously `0.9999999999999999` in some cases).

## Changes in 0.4.0

- `berkowitz_tail_test(pnl, var, ...)` now takes the realized PnL and the VaR series, instead of the PIT, and assumes a normal VaR with zero mean.
- `berkowtiz_tail_test` (misspelled, deprecated in 0.3.0) was removed.

## Changes in 0.3.0

Version 0.3.0 fixes errors and changes the output of some functions:

- `kupiec_test` no longer rejects automatically with zero violations and returns the p-value.
- `duration_test` fixes the likelihood when the series does not end with a violation (the usual case); p-values change in most samples.
- `zero_mean_test` works with pandas 2 and 3 and accepts Series.
- `berkowitz_tail_test` receives the PIT of the model being tested. Results differ slightly from 0.2.x because the conditional mean is now forecast without look-ahead.
- Result keys are standardized (`statistic`, `p-value`, `decision`).
- New tests: `binomial_test`, `mid_p_test`, `poisson_binomial_test` and `binomial_power`.

## References

- Berkowitz, J. (2001). Testing density forecasts, with applications to risk management. *Journal of Business & Economic Statistics*, 19(4), 465-474.
- Christoffersen, P. and Pelletier, D. (2004). Backtesting Value-at-Risk: a duration-based approach. *Journal of Financial Econometrics*, 2(1), 84-108.
- Giambiagi, F. (Org.). *Derivativos e risco de mercado*. Elsevier. Ch. 10 (Backtesting).
- Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73-84.
- Lancaster, H. O. (1961). Significance tests in discrete distributions. *Journal of the American Statistical Association*, 56(294), 223-234.
