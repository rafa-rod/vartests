<!-- buttons -->
<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
            alt="MIT license"></a> &nbsp;
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg"
            alt="Code style: black"></a> &nbsp;
    <a href="http://mypy-lang.org/">
        <img src="http://www.mypy-lang.org/static/mypy_badge.svg"
            alt="Checked with mypy"></a> &nbsp;
</p>

<!-- content -->

**vartests** is a Python library to perform some statistical tests to evaluate Value at Risk (VaR) Models, such as:

- **T-test**: verify if mean of distribution is zero;
- **Kupiec Test (1995)**: verify if the number of violations is consistent with the violations predicted by the model;
- **Berkowitz Test (2001)**: verify if conditional distributions of returns "GARCH(1,1)"  used in the VaR Model is adherent to the data. In this specific test, we do not observe the whole data, only the tail;
- **Christoffersen and Pelletier Test (2004)**: also known as Duration Test. Duration is time between violations of VaR. It tests if VaR Model has quickly response to market movements by consequence the violations do not form volatility clusters. This test verifies if violations has no memory i.e. should be independent.

## Installation

### Using pip

You can install using the pip package manager by running:

```sh
pip install vartests
```

Alternatively, you could install the latest version directly from Github:

```sh
pip install https://github.com/rafa-rod/vartests/archive/refs/heads/main.zip
```

## Why vartests is important?

After VaR calculation, it is necessary to perform statistic tests to evaluate the VaR Models. To select the best model, they should be validated by backtests.

## Example

First of all, lets read a file with a PnL (distribution of profit and loss) of a portfolio in which also contains the VaR and its violations.

```python
import pandas as pd

data = pd.read_excel("Example.xlsx", index_col=0)
violations = data["Violations"]
pnl = data["PnL"] 
data.sample(5)
```

The dataframe looks like:

```
' |     PnL       |      VaR        |   Violations |
  | -889.003707   | -2554.503872    |            0 |
  | -2554.503872  | -2202.221691    |            1 | 
  | -887.527423   | -2193.692570    |            0 |  
  | -274.344126   | -2160.290746    |            0 | 
  | 1376.018638   | -5719.833100    |            0 |'
```

Not all tests should be applied to the VaR Model. Some of them should be applied when the VaR Model has the assumption of zero mean or follow a specific distribution.

```python
import vartests

vartests.zero_mean_test(pnl.values, conf_level=0.95)
```

This assumption is commonly used in parametric VaR like EWMA and GARCH Models. Besides that, is necessary check assumption of the distribution. So you should test with Berkowitz (2001):

```python
import vartests

vartests.berkowtiz_tail_test(pnl, volatility_window=252, var_conf_level=0.99, conf_level=0.95)
```

The following tests should be used to any kind of VaR Models.

```python
import vartests

vartests.kupiec_test(violations, var_conf_level=0.99, conf_level=0.95)

vartests.duration_test(violations, conf_level=0.95)
```

If you want to see the failure ratio of the VaR Model, just type:

```python
import vartests

vartests.failure_rate(violations)
```