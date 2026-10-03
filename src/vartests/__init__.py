from . import version
from .coverage import (
    binomial_power,
    binomial_test,
    failure_rate,
    kupiec_test,
    mid_p_test,
    poisson_binomial_pmf,
    poisson_binomial_test,
)
from .distribution import (
    berkowitz_tail_test,
    berkowtiz_tail_test,
    garch_pit,
    zero_mean_test,
)
from .independence import duration_test

__version__ = version.__version__
__author__ = "Rafael Rodrigues, rafael.rafarod@gmail.com"

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Coverage
    "failure_rate",
    "kupiec_test",
    "binomial_test",
    "mid_p_test",
    "poisson_binomial_test",
    "poisson_binomial_pmf",
    "binomial_power",
    # Independence
    "duration_test",
    # Distribution
    "zero_mean_test",
    "garch_pit",
    "berkowitz_tail_test",
    # Deprecated
    "berkowtiz_tail_test",
]
