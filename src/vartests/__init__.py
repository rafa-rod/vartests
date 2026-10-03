from . import version
from .vartests import (
    berkowtiz_tail_test,
    duration_test,
    failure_rate,
    kupiec_test,
    zero_mean_test,
)

__version__ = version.__version__
__author__ = "Rafael Rodrigues, rafael.rafarod@gmail.com"

__all__ = [
    "__version__",
    "__author__",
    "zero_mean_test",
    "duration_test",
    "failure_rate",
    "kupiec_test",
    "berkowtiz_tail_test",
]
