# -*- coding: utf-8 -*-

from scipy import stats
import numpy as np
from scipy import optimize
from scipy.stats import chi2
import math
import arch
import time
import pygosolnp
import pandas as pd
from tqdm import tqdm
from typing import List, Union, Dict

import warnings

warnings.filterwarnings("ignore")


def zero_mean_test(
    data: pd.DataFrame, true_mu: float = 0, conf_level: float = 0.95
) -> Dict:
    """ Perfom a t-Test if mean of distribution:
         - null hypothesis (H0) = zero
         - alternative hypothesis (H1) != zero
         
        Parameters:
            data (dataframe):   pnl (distribution of profit and loss) or return
            true_mu (float):    expected mean of distribuition
            conf_level (float): test confidence level
        Returns:
            answer (dict):      statistics and decision of the test
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be dataframe.")

    significance = 1 - conf_level

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    t = (mean - true_mu) / (std / np.sqrt(len(data)))
    """p<0.05, 2-tail"""
    t_padrao = stats.t.ppf(1 - round(significance / 2, 4), len(data) - 1)
    pvalue = stats.ttest_1samp(data, popmean=true_mu, alternative="two-sided")[-1]
    H0 = "Mean of distribution = 0"
    if pvalue > significance:  # ou t < np.abs(t_padrao):
        decision = "Fail to rejected H0."
    else:
        decision = "Reject H0."

    answer = {
        "null hypothesis": H0,
        "decision": decision,
        "t-test statistc": t,
        "t-tabuladed": t_padrao,
        "p-value": pvalue,
    }

    return answer


def duration_test(
    violations: Union[List[int], pd.Series, pd.DataFrame], conf_level: float = 0.95
) -> Dict:

    """Perform the Christoffersen and Pelletier Test (2004) called Duration Test.
        The main objective is to know if the VaR model responds quickly to market movements
         in order to do not form volatility clusters.
        Duration is time betwenn violations of VaR.
        This test verifies if violations has no memory i.e. should be independent.

        Parameters:
            violations (series): series of violations of VaR
            conf_level (float):  test confidence level
        Returns:
            answer (dict):       statistics and decision of the test
    """
    typeok = False
    if isinstance(violations, pd.core.series.Series) or isinstance(violations, pd.core.frame.DataFrame):
        violations = violations.values.flatten()
        typeok = True
    elif isinstance(violations, np.ndarray):
        violations = violations.flatten()
        typeok = True        
    elif isinstance(violations, List):
        typeok = True
    if not typeok:
        raise ValueError("Input must be list, array, series or dataframe.")
        
    N = sum(violations)
    first_hit = violations[0]
    last_hit = violations[-1]

    duration = [i+1 for i, x in enumerate(violations) if x == 1]

    D = np.diff(duration)

    TN = len(violations)
    C = np.zeros(len(D))

    if not duration or (D.shape[0]==0 and len(duration)==0):
        duration = [0]
        D = [0]
        N = 1
       
    if first_hit == 0:
        C = np.append(1, C)
        D = np.append(duration[0], D)  # days until first violation

    if last_hit == 0:
        C = np.append(C, 1)
        D = np.append(D, TN - duration[-1])
       
    else:
        N = len(D)
        
    def likDurationW(x, D, C, N):
        b = x
        a = ((N - C[0] - C[N-1]) / (sum(D ** b))) ** (1 / b)
        lik = (
            C[0] * np.log(pweibull(D[0], a, b, survival=True))
            + (1 - C[0]) * dweibull(D[0], a, b, log=True)
            + sum(dweibull(D[1 : (N - 1)], a, b, log=True))
            + C[N-1] * np.log(pweibull(D[N-1], a, b, survival=True))
            + (1 - C[N-1]) * dweibull(D[N-1], a, b, log=True)
        )
    
        if np.isnan(lik) or np.isinf(lik):
            lik = 1e10
        else:
            lik = -lik
        return lik
    
    # When b=1 we get the exponential
    def dweibull(D, a, b, log=False):
        # density of Weibull
        pdf = b * np.log(a) + np.log(b) + (b - 1) * np.log(D) - (a * D) ** b
        if not log:
            pdf = np.exp(pdf)
        return pdf
    
    def pweibull(D, a, b, survival=False):
        # distribution of Weibull
        cdf = 1 - np.exp(-((a * D) ** b))
        if survival:
            cdf = 1 - cdf
        return cdf

    optimizedBetas = optimize.minimize(
        likDurationW, x0=[2], args=(D, C, N), method="L-BFGS-B", bounds=[(0.001, 10)]
    )

    print(optimizedBetas.message)

    b = optimizedBetas.x
    uLL = -likDurationW(b, D, C, N)
    rLL = -likDurationW(np.array([1]), D, C, N)
    LR = 2 * (uLL - rLL)
    LRp = 1 - chi2.cdf(LR, 1)

    H0 = "Duration Between Exceedances have no memory (Weibull b=1 = Exponential)"
    # i.e. whether we fail to reject the alternative in the LR test that b=1 (hence correct model)
    if LRp < (1 - conf_level):
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"

    answer = {
        "weibull exponential": b,
        "unrestricted log-likelihood": uLL,
        "restricted log-likelihood": rLL,
        "log-likelihood": LR,
        "log-likelihood ratio test statistic": LRp,
        "null hypothesis": H0,
        "decision": decision,
    }

    return answer


def failure_rate(violations: Union[List[int], pd.Series, pd.DataFrame]) -> Dict:
    if isinstance(violations, pd.core.series.Series) or isinstance(
        violations, pd.core.frame.DataFrame
    ):
        N = N = violations.sum()
    elif isinstance(violations, List) or isinstance(violations, np.ndarray):
        N = sum(violations)
    else:
        raise ValueError("Input must be list, array, series or dataframe.")
    TN = len(violations)

    answer = {"failure rate": N / TN}
    print(f"Failure rate of {round((N/TN)*100,2)}%")
    return answer


def kupiec_test(
    violations: Union[pd.Series, pd.DataFrame],
    var_conf_level: float = 0.99,
    conf_level: float = 0.95,
) -> Dict:

    """Perform Kupiec Test (1995).
       The main goal is to verify if the number of violations, i.e. proportion of failures, is consistent with the
       violations predicted by the model.
       
        Parameters:
            violations (series):    series of violations of VaR
            var_conf_level (float): VaR confidence level
            conf_level (float):     test confidence level
        Returns:
            answer (dict):          statistics and decision of the test
    """
    if isinstance(violations, pd.core.series.Series):
        v = violations[violations == 1].count()
    elif isinstance(violations, pd.core.frame.DataFrame):
        v = violations[violations == 1].count().values[0]
    else:
        raise ValueError("Input must be series or dataframe.")

    N = violations.shape[0]
    theta = 1 - (v / N)

    if v < 0.001:
        V = -2 * np.log((1 - (v / N)) ** (N))
    else:
        part1 = ((1 - var_conf_level) ** (v)) * (var_conf_level ** (N - v))

        part11 = ((1 - theta) ** (v)) * (theta ** (N - v))

        fact = math.factorial(N) / (math.factorial(v) * math.factorial(N - v))

        num1 = part1 * fact
        den1 = part11 * fact

        V = -2 * (np.log(num1 / den1))

    chi_square_test = chi2.cdf(V, 1)  # one degree of freedom

    if chi_square_test < conf_level:
        result = "Fail to reject H0"
    elif v == 0 and N <= 255 and var_conf_level == 0.99:
        result = "Fail to reject H0"
    else:
        result = "Reject H0"

    return {
        "statictic test": V,
        "chi square value": chi_square_test,
        "null hypothesis": f"Probability of failure is {round(1-var_conf_level,3)}",
        "result": result,
    }


def berkowtiz_tail_test(
    pnl: pd.DataFrame,
    volatility_window: int = 252,
    var_conf_level: float = 0.99,
    conf_level: float = 0.95,
    random_seed: int = 443,
) -> Dict:
    """Perform Berkowitz Test (2001).
        The goal is to verify if conditional distributions of returns "GARCH(1,1)" 
        used in the VaR Model is adherent to the data.
        In this specific test, we do not observe the whole data, only the tail.
        
        Parameters:
            data (dataframe):        pnl (distribution of profit and loss) or return
            volatility_window (int): window to cabibrate volatility GARCH model
            var_conf_level (float):  VaR confidence level
            conf_level (float):      test confidence level
            random_seed (int):       integer value to set seed to random values of the optimizer
        Returns:
            answer (dict):           statistics and decision of the test
    """
    if not isinstance(pnl, pd.DataFrame):
        raise ValueError("Input must be dataframe.")

    print("Normalizing returns...")
    conditional_vol, conditional_mean = pd.DataFrame(), pd.DataFrame()
    for t in tqdm(range(pnl.shape[0] - volatility_window + 1)):
        am = arch.arch_model(
            pnl[(t) : (volatility_window + t)],
            vol="GARCH",
            dist="normal",
            rescale=False,
        ).fit(disp="off")
        cond_vol = am.forecast(horizon=1, reindex=False).variance.apply(np.sqrt)
        cond_mean = am.forecast(horizon=1, reindex=False).mean
        conditional_vol = pd.concat([conditional_vol, cond_vol])
        conditional_mean = pd.concat([conditional_mean, cond_mean])

    ret_padr = (pnl.values - conditional_mean.values) / conditional_vol.values

    zeta = stats.norm.ppf(stats.norm.cdf(ret_padr))

    alpha = 1 - var_conf_level
    significance = 1 - conf_level

    def objective(x):
        # pars[0] => media
        # pars[1] => vol incondicional
        p1 = zeta[np.where(zeta < stats.norm.ppf((alpha)))]
        p2 = zeta[np.where(zeta >= stats.norm.ppf(alpha))] * 0 + stats.norm.ppf(alpha)
        return -(
            sum(np.log(stats.norm.pdf((p1 - x[0]) / x[1]) / x[1]))
            + sum(np.log(1 - stats.norm.cdf((p2 - x[0]) / x[1])))
        )

    print("Optimizing...")
    start = time.time()
    optimum_result = pygosolnp.solve(
        obj_func=objective,
        par_lower_limit=[-10, 0.01],
        par_upper_limit=[10, 3],
        number_of_simulations=200,  # This represents the number of starting guesses to use
        number_of_restarts=20,  # This specifies how many restarts to run from the best starting guesses
        number_of_processes=None,  # None here means to run everything single-processed
        seed=random_seed,  # Seed for reproducibility, if omitted the default random seed is used (typically cpu clock based)
        pysolnp_max_major_iter=100,  # Pysolnp property
        debug=False,
    )
    print("")
    print(f"Elapsed time: {time.time() - start} s")

    # all_results = optimum_result.all_results
    # print("; ".join([f"Solution {index + 1}: {solution.obj_value}" for index, solution in enumerate(all_results)]))
    best_solution = optimum_result.best_solution
    # print(f"Best solution {best_solution.obj_value} for parameters {best_solution.parameters}.")

    uLL = -best_solution.obj_value
    rLL = -objective([0, 1])
    LR = 2 * (uLL - rLL)
    chid = 1 - stats.chi2.cdf(LR, 2)
    if chid < significance:
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"
    H0 = "Distribuition is Normal(0,1)"

    answer = {
        "solution": best_solution,
        "ull": uLL,
        "rll": rLL,
        "LR": LR,
        "chi square test": chid,
        "null hypothesis": H0,
        "decision": decision,
    }

    return answer