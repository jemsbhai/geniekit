"""Statistics used by the aggregate layer, implemented without optional dependencies.

- :func:`wilson_interval`: confidence interval for a proportion.
- :func:`risk_difference`: difference of two proportions with a Wald interval.
- :func:`fisher_exact`: two-sided Fisher exact test for a 2x2 table.
- :func:`chi_square`: Pearson chi-square statistic for an RxC table; the
  p-value uses SciPy when it is installed and is None otherwise.
- :func:`kaplan_meier`: product-limit estimate with log-log confidence
  limits, evaluated at requested times, plus the median.

Each function is checked in the test suite against hand-computed values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "KaplanMeier",
    "chi_square",
    "fisher_exact",
    "kaplan_meier",
    "risk_difference",
    "wilson_interval",
]

Z95 = 1.959963984540054


def wilson_interval(successes: float, n: float, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0 or pd.isna(n) or pd.isna(successes):
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def risk_difference(a: float, n_a: float, b: float, n_b: float, z: float = Z95) -> tuple[float, float, float]:
    """Difference of proportions a/n_a minus b/n_b with a Wald interval."""
    if n_a <= 0 or n_b <= 0:
        return (math.nan, math.nan, math.nan)
    p1, p2 = a / n_a, b / n_b
    se = math.sqrt(p1 * (1 - p1) / n_a + p2 * (1 - p2) / n_b)
    d = p1 - p2
    return (d, d - z * se, d + z * se)


def _hypergeom_pmf(k: int, n_row1: int, n_col1: int, n: int) -> float:
    return math.comb(n_row1, k) * math.comb(n - n_row1, n_col1 - k) / math.comb(n, n_col1)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the table [[a, b], [c, d]].

    Two-sided in the usual sense (as in R's ``fisher.test``): the sum of the
    probabilities of all tables at least as unlikely as the observed one.
    """
    a, b, c, d = int(a), int(b), int(c), int(d)
    n_row1, n_col1, n = a + b, a + c, a + b + c + d
    if n == 0:
        return math.nan
    observed = _hypergeom_pmf(a, n_row1, n_col1, n)
    lo, hi = max(0, n_col1 - (n - n_row1)), min(n_row1, n_col1)
    total = 0.0
    for k in range(lo, hi + 1):
        pk = _hypergeom_pmf(k, n_row1, n_col1, n)
        if pk <= observed * (1 + 1e-7):
            total += pk
    return min(1.0, total)


def chi_square(table: pd.DataFrame | np.ndarray) -> tuple[float, int, float | None]:
    """Pearson chi-square statistic, degrees of freedom and p-value (None without SciPy)."""
    obs = np.asarray(table, dtype=float)
    obs = obs[obs.sum(axis=1) > 0][:, obs.sum(axis=0) > 0]
    if obs.ndim != 2 or obs.shape[0] < 2 or obs.shape[1] < 2:
        return (math.nan, 0, None)
    expected = obs.sum(axis=1, keepdims=True) * obs.sum(axis=0, keepdims=True) / obs.sum()
    stat = float(((obs - expected) ** 2 / expected).sum())
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    try:
        from scipy import stats as sps  # type: ignore[import-not-found]
    except ImportError:
        return (stat, dof, None)
    return (stat, dof, float(sps.chi2.sf(stat, dof)))


@dataclass
class KaplanMeier:
    """Product-limit estimate.

    ``table`` has one row per distinct event time: ``time``, ``at_risk``,
    ``events``, ``censored``, ``survival``, ``lower``, ``upper``.
    """

    n: int
    n_events: int
    table: pd.DataFrame
    median: float

    def at(self, time: float) -> tuple[float, float, float]:
        """Survival probability and confidence limits at ``time`` (1.0 before the first event)."""
        rows = self.table[self.table["time"] <= time]
        if rows.empty:
            return (1.0, 1.0, 1.0)
        last = rows.iloc[-1]
        return (float(last["survival"]), float(last["lower"]), float(last["upper"]))

    def at_risk(self, time: float) -> int:
        """Number still under observation just after ``time``."""
        times = self.table.attrs.get("_times")
        if times is None:
            return 0
        return int((times > time).sum())


def kaplan_meier(
    times: np.ndarray | pd.Series, events: np.ndarray | pd.Series, z: float = Z95
) -> KaplanMeier:
    """Kaplan-Meier estimate with log-log (Kalbfleisch-Prentice) confidence limits.

    ``times`` are durations, ``events`` 1 for an event and 0 for censoring.
    Records with a missing time are dropped.
    """
    t = pd.to_numeric(pd.Series(times), errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(pd.Series(events), errors="coerce").to_numpy(dtype=float)
    keep = ~np.isnan(t) & ~np.isnan(e)
    t, e = t[keep], e[keep].astype(int)
    order = np.argsort(t, kind="stable")
    t, e = t[order], e[order]
    n = len(t)
    rows = []
    surv = 1.0
    var_sum = 0.0  # Greenwood sum of d / (r (r - d))
    median = math.nan
    for time in np.unique(t[e == 1]):
        at_risk = int((t >= time).sum())
        d = int(((t == time) & (e == 1)).sum())
        c = int(((t == time) & (e == 0)).sum())
        surv *= 1 - d / at_risk
        if at_risk > d:
            var_sum += d / (at_risk * (at_risk - d))
        if 0 < surv < 1 and var_sum > 0:
            se = math.sqrt(var_sum) / abs(math.log(surv))
            lower = surv ** math.exp(z * se)
            upper = surv ** math.exp(-z * se)
        elif surv == 0:
            lower = upper = 0.0
        else:
            lower = upper = surv
        if math.isnan(median) and surv <= 0.5:
            median = float(time)
        rows.append(
            {
                "time": float(time),
                "at_risk": at_risk,
                "events": d,
                "censored": c,
                "survival": surv,
                "lower": lower,
                "upper": upper,
            }
        )
    table = pd.DataFrame(
        rows, columns=["time", "at_risk", "events", "censored", "survival", "lower", "upper"]
    )
    table.attrs["_times"] = t
    return KaplanMeier(n=n, n_events=int(e.sum()), table=table, median=median)
