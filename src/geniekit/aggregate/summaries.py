"""Aggregate summaries with disclosure control.

Every function takes a patient-level frame (normally from
:func:`geniekit.aggregate.analytic_table`) and returns a :class:`SafeTable`:
counts below the minimum cell size are suppressed, a second cell is
suppressed where one suppressed cell could be recovered from a margin, and
percentages, intervals, quantiles, differences and p-values that derive from
a suppressed count are suppressed with it. No function returns rows of the
input.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cohort import DAYS_PER_MONTH
from .disclosure import DisclosureControl, SafeTable
from .stats import chi_square, fisher_exact, kaplan_meier, risk_difference, wilson_interval

__all__ = [
    "AggregateReport",
    "early_failure_table",
    "flag_frequency_table",
    "frequency_table",
    "km_table",
    "numeric_summary",
    "proportion_table",
]

TOTAL = "Total"
MISSING = "Missing"


def _control(control: DisclosureControl | None) -> DisclosureControl:
    return control if control is not None else DisclosureControl()


def _p(value: float | None) -> object:
    """A p-value rounded to four decimals, with values below 0.0001 shown as such."""
    if value is None or pd.isna(value):
        return np.nan
    return "<0.0001" if value < 0.0001 else round(float(value), 4)


def _groups(frame: pd.DataFrame, by: str | None, dropna: bool) -> pd.Series:
    """The grouping labels as strings (``Total`` for a single group)."""
    if by is None:
        return pd.Series(TOTAL, index=frame.index, dtype=object)
    g = frame[by]
    if pd.api.types.is_bool_dtype(g) or str(g.dtype) == "boolean":
        labels = g.map({True: f"{by}: yes", False: f"{by}: no"})
    else:
        labels = g.astype(object).where(g.notna(), None)
    labels = labels.astype(object)
    if not dropna:
        labels = labels.where(g.notna(), MISSING)
    return labels


def _order(labels: Iterable[Any]) -> list[Any]:
    vals = [v for v in dict.fromkeys(labels) if v is not None and v == v]
    try:
        return sorted(vals)
    except TypeError:
        return [str(v) for v in vals]


def _shown_groups(groups: pd.Series, ctl: DisclosureControl) -> tuple[list[Any], list[tuple[Any, int]]]:
    """Groups large enough to summarise (at least the minimum denominator) and the omitted ones with sizes."""
    sizes = groups.value_counts()
    order = _order(groups)
    shown = [g for g in order if sizes.get(g, 0) >= ctl.denominator]
    omitted = [(g, int(sizes.get(g, 0))) for g in order if g not in shown]
    return shown, omitted


def _group_notes(omitted: list[tuple[Any, int]], ctl: DisclosureControl) -> list[str]:
    if not omitted:
        return []
    text = ", ".join(f"{g} ({n if n >= ctl.min_count else ctl.label})" for g, n in omitted)
    return [
        f"Groups with fewer than {ctl.denominator} records are not shown: {text}; the total excludes them."
    ]


# ----------------------------------------------------------------------------- frequencies
def frequency_table(
    frame: pd.DataFrame,
    variable: str,
    by: str | None = None,
    control: DisclosureControl | None = None,
    name: str | None = None,
    dropna: bool = False,
) -> SafeTable:
    """Counts and column percentages of ``variable``, optionally by ``by``.

    Rows are the levels of ``variable`` (plus ``Total``), columns ``<group> n``
    and ``<group> %`` for every group plus ``Total``. Percentages are of the
    column total. Complementary suppression protects suppressed counts
    against the margins.
    """
    ctl = _control(control)
    rows = _groups(frame, variable, dropna)
    cols = _groups(frame, by, dropna)
    keep = rows.notna() & cols.notna()
    rows, cols = rows[keep], cols[keep]
    if by is None:
        vc = rows.value_counts(dropna=False)
        vc = vc.reindex(_order(vc.index) + ([MISSING] if MISSING in vc.index else []))
        counts = pd.DataFrame({TOTAL: vc.astype(int)})
        counts.loc[TOTAL] = int(vc.sum())
    else:
        counts = pd.crosstab(rows, cols, margins=True, margins_name=TOTAL, dropna=False)
    counts.index.name = variable
    counts.columns.name = by
    masked, mask, records, omitted = ctl.mask_counts(counts, total_row=True, total_col=by is not None)
    shown = masked.map(lambda v: np.nan if isinstance(v, str) else float(v))
    totals = shown.loc[TOTAL]
    out = pd.DataFrame(index=masked.index)
    for col in masked.columns:
        out[f"{col} n"] = masked[col]
        pct = []
        for level in masked.index:
            if mask.loc[level, col]:
                pct.append(ctl.label)
            else:
                pct.append(ctl.percent(shown.loc[level, col], totals[col]))
        out[f"{col} %"] = pct
    notes = (
        [f"Column percentages; {MISSING} rows and columns count records with no value."]
        if not dropna
        else ["Column percentages; records with no value excluded."]
    )
    notes += _omitted_notes(omitted, ctl)
    return SafeTable(
        name=name or (f"{variable} by {by}" if by else variable),
        frame=out,
        n=int(keep.sum()),
        suppressed=records,
        notes=notes,
        settings=ctl.describe(),
        omitted=omitted,
    )


def _omitted_notes(omitted: dict[str, list[tuple[Any, Any]]], ctl: DisclosureControl) -> list[str]:
    notes = []
    for kind in ("rows", "columns"):
        items = omitted.get(kind, [])
        if items:
            text = ", ".join(f"{level} ({n if n >= ctl.min_count else ctl.label})" for level, n in items)
            notes.append(
                f"{kind.capitalize()} omitted rather than suppressing larger cells: {text}; "
                "totals cover the lines shown."
            )
    return notes


def numeric_summary(
    frame: pd.DataFrame,
    variable: str,
    by: str | None = None,
    control: DisclosureControl | None = None,
    name: str | None = None,
    decimals: int = 1,
) -> SafeTable:
    """Median and quartiles of ``variable`` per group (never the minimum or maximum)."""
    ctl = _control(control)
    values = pd.to_numeric(frame[variable], errors="coerce")
    groups = _groups(frame, by, dropna=True)
    valid = values.notna() & groups.notna()
    values, groups = values[valid], groups[valid]
    shown, omitted = _shown_groups(groups, ctl) if by else ([], [])
    rows = []
    for g in [*shown, TOTAL]:
        v = (
            values[groups.isin(shown)]
            if (g == TOTAL and by)
            else (values if g == TOTAL else values[groups == g])
        )
        n = len(v)
        if n == 0:
            continue
        if ctl.allows(n):
            q1, med, q3 = (round(float(x), decimals) for x in v.quantile([0.25, 0.5, 0.75]))
            rows.append({"group": g, "n": n, "median": med, "q1": q1, "q3": q3})
        else:
            rows.append({"group": g, "n": n, "median": ctl.label, "q1": ctl.label, "q3": ctl.label})
    out = pd.DataFrame(rows, columns=["group", "n", "median", "q1", "q3"]).set_index("group")
    out.index.name = by or "group"
    return SafeTable(
        name=name or (f"{variable} by {by}" if by else variable),
        frame=out,
        n=int(valid.sum()),
        notes=[f"{variable}: median and quartiles of records with a value.", *_group_notes(omitted, ctl)],
        settings=ctl.describe(),
    )


# ----------------------------------------------------------------------------- proportions
def _proportion_rows(
    outcome: pd.Series, groups: pd.Series, order: list[Any], ctl: DisclosureControl, reference: Any | None
) -> tuple[list[dict[str, Any]], list[Any]]:
    rows: list[dict[str, Any]] = []
    stats: dict[Any, tuple[int, int]] = {}
    for g in [*order, TOTAL]:
        o = outcome if g == TOTAL else outcome[groups == g]
        o = o.dropna().astype(bool)
        n, k = len(o), int(o.sum())
        stats[g] = (n, k)
        row: dict[str, Any] = {"group": g, "n": n, "events": k}
        small = (0 < k < ctl.min_count) or (0 < n - k < ctl.min_count) or n < ctl.denominator
        if n == 0:
            row.update({"events": 0, "%": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
        elif small:
            row.update({"events": ctl.label, "%": ctl.label, "ci_lower": ctl.label, "ci_upper": ctl.label})
        else:
            lo, hi = wilson_interval(k, n)
            row.update(
                {"%": round(100 * k / n, 1), "ci_lower": round(100 * lo, 1), "ci_upper": round(100 * hi, 1)}
            )
        rows.append(row)
    ref = reference if reference is not None else (order[0] if order else None)
    if ref is not None and len(order) > 1:
        n_r, k_r = stats[ref]
        ref_small = (0 < k_r < ctl.min_count) or (0 < n_r - k_r < ctl.min_count) or n_r < ctl.denominator
        for row in rows:
            g = row["group"]
            if g in (ref, TOTAL):
                row.update(
                    {"risk_difference": np.nan, "rd_lower": np.nan, "rd_upper": np.nan, "p_fisher": np.nan}
                )
                continue
            n_g, k_g = stats[g]
            small = (0 < k_g < ctl.min_count) or (0 < n_g - k_g < ctl.min_count) or n_g < ctl.denominator
            if small or ref_small or n_g == 0 or n_r == 0:
                row.update(
                    {
                        "risk_difference": ctl.label,
                        "rd_lower": ctl.label,
                        "rd_upper": ctl.label,
                        "p_fisher": ctl.label,
                    }
                )
            else:
                d, lo, hi = risk_difference(k_g, n_g, k_r, n_r)
                row.update(
                    {
                        "risk_difference": round(100 * d, 1),
                        "rd_lower": round(100 * lo, 1),
                        "rd_upper": round(100 * hi, 1),
                        "p_fisher": _p(fisher_exact(k_g, n_g - k_g, k_r, n_r - k_r)),
                    }
                )
    return rows, [ref] if ref is not None else []


def proportion_table(
    frame: pd.DataFrame,
    outcome: str,
    by: str | None = None,
    control: DisclosureControl | None = None,
    name: str | None = None,
    reference: Any | None = None,
) -> SafeTable:
    """Proportion of records with ``outcome`` (a boolean column) per group, with Wilson intervals.

    With a grouping, each group is compared with the reference group (the
    first level unless ``reference`` is given): risk difference in percentage
    points with a Wald interval, and a two-sided Fisher exact p-value.
    """
    ctl = _control(control)
    groups = _groups(frame, by, dropna=True)
    shown, omitted = _shown_groups(groups[frame[outcome].notna() & groups.notna()], ctl) if by else ([], [])
    keep = groups.isin(shown) if by else pd.Series(True, index=frame.index)
    rows, refs = _proportion_rows(frame.loc[keep, outcome], groups[keep], shown, ctl, reference)
    out = pd.DataFrame(rows).set_index("group")
    out.index.name = by or "group"
    notes = [f"{outcome}: proportion of records with the outcome; Wilson 95% interval."]
    if refs:
        notes.append(
            f"Risk difference (percentage points, Wald 95% interval) and Fisher exact p versus {refs[0]!r}."
        )
    notes += _group_notes(omitted, ctl)
    return SafeTable(
        name=name or (f"{outcome} by {by}" if by else outcome),
        frame=out,
        n=int(frame.loc[keep, outcome].notna().sum()),
        notes=notes,
        settings=ctl.describe(),
    )


def flag_frequency_table(
    frame: pd.DataFrame,
    flags: Iterable[str],
    by: str | None = None,
    control: DisclosureControl | None = None,
    name: str | None = None,
    subset: str | None = None,
) -> SafeTable:
    """Frequency of each boolean flag (one row per flag), per group and overall.

    ``subset`` names a boolean column restricting the denominator (for
    example ``sequenced``). With exactly two groups a Fisher exact p-value
    compares them; with more, a chi-square p-value (SciPy) when available.
    """
    ctl = _control(control)
    base = frame if subset is None else frame[frame[subset].fillna(False).astype(bool)]
    groups = _groups(base, by, dropna=True)
    order, omitted = _shown_groups(groups[groups.notna()], ctl) if by else ([], [])
    if by:
        keep = groups.isin(order)
        base, groups = base[keep], groups[keep]
    rows = []
    for flag in flags:
        values = base[flag]
        row: dict[str, Any] = {"flag": flag}
        counts: dict[Any, tuple[int, int]] = {}
        for g in [*order, TOTAL]:
            v = values if g == TOTAL else values[groups == g]
            v = v.dropna().astype(bool)
            n, k = len(v), int(v.sum())
            counts[g] = (n, k)
            row[f"{g} n"] = n
            small = (0 < k < ctl.min_count) or (0 < n - k < ctl.min_count) or n < ctl.denominator
            if n == 0:
                row[f"{g} altered"] = 0
                row[f"{g} %"] = np.nan
                row[f"{g} ci"] = np.nan
            elif small:
                row[f"{g} altered"] = ctl.label
                row[f"{g} %"] = ctl.label
                row[f"{g} ci"] = ctl.label
            else:
                lo, hi = wilson_interval(k, n)
                row[f"{g} altered"] = k
                row[f"{g} %"] = round(100 * k / n, 1)
                row[f"{g} ci"] = f"{100 * lo:.1f} to {100 * hi:.1f}"
        if len(order) >= 2:
            any_small = any(
                (0 < k < ctl.min_count) or (0 < n - k < ctl.min_count) or n < ctl.denominator
                for n, k in (counts[g] for g in order)
            )
            if any_small:
                row["p"] = ctl.label
            elif len(order) == 2:
                (n1, k1), (n2, k2) = counts[order[0]], counts[order[1]]
                row["p"] = _p(fisher_exact(k1, n1 - k1, k2, n2 - k2))
            else:
                table = np.array([[counts[g][1], counts[g][0] - counts[g][1]] for g in order])
                _, _, p = chi_square(table)
                row["p"] = _p(p)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("flag")
    notes = [
        "Per flag: records with a value, records altered (True) and the percentage with a Wilson 95% interval"
    ]
    if subset:
        notes.append(f"Denominator restricted to records with {subset} = True.")
    if len(order) == 2:
        notes.append(f"p: Fisher exact test, {order[0]!r} versus {order[1]!r}.")
    elif len(order) > 2:
        notes.append("p: Pearson chi-square across groups (needs SciPy).")
    notes += _group_notes(omitted, ctl)
    return SafeTable(
        name=name or ("flags by " + by if by else "flags"),
        frame=out,
        n=len(base),
        notes=notes,
        settings=ctl.describe(),
    )


# ----------------------------------------------------------------------------- time to event
def km_table(
    frame: pd.DataFrame,
    time: str,
    event: str,
    by: str | None = None,
    at_months: Iterable[float] = (6, 12, 24),
    control: DisclosureControl | None = None,
    name: str | None = None,
    failure: bool = False,
) -> SafeTable:
    """Kaplan-Meier estimates at fixed times per group, with the median.

    ``time`` is in days; ``at_months`` are converted with the GENIE convention
    of 30.4 days per month. With ``failure=True`` the table reports the
    cumulative probability of the event (1 minus survival). Groups with fewer
    records than the minimum denominator are shown with their size only.
    """
    ctl = _control(control)
    groups = _groups(frame, by, dropna=True)
    t_all = pd.to_numeric(frame[time], errors="coerce")
    e_all = pd.to_numeric(frame[event], errors="coerce")
    valid = t_all.notna() & e_all.notna() & groups.notna()
    shown, omitted = _shown_groups(groups[valid], ctl) if by else ([], [])
    order = [*shown, TOTAL] if by else [TOTAL]
    rows = []
    for g in order:
        if g != TOTAL:
            sel = groups == g
        elif by:
            sel = groups.isin(shown)
        else:
            sel = pd.Series(True, index=frame.index)
        sel &= valid
        n = int(sel.sum())
        row: dict[str, Any] = {"group": g, "n": n, "events": np.nan}
        for m in at_months:
            row.update({f"{m} mo %": np.nan, f"{m} mo ci": np.nan, f"{m} mo at risk": np.nan})
        row["median months"] = np.nan
        if n == 0:
            rows.append(row)
            continue
        km = kaplan_meier(t_all[sel], e_all[sel])
        events = km.n_events
        allowed = ctl.allows(n) and not (0 < events < ctl.min_count)
        row["events"] = events if allowed or events == 0 else ctl.label
        for m in at_months:
            s, lo, hi = km.at(m * DAYS_PER_MONTH)
            at_risk = km.at_risk(m * DAYS_PER_MONTH)
            if not allowed:
                row[f"{m} mo %"] = ctl.label
                row[f"{m} mo ci"] = ctl.label
                row[f"{m} mo at risk"] = ctl.label
                continue
            if failure:
                s, lo, hi = 1 - s, 1 - hi, 1 - lo
            row[f"{m} mo %"] = round(100 * s, 1)
            row[f"{m} mo ci"] = f"{100 * lo:.1f} to {100 * hi:.1f}"
            row[f"{m} mo at risk"] = at_risk if (at_risk == 0 or at_risk >= ctl.min_count) else ctl.label
        if allowed:
            row["median months"] = (
                round(km.median / DAYS_PER_MONTH, 1) if not math.isnan(km.median) else "not reached"
            )
        else:
            row["median months"] = ctl.label
        rows.append(row)
    out = pd.DataFrame(rows).set_index("group")
    out.index.name = by or "group"
    what = "cumulative probability of the event" if failure else "survival probability"
    return SafeTable(
        name=name or (f"{event} by {by}" if by else event),
        frame=out,
        n=int(valid.sum()),
        notes=[
            f"Kaplan-Meier {what} at fixed times with log-log 95% limits; median time to event in months "
            "(30.4 days per month).",
            *_group_notes(omitted, ctl),
        ],
        settings=ctl.describe(),
    )


def early_failure_table(
    frame: pd.DataFrame,
    months: Iterable[float] = (6, 9),
    by: str | None = None,
    control: DisclosureControl | None = None,
    name: str | None = None,
    reference: Any | None = None,
    time: str = "failure_days",
    event: str = "failure_event",
) -> SafeTable:
    """Early treatment failure after first-line therapy, at each horizon in months.

    For each horizon M: ``evaluable`` records are those with a failure on or
    before M months or with follow-up beyond M months without failure;
    ``failed`` is the number and proportion failing among them (Wilson
    interval), compared with the reference group by risk difference and
    Fisher exact p; ``km failed %`` is the Kaplan-Meier cumulative failure
    probability at M months over all records, which uses the censored
    follow-up as well.
    """
    ctl = _control(control)
    started = frame[pd.to_numeric(frame[time], errors="coerce").notna()].copy()
    groups = _groups(started, by, dropna=True)
    order, omitted = _shown_groups(groups[groups.notna()], ctl) if by else ([], [])
    if by:
        keep = groups.isin(order)
        started, groups = started[keep], groups[keep]
    t = pd.to_numeric(started[time], errors="coerce")
    e = pd.to_numeric(started[event], errors="coerce").fillna(0).astype(int)
    if started.empty:
        return SafeTable(
            name=name or ("early failure by " + by if by else "early failure"),
            frame=pd.DataFrame(columns=["started"]),
            n=0,
            notes=["No record started a first line.", *_group_notes(omitted, ctl)],
            settings={**ctl.describe(), "months": list(months)},
        )
    out: pd.DataFrame | None = None
    for m in months:
        horizon = m * DAYS_PER_MONTH
        failed = (e == 1) & (t <= horizon)
        evaluable = failed | (t >= horizon)
        tmp = started.assign(_eval=evaluable, _failed=failed.where(evaluable, other=pd.NA).astype("boolean"))
        rows, _ = _proportion_rows(tmp["_failed"], groups, order, ctl, reference)
        km = km_table(started, time, event, by=by, at_months=[m], control=ctl, failure=True)
        part = pd.DataFrame(rows).set_index("group")
        part = part.rename(
            columns={
                "n": f"{m} mo evaluable",
                "events": f"{m} mo failed",
                "%": f"{m} mo failed %",
                "ci_lower": f"{m} mo ci_lower",
                "ci_upper": f"{m} mo ci_upper",
                "risk_difference": f"{m} mo risk_difference",
                "rd_lower": f"{m} mo rd_lower",
                "rd_upper": f"{m} mo rd_upper",
                "p_fisher": f"{m} mo p_fisher",
            }
        )
        part[f"{m} mo km failed %"] = km.frame[f"{m} mo %"].reindex(part.index)
        part[f"{m} mo km ci"] = km.frame[f"{m} mo ci"].reindex(part.index)
        out = part if out is None else out.join(part)
    assert out is not None
    n_started = pd.Series({g: int((groups == g).sum()) for g in order} | {TOTAL: len(started)})
    out.insert(0, "started", n_started.reindex(out.index))
    out.index.name = by or "group"
    notes = [
        "Records that started a first line; for each horizon, evaluable = failed on or before the horizon or "
        "followed beyond it without failure; failed % with Wilson 95% interval among evaluable records.",
        "km failed %: Kaplan-Meier cumulative failure probability at the horizon over all started records "
        "(log-log 95% limits).",
        "Failure = first progression assessment after the first-line start, the start of the next line, "
        "or death.",
    ]
    if order:
        ref = reference if reference is not None else order[0]
        notes.append(f"Risk difference (percentage points) and Fisher exact p versus {ref!r}.")
    notes += _group_notes(omitted, ctl)
    return SafeTable(
        name=name or ("early failure by " + by if by else "early failure"),
        frame=out,
        n=len(started),
        notes=notes,
        settings={**ctl.describe(), "months": list(months)},
    )


# ----------------------------------------------------------------------------- report container
@dataclass
class AggregateReport:
    """A named collection of :class:`SafeTable` with shared settings and notes."""

    name: str
    tables: dict[str, SafeTable] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> SafeTable:
        return self.tables[key]

    def __iter__(self):
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)

    def add(self, table: SafeTable, key: str | None = None) -> SafeTable:
        self.tables[key or table.name] = table
        return table

    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "table": k,
                "rows": t.frame.shape[0],
                "columns": t.frame.shape[1],
                "n": t.n,
                "suppressed": t.n_suppressed,
            }
            for k, t in self.tables.items()
        ]
        return pd.DataFrame(rows, columns=["table", "rows", "columns", "n", "suppressed"])

    def to_dir(self, path: str | Path) -> Path:
        """Write every table as CSV plus ``report.json`` with settings, notes and suppressions."""
        import json

        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "name": self.name,
            "settings": self.settings,
            "notes": self.notes,
            "tables": {},
        }
        for key, table in self.tables.items():
            table.frame.to_csv(folder / f"{key}.csv")
            manifest["tables"][key] = {
                "file": f"{key}.csv",
                "n": table.n,
                "notes": table.notes,
                "suppressed": [
                    {"row": str(s.row), "column": str(s.column), "kind": s.kind} for s in table.suppressed
                ],
            }
        (folder / "report.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return folder

    def __repr__(self) -> str:
        return f"AggregateReport({self.name!r}, {len(self.tables)} tables)"
