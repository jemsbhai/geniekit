"""Apply the crosswalk to a cohort and stack harmonised cohorts.

>>> from geniekit import synthetic, harmonize
>>> nsclc = harmonize.harmonize(synthetic.generate("NSCLC", 30, seed=1))
>>> "ca_cigarette" in nsclc["cancer_index"].columns
True
>>> stacked = harmonize.stack([nsclc, harmonize.harmonize(synthetic.generate("BLADDER", 30, seed=2))])
>>> sorted(stacked["cancer_index"]["cohort"].unique())
['BLADDER', 'NSCLC']
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import dictionary
from ..io.cohort import CLINICAL_TABLES, CohortData
from ..releases import COHORTS
from . import derive as derive_methods
from . import genomic
from .crosswalk import Crosswalk, Rule, load_crosswalk
from .report import HarmonizationReport, Record

__all__ = ["HarmonizedCohort", "StackedData", "harmonize", "stack"]

_TIME_SUFFIXES = ("_days", "_mos", "_yrs")
_CODE_RE = re.compile(r"^\s*([^\s=]{1,12})\s*=\s*\S")
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?(\s*-\s*\d+)?$")


@dataclass
class HarmonizedCohort(CohortData):
    """A :class:`CohortData` after harmonisation, with its report."""

    report: HarmonizationReport = field(default_factory=HarmonizationReport)
    crosswalk_version: int | None = None

    def __repr__(self) -> str:
        n = len(self.report.applied()) if len(self.report) else 0
        return f"HarmonizedCohort({self.label}: {len(self.tables)} tables, {n} rules applied)"


def _norm_value(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return " ".join(text.split()) if text else None


def _apply_rename(
    frame: pd.DataFrame, rule: Rule, keep_original: bool
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Returns the frame and a list of (source, target, status)."""
    pairs: list[tuple[str, str]] = [(rule.params["from"], rule.params["to"])]
    for sib in rule.params.get("siblings", []) or []:
        for suffix in _TIME_SUFFIXES:
            pairs.append((f"{sib['from']}{suffix}", f"{sib['to']}{suffix}"))
    results: list[tuple[str, str, str]] = []
    out = frame
    for src, dst in pairs:
        if src not in out.columns:
            results.append((src, dst, "skipped: column absent"))
            continue
        if dst in out.columns:
            results.append((src, dst, "skipped: target already present"))
            continue
        if keep_original:
            out = out.copy()
            out.insert(out.columns.get_loc(src) + 1, f"{src}_orig", out[src])
        out = out.rename(columns={src: dst})
        results.append((src, dst, "applied"))
    return out, results


def _apply_map(
    frame: pd.DataFrame,
    variable: str,
    mapping: dict[str, str],
    keep_detail: bool,
    known: set[str] | None = None,
) -> tuple[pd.DataFrame, int, list[str]]:
    """Map values of one column; returns (frame, n_changed, unmapped values).

    ``known`` holds the levels that need no mapping (for partial, spelling-only
    maps); values outside the map, its targets and ``known`` are reported.
    """
    norm_map = {" ".join(k.split()): v for k, v in mapping.items()}
    canonical = set(norm_map.values()) | (known or set())
    series = frame[variable]
    new_values: list[object] = []
    changed = 0
    unmapped: set[str] = set()
    for value in series:
        n = _norm_value(value)
        if n is None:
            new_values.append(value)
            continue
        if n in norm_map:
            target = norm_map[n]
            new_values.append(target)
            if target != n:
                changed += 1
        else:
            new_values.append(value)
            if n not in canonical and not _NUMBER_RE.match(n):
                unmapped.add(n)
    out = frame.copy()
    if keep_detail:
        detail_col = f"{variable}_detail"
        if detail_col not in out.columns:
            out.insert(out.columns.get_loc(variable) + 1, detail_col, series.astype(object))
    out[variable] = pd.array(new_values, dtype=object)
    return out, changed, sorted(unmapped)


def _apply_coerce(frame: pd.DataFrame, variable: str, to: str) -> tuple[pd.DataFrame, int, int]:
    series = frame[variable]
    n_before = int(series.notna().sum())
    if to == "Int64":
        converted = pd.to_numeric(series, errors="coerce")
        converted = converted.round().astype("Int64")
    elif to in ("float", "Float64"):
        converted = pd.to_numeric(series, errors="coerce").astype("Float64" if to == "Float64" else float)
    elif to in ("string", "str"):
        converted = (
            series.astype(object).where(series.notna(), None).map(lambda v: None if v is None else str(v))
        )
    else:
        raise ValueError(f"unsupported coercion target {to!r}")
    n_after = int(pd.Series(converted).notna().sum())
    out = frame.copy()
    out[variable] = converted
    return out, n_before, n_after


def _known_levels(
    dd: dictionary.DataDictionary, table: str, variable: str, original: str, exclude: tuple[str, ...]
) -> set[str]:
    """Declared levels of a variable in the cohorts a rule does not apply to (the canonical ones)."""
    dict_table = "cancer" if table in ("cancer_index", "cancer_non_index") else table
    levels: set[str] = set()
    for name in {variable, original}:
        try:
            values = dd.values(name, dict_table)
        except ValueError:
            continue
        for cohort, vals in values.items():
            if cohort in exclude:
                continue
            for v in vals:
                text = " ".join(v.split())
                levels.add(text)
                m = _CODE_RE.match(text)
                if m:
                    levels.add(m.group(1))
    return levels


def _rule_variables(rule: Rule, frame: pd.DataFrame) -> list[str]:
    if "variables" in rule.params:
        return [v for v in rule.params["variables"] if v in frame.columns]
    v = rule.params.get("variable")
    return [v] if v and v in frame.columns else []


def harmonize(
    data: CohortData,
    crosswalk: Crosswalk | None = None,
    keep_original: bool = False,
    genomic_tables: bool = True,
) -> HarmonizedCohort:
    """Apply the crosswalk rules that concern ``data.cohort`` to every table.

    Parameters
    ----------
    data:
        A loaded or synthetic release.
    crosswalk:
        Defaults to the packaged crosswalk.
    keep_original:
        Keep renamed columns under ``<name>_orig`` as well. Collapsed columns
        always keep their original value in ``<name>_detail``.
    genomic_tables:
        Also unify the structural-variant format and add ``sample_id`` to the
        mutations table.
    """
    cw = crosswalk or load_crosswalk()
    dd = dictionary.load()
    report = HarmonizationReport()
    tables: dict[str, pd.DataFrame] = {}
    for rule in cw.rules_for(data.cohort, kinds=("note",)):
        if rule.table == "*":
            report.add(
                Record(
                    data.cohort,
                    "*",
                    rule.id,
                    "note",
                    "",
                    "",
                    0,
                    0,
                    "note",
                    " ".join(rule.basis.split())[:300],
                    rule.caveat or "",
                )
            )
    for name, frame in data.tables.items():
        if name not in CLINICAL_TABLES:
            tables[name] = frame
            continue
        out = frame
        renamed_from: dict[str, str] = {}
        for rule in cw.rules_for(data.cohort, name):
            if rule.table == "*":
                continue
            if rule.kind == "rename":
                out, results = _apply_rename(out, rule, keep_original)
                for src, dst, status in results:
                    if status == "applied":
                        renamed_from[dst] = src
                    report.add(
                        Record(
                            data.cohort,
                            name,
                            rule.id,
                            "rename",
                            src,
                            dst,
                            len(out),
                            len(out) if status == "applied" else 0,
                            "applied" if status == "applied" else "skipped",
                            "" if status == "applied" else status,
                            rule.caveat or "",
                        )
                    )
            elif rule.kind in ("recode", "collapse"):
                for var in _rule_variables(rule, out):
                    known = _known_levels(dd, name, var, renamed_from.get(var, var), rule.cohorts)
                    out, changed, unmapped = _apply_map(
                        out, var, rule.params["map"], keep_detail=rule.kind == "collapse", known=known
                    )
                    detail = f"unmapped: {', '.join(unmapped[:10])}" if unmapped else ""
                    report.add(
                        Record(
                            data.cohort,
                            name,
                            rule.id,
                            rule.kind,
                            var,
                            var,
                            len(out),
                            changed,
                            "applied",
                            detail,
                            rule.caveat or "",
                        )
                    )
                for var in set(rule.variables) - set(out.columns):
                    report.add(
                        Record(
                            data.cohort,
                            name,
                            rule.id,
                            rule.kind,
                            var,
                            var,
                            len(out),
                            0,
                            "skipped",
                            "column absent",
                            rule.caveat or "",
                        )
                    )
            elif rule.kind == "coerce":
                for var in _rule_variables(rule, out):
                    out, before, after = _apply_coerce(out, var, rule.params["to"])
                    detail = f"{before - after} value(s) became missing" if after < before else ""
                    report.add(
                        Record(
                            data.cohort,
                            name,
                            rule.id,
                            "coerce",
                            var,
                            var,
                            len(out),
                            after,
                            "applied",
                            detail,
                            rule.caveat or "",
                        )
                    )
            elif rule.kind == "derive":
                method = derive_methods.METHODS[rule.params["method"]]
                out, detail = method(out, data.cohort, **(rule.params.get("options") or {}))
                status = "skipped" if detail.startswith("skipped") else "applied"
                report.add(
                    Record(
                        data.cohort,
                        name,
                        rule.id,
                        "derive",
                        rule.params["method"],
                        ",".join(rule.params.get("outputs", [])),
                        len(out),
                        len(out) if status == "applied" else 0,
                        status,
                        detail,
                        rule.caveat or "",
                    )
                )
            elif rule.kind == "note":
                present = [v for v in rule.variables if v in out.columns]
                if present or not rule.variables:
                    report.add(
                        Record(
                            data.cohort,
                            name,
                            rule.id,
                            "note",
                            ",".join(present),
                            "",
                            len(out),
                            0,
                            "note",
                            " ".join(rule.basis.split())[:300],
                            rule.caveat or "",
                        )
                    )
        tables[name] = out
    if genomic_tables:
        if "mutations" in tables:
            tables["mutations"] = genomic.mutations_with_sample_id(tables["mutations"])
            report.add(
                Record(
                    data.cohort,
                    "mutations",
                    "genomic-sample-id",
                    "derive",
                    "Tumor_Sample_Barcode",
                    "sample_id",
                    len(tables["mutations"]),
                    len(tables["mutations"]),
                    "applied",
                    "",
                    "",
                )
            )
        src_name = "fusions" if "fusions" in data.tables else ("sv" if "sv" in data.tables else None)
        if src_name is not None:
            unified = genomic.structural_variants(data.tables[src_name], source_format=src_name)
            tables.pop("fusions", None)
            tables["sv"] = unified
            report.add(
                Record(
                    data.cohort,
                    "sv",
                    "genomic-structural-variants",
                    "rename",
                    src_name,
                    "sv",
                    len(unified),
                    len(unified),
                    "applied",
                    f"unified from {src_name} format",
                    "",
                )
            )
        if "cna" in tables:
            tables["cna_long"] = genomic.cna_to_long(tables["cna"])
            report.add(
                Record(
                    data.cohort,
                    "cna_long",
                    "genomic-cna-long",
                    "derive",
                    "cna",
                    "cna_long",
                    len(tables["cna_long"]),
                    len(tables["cna_long"]),
                    "applied",
                    "non-zero calls only",
                    "",
                )
            )
    return HarmonizedCohort(
        cohort=data.cohort,
        version=data.version,
        tables=tables,
        sources=dict(data.sources),
        synthetic=data.synthetic,
        validation=dict(data.validation),
        report=report,
        crosswalk_version=cw.version,
    )


@dataclass
class StackedData:
    """Harmonised tables from several cohorts concatenated, with provenance."""

    tables: dict[str, pd.DataFrame]
    cohorts: list[str]
    report: HarmonizationReport
    columns_by_cohort: dict[str, dict[str, set[str]]]

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def __contains__(self, name: object) -> bool:
        return name in self.tables

    @property
    def names(self) -> list[str]:
        return list(self.tables)

    def availability(self, table: str) -> pd.DataFrame:
        """Boolean matrix: which cohorts contributed each column of ``table``."""
        cols = list(self.tables[table].columns)
        data = {c: [col in self.columns_by_cohort[table].get(c, set()) for col in cols] for c in self.cohorts}
        return pd.DataFrame(data, index=cols)

    def common_columns(self, table: str) -> list[str]:
        av = self.availability(table)
        return [c for c, row in av.iterrows() if bool(row.all())]

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, frame in self.tables.items():
            per = frame.groupby("cohort").size() if "cohort" in frame.columns else pd.Series(dtype=int)
            rows.append(
                {
                    "table": name,
                    "rows": len(frame),
                    "columns": frame.shape[1],
                    "common_columns": len(self.common_columns(name)),
                    "cohorts": ",".join(str(c) for c in per.index),
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return f"StackedData({len(self.tables)} tables, cohorts={self.cohorts})"


def stack(
    items: Iterable[CohortData], tables: Iterable[str] | None = None, auto_harmonize: bool = True
) -> StackedData:
    """Concatenate harmonised cohorts table by table.

    Clinical tables already carry ``cohort``; genomic tables receive one. A
    ``schema_generation`` column is added everywhere. Columns absent from a
    cohort are filled with missing values; :meth:`StackedData.availability`
    shows which cohorts contributed each column. Plain :class:`CohortData`
    inputs are harmonised first when ``auto_harmonize`` is true.
    """
    cw = load_crosswalk()
    harmonised: list[HarmonizedCohort] = []
    for item in items:
        if isinstance(item, HarmonizedCohort):
            harmonised.append(item)
        elif auto_harmonize:
            harmonised.append(harmonize(item, cw))
        else:
            raise TypeError("stack() needs HarmonizedCohort inputs when auto_harmonize is False")
    order = {c: i for i, c in enumerate(COHORTS)}
    harmonised.sort(key=lambda h: order.get(h.cohort, len(order)))
    cohorts = [h.cohort for h in harmonised]
    if len(set(cohorts)) != len(cohorts):
        raise ValueError(f"duplicate cohorts in stack: {cohorts}")
    wanted = set(tables) if tables else None
    names: list[str] = []
    for h in harmonised:
        for n in h.names:
            if n not in names and (wanted is None or n in wanted):
                names.append(n)
    out: dict[str, pd.DataFrame] = {}
    columns_by_cohort: dict[str, dict[str, set[str]]] = {}
    report = HarmonizationReport()
    for h in harmonised:
        report.extend(h.report)
    for name in names:
        parts = []
        columns_by_cohort[name] = {}
        for h in harmonised:
            if name not in h.tables:
                continue
            frame = h.tables[name]
            if name == "cna":
                continue  # wide matrices are not stackable; cna_long is
            frame = frame.copy()
            if "cohort" not in frame.columns:
                frame.insert(0, "cohort", h.cohort)
            frame["schema_generation"] = cw.generation(h.cohort)
            columns_by_cohort[name][h.cohort] = set(frame.columns)
            parts.append(frame)
        if parts:
            # keep every column of every cohort, but let dtypes be decided by the
            # cohorts that populate a column (pandas deprecates all-NA entries in concat)
            all_columns: list[str] = []
            for part in parts:
                all_columns.extend(c for c in part.columns if c not in all_columns)
            trimmed = [part.dropna(axis=1, how="all") for part in parts]
            out[name] = pd.concat(trimmed, ignore_index=True, sort=False).reindex(columns=all_columns)
    return StackedData(tables=out, cohorts=cohorts, report=report, columns_by_cohort=columns_by_cohort)
