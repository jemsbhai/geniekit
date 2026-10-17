"""Machine-readable data dictionary for the six public GENIE BPC cohorts.

AACR publishes a Variable Synopsis workbook with every public BPC release. The
six workbooks are compiled by ``data-raw/build_dictionary.py`` into one long
table shipped as ``resources/dictionary.csv.gz``, with one row per
(cohort, table, variable). This module exposes that table and the
cross-cohort comparisons that motivate the harmonisation rules:

* which variables exist in which cohorts (``presence``),
* variables whose declared type differs across cohorts,
* variables whose permitted value sets differ across cohorts.

The dictionary contains no patient-level data.

Example
-------
>>> from geniekit import dictionary
>>> dd = dictionary.load()
>>> dd.compare("ca_cigarette", table="cancer").present_in
('BrCa', 'PANC', 'Prostate', 'BLADDER')
>>> dd.differences(table="patient").kind.value_counts()  # doctest: +SKIP
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources as importlib_resources

import pandas as pd

from .releases import COHORTS, normalize_cohort

__all__ = [
    "TABLES",
    "DataDictionary",
    "VariableComparison",
    "load",
    "normalize_level",
]

#: Logical table names in the order they appear in the synopses.
TABLES: tuple[str, ...] = (
    "patient",
    "cancer",
    "regimen",
    "imaging",
    "pathology",
    "med_onc",
    "tumor_marker",
    "radiation",
    "cpt",
)

# hyphen, non-breaking hyphen, figure dash, en dash, em dash, minus sign
_DASH_CHARS = "".join(chr(c) for c in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212))
_DASHES = re.compile("[" + re.escape(_DASH_CHARS) + "]")
_WS = re.compile(r"\s+")


def normalize_level(level: str) -> str:
    """Cosmetic normalisation of a permitted value for set comparison.

    Lower-cases, collapses whitespace, unifies dash characters and strips
    trailing punctuation so that ``"Mixed Gliomas"`` and ``"Mixed gliomas"``
    compare equal while genuinely different levels do not.
    """
    s = _DASHES.sub("-", str(level))
    s = _WS.sub(" ", s).strip().lower()
    s = s.replace(" - ", "-").replace("- ", "-").replace(" -", "-")
    return s.rstrip(".;, ")


@dataclass(frozen=True)
class VariableComparison:
    """How one variable is declared across the cohorts."""

    table: str
    variable: str
    present_in: tuple[str, ...]
    missing_in: tuple[str, ...]
    labels: dict[str, str]
    dtypes: dict[str, str]
    values: dict[str, list[str]]

    @property
    def is_universal(self) -> bool:
        return not self.missing_in

    @property
    def dtype_conflict(self) -> bool:
        return len(set(self.dtypes.values())) > 1

    @property
    def label_conflict(self) -> bool:
        return len({normalize_level(v) for v in self.labels.values()}) > 1

    @property
    def value_set_conflict(self) -> bool:
        """True when the permitted value sets differ after cosmetic normalisation."""
        sets = {frozenset(normalize_level(v) for v in vals) for vals in self.values.values() if vals}
        return len(sets) > 1

    @property
    def value_text_conflict(self) -> bool:
        """True when value sets are equal after normalisation but differ as raw text."""
        if self.value_set_conflict:
            return False
        raw = {tuple(sorted(vals)) for vals in self.values.values() if vals}
        return len(raw) > 1

    @property
    def union_values(self) -> list[str]:
        seen: dict[str, str] = {}
        for vals in self.values.values():
            for v in vals:
                seen.setdefault(normalize_level(v), v)
        return list(seen.values())

    def summary(self) -> str:
        parts = [f"{self.table}.{self.variable}: present in {', '.join(self.present_in) or 'none'}"]
        if self.missing_in:
            parts.append(f"missing in {', '.join(self.missing_in)}")
        if self.dtype_conflict:
            parts.append("dtype conflict " + ", ".join(f"{c}={t}" for c, t in self.dtypes.items()))
        if self.value_set_conflict:
            parts.append("value sets differ")
        elif self.value_text_conflict:
            parts.append("value text differs (cosmetic)")
        return "; ".join(parts)


class DataDictionary:
    """The compiled dictionary with cross-cohort comparison helpers."""

    def __init__(self, frame: pd.DataFrame) -> None:
        required = {"cohort", "version", "schema_generation", "table", "variable", "label", "dtype", "values"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"dictionary frame lacks columns: {sorted(missing)}")
        frame = frame.copy()
        if len(frame) and not isinstance(frame["values"].iloc[0], list):
            frame["values"] = [json.loads(v) if isinstance(v, str) and v else [] for v in frame["values"]]
        self._frame = frame.reset_index(drop=True)
        present = [c for c in COHORTS if c in set(self._frame["cohort"])]
        self.cohorts: tuple[str, ...] = tuple(present)

    # -- basic access --------------------------------------------------------
    @property
    def frame(self) -> pd.DataFrame:
        """The full long table (one row per cohort, table, variable)."""
        return self._frame

    def tables(self) -> list[str]:
        return [t for t in TABLES if t in set(self._frame["table"])]

    def schema_generation(self, cohort: str) -> int:
        c = normalize_cohort(cohort)
        gens = self._frame.loc[self._frame["cohort"] == c, "schema_generation"].unique()
        if len(gens) != 1:
            raise KeyError(f"cohort {c!r} not in dictionary")
        return int(gens[0])

    def variables(self, cohort: str | None = None, table: str | None = None) -> pd.DataFrame:
        """Rows filtered by cohort and/or table."""
        f = self._frame
        if cohort is not None:
            f = f[f["cohort"] == normalize_cohort(cohort)]
        if table is not None:
            self._check_table(table)
            f = f[f["table"] == table]
        return f.reset_index(drop=True)

    def lookup(self, variable: str, table: str | None = None) -> pd.DataFrame:
        """All declarations of ``variable`` (across cohorts and, if unspecified, tables)."""
        f = self._frame[self._frame["variable"] == variable]
        if table is not None:
            self._check_table(table)
            f = f[f["table"] == table]
        return f.reset_index(drop=True)

    def values(self, variable: str, table: str) -> dict[str, list[str]]:
        """Permitted values per cohort for one variable."""
        rows = self.lookup(variable, table)
        return {c: list(v) for c, v in zip(rows["cohort"], rows["values"], strict=True)}

    # -- comparisons ---------------------------------------------------------
    def presence(self, table: str | None = None, cohorts: Iterable[str] | None = None) -> pd.DataFrame:
        """Boolean matrix: rows (table, variable), columns cohorts."""
        f = self.variables(table=table)
        cols = tuple(normalize_cohort(c) for c in cohorts) if cohorts else self.cohorts
        f = f[f["cohort"].isin(cols)]
        mat = f.assign(flag=True).pivot_table(
            index=["table", "variable"], columns="cohort", values="flag", aggfunc="any", fill_value=False
        )
        mat = mat.reindex(columns=list(cols), fill_value=False).astype(bool)
        mat.columns.name = None
        # keep synopsis order of first appearance
        order = f.drop_duplicates(["table", "variable"])[["table", "variable"]]
        return mat.reindex(pd.MultiIndex.from_frame(order))

    def common_variables(self, table: str, cohorts: Iterable[str] | None = None) -> list[str]:
        """Variables present in every requested cohort for ``table``."""
        mat = self.presence(table, cohorts)
        return [v for (_, v), row in mat.iterrows() if bool(row.all())]

    def compare(self, variable: str, table: str) -> VariableComparison:
        rows = self.lookup(variable, table)
        if rows.empty:
            raise KeyError(f"{table}.{variable} is not declared in any cohort")
        by = {c: r for c, r in zip(rows["cohort"], rows.to_dict("records"), strict=True)}
        present = tuple(c for c in self.cohorts if c in by)
        return VariableComparison(
            table=table,
            variable=variable,
            present_in=present,
            missing_in=tuple(c for c in self.cohorts if c not in by),
            labels={c: str(by[c]["label"]) for c in present},
            dtypes={c: str(by[c]["dtype"]) for c in present},
            values={c: list(by[c]["values"]) for c in present},
        )

    def differences(self, table: str | None = None, kinds: Iterable[str] | None = None) -> pd.DataFrame:
        """Every variable that is not declared identically in all cohorts.

        Returns one row per (table, variable, kind) with ``kind`` in
        ``{"missing_in_some", "dtype", "value_set", "value_text", "label"}``
        and a human-readable ``detail`` column.
        """
        wanted = set(kinds) if kinds else {"missing_in_some", "dtype", "value_set", "value_text", "label"}
        f = self.variables(table=table)
        rows: list[dict] = []
        for (tbl, var), _ in f.groupby(["table", "variable"], sort=False):
            cmp = self.compare(var, tbl)
            if "missing_in_some" in wanted and cmp.missing_in:
                rows.append(
                    {
                        "table": tbl,
                        "variable": var,
                        "kind": "missing_in_some",
                        "present_in": ",".join(cmp.present_in),
                        "detail": f"missing in {', '.join(cmp.missing_in)}",
                    }
                )
            if "dtype" in wanted and cmp.dtype_conflict:
                rows.append(
                    {
                        "table": tbl,
                        "variable": var,
                        "kind": "dtype",
                        "present_in": ",".join(cmp.present_in),
                        "detail": ", ".join(f"{c}={t}" for c, t in cmp.dtypes.items()),
                    }
                )
            if "value_set" in wanted and cmp.value_set_conflict:
                union = {normalize_level(v) for vals in cmp.values.values() for v in vals}
                lacks = {
                    c: sorted(union - {normalize_level(v) for v in vals})
                    for c, vals in cmp.values.items()
                    if vals
                }
                rows.append(
                    {
                        "table": tbl,
                        "variable": var,
                        "kind": "value_set",
                        "present_in": ",".join(cmp.present_in),
                        "detail": "; ".join(
                            f"{c} lacks {len(missing)}" for c, missing in lacks.items() if missing
                        ),
                    }
                )
            if "value_text" in wanted and cmp.value_text_conflict:
                rows.append(
                    {
                        "table": tbl,
                        "variable": var,
                        "kind": "value_text",
                        "present_in": ",".join(cmp.present_in),
                        "detail": "same levels, different spelling/case",
                    }
                )
            if "label" in wanted and cmp.label_conflict:
                rows.append(
                    {
                        "table": tbl,
                        "variable": var,
                        "kind": "label",
                        "present_in": ",".join(cmp.present_in),
                        "detail": " | ".join(sorted({v for v in cmp.labels.values()}))[:200],
                    }
                )
        return pd.DataFrame(rows, columns=["table", "variable", "kind", "present_in", "detail"])

    # -- helpers -------------------------------------------------------------
    def _check_table(self, table: str) -> None:
        if table not in TABLES:
            raise ValueError(f"Unknown table {table!r}; expected one of {', '.join(TABLES)}")

    def __repr__(self) -> str:
        n_vars = self._frame.drop_duplicates(["table", "variable"]).shape[0]
        return f"DataDictionary({len(self._frame)} rows, {n_vars} variables, cohorts={list(self.cohorts)})"


@lru_cache(maxsize=1)
def load() -> DataDictionary:
    """Load the packaged dictionary (cached)."""
    path = importlib_resources.files("geniekit.resources").joinpath("dictionary.csv.gz")
    with path.open("rb") as fh:
        frame = pd.read_csv(fh, compression="gzip", dtype=str, keep_default_na=False)
    frame["schema_generation"] = frame["schema_generation"].astype(int)
    frame["n_values"] = frame["n_values"].astype(int)
    frame["order"] = frame["order"].astype(int)
    return DataDictionary(frame)
