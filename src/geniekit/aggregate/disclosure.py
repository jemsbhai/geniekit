"""Disclosure control for aggregate output.

Every table the aggregate layer returns passes through
:meth:`DisclosureControl.apply`, which replaces counts below the minimum
cell size with a marker, suppresses a second cell in any row or column of a
contingency table that has exactly one suppressed cell (so that the hidden
value cannot be recovered from the margin), hides percentages and other
statistics derived from suppressed counts, and records what was suppressed.

The default minimum cell size is 5, the threshold most registries and
statistical offices use for small-cell suppression; some US programmes use
11. Set it to suit the data use agreement the output is released under.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["DisclosureControl", "SafeTable", "SuppressionRecord"]

DEFAULT_MIN_COUNT = 5


@dataclass(frozen=True)
class SuppressionRecord:
    row: Any
    column: Any
    kind: str  # "primary" (below the minimum) or "complementary" (protects another cell)


@dataclass
class SafeTable:
    """An aggregate table after disclosure control.

    ``frame`` is the masked table (object dtype where cells were replaced
    by the marker); ``suppressed`` lists every suppressed cell;
    ``n`` is the number of records the table was computed from and
    ``notes`` carries the definitions the reader needs.
    """

    name: str
    frame: pd.DataFrame
    n: int
    suppressed: list[SuppressionRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    omitted: dict[str, list[tuple[Any, Any]]] = field(default_factory=lambda: {"rows": [], "columns": []})

    @property
    def n_suppressed(self) -> int:
        return len(self.suppressed)

    @property
    def n_omitted(self) -> int:
        return len(self.omitted.get("rows", [])) + len(self.omitted.get("columns", []))

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        self.frame.to_csv(path)
        return path

    def __repr__(self) -> str:
        shape = f"{self.frame.shape[0]}x{self.frame.shape[1]}"
        return f"SafeTable({self.name!r}, {shape}, n={self.n}, suppressed={self.n_suppressed})"


@dataclass(frozen=True)
class DisclosureControl:
    """Rules applied to every output table.

    Parameters
    ----------
    min_count:
        Counts below this are suppressed (``0`` is never suppressed: an empty
        cell reveals nothing about an individual, and suppressing it would
        hide the structure of the table).
    complementary:
        Apply complementary suppression to contingency tables: when a row or
        column contains exactly one suppressed cell and a margin, the next
        smallest cell in it is suppressed too, repeatedly until stable.
    marker:
        The text written in place of a suppressed value.
    min_denominator:
        Percentages, proportions and quantiles are shown only when their
        denominator (the number of records they summarise) reaches this
        value; defaults to ``min_count``.
    """

    min_count: int = DEFAULT_MIN_COUNT
    complementary: bool = True
    marker: str = "<min"
    min_denominator: int | None = None
    suppress_zero: bool = False

    def __post_init__(self) -> None:
        if self.min_count < 1:
            raise ValueError("min_count must be at least 1")
        if self.min_denominator is not None and self.min_denominator < 1:
            raise ValueError("min_denominator must be at least 1")

    @property
    def denominator(self) -> int:
        return self.min_count if self.min_denominator is None else self.min_denominator

    @property
    def label(self) -> str:
        return f"<{self.min_count}" if self.marker == "<min" else self.marker

    def describe(self) -> dict[str, Any]:
        return {
            "min_count": self.min_count,
            "complementary": self.complementary,
            "marker": self.label,
            "min_denominator": self.denominator,
            "suppress_zero": self.suppress_zero,
        }

    # -- core ------------------------------------------------------------------
    def small(self, counts: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Boolean mask of cells that must be suppressed on their own account."""
        values = pd.to_numeric(
            counts.stack() if isinstance(counts, pd.DataFrame) else counts, errors="coerce"
        )
        mask = (values < self.min_count) & values.notna()
        if not self.suppress_zero:
            mask &= values != 0
        return mask.unstack() if isinstance(counts, pd.DataFrame) else mask

    def mask_counts(
        self, counts: pd.DataFrame, total_row: bool = False, total_col: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[SuppressionRecord], dict[str, list[tuple[Any, Any]]]]:
        """Suppress a table of counts.

        ``total_row`` and ``total_col`` say whether the last row and column
        hold totals. Returns the masked table (object dtype, totals
        recomputed over the lines that remain), the boolean suppression mask,
        the suppression records, and the lines omitted
        (``{"rows": [(level, count), ...], "columns": [...]}``).

        Policy, applied until no small cell remains: a cell with a count
        below the minimum (and above zero) must be hidden. Hiding it by
        suppression also requires a second suppressed cell in every line
        (row or column) that has a total, otherwise the hidden value follows
        from the total; this complementary suppression can cascade through a
        table. The alternative is to omit the whole row or column that holds
        the small cell. The method computes the records lost by each option
        (the sum of the further suppressed cells, or the omitted line's
        total) and takes the cheaper one, so a tiny category is dropped and
        reported rather than blanking the large cells next to it.
        """
        numeric = counts.apply(pd.to_numeric, errors="coerce")
        body = numeric.iloc[
            : len(numeric.index) - (1 if total_row else 0), : len(numeric.columns) - (1 if total_col else 0)
        ]
        body = body.astype(float)
        omitted: dict[str, list[tuple[Any, Any]]] = {"rows": [], "columns": []}
        while True:
            small = self.small(body).to_numpy() if body.size else np.zeros(body.shape, dtype=bool)
            if not small.any():
                mask_body = np.zeros(body.shape, dtype=bool)
                records: list[SuppressionRecord] = []
                break
            row_tot = body.sum(axis=1)
            col_tot = body.sum(axis=0)
            rows_with = [i for i in range(body.shape[0]) if small[i, :].any()]
            cols_with = [j for j in range(body.shape[1]) if small[:, j].any()]
            drop_options: list[tuple[float, str, int]] = []
            if total_col or body.shape[1] == 1:
                drop_options += [(float(row_tot.iloc[i]), "row", i) for i in rows_with]
            if total_row or body.shape[0] == 1:
                drop_options += [(float(col_tot.iloc[j]), "col", j) for j in cols_with]
            primary = small.copy()
            comp_mask, comp_records = self._complementary_plan(body, primary, total_row, total_col)
            comp_loss = float(body.to_numpy()[comp_mask & ~primary].sum())
            best_drop = min(drop_options, default=None)
            if best_drop is not None and best_drop[0] < comp_loss:
                loss, kind, k = best_drop
                if kind == "row":
                    omitted["rows"].append((body.index[k], int(loss)))
                    body = body.drop(index=body.index[k])
                else:
                    omitted["columns"].append((body.columns[k], int(loss)))
                    body = body.drop(columns=body.columns[k])
                continue
            mask_body = comp_mask
            rows_i, cols_i = np.where(primary)
            records = [
                SuppressionRecord(body.index[r], body.columns[c], "primary")
                for r, c in zip(rows_i, cols_i, strict=True)
            ]
            records += comp_records
            break
        # rebuild the table with recomputed totals
        out = body.copy()
        mask = pd.DataFrame(mask_body, index=body.index, columns=body.columns)
        if total_col:
            out[numeric.columns[-1]] = body.sum(axis=1)
            mask[numeric.columns[-1]] = False
        if total_row:
            out.loc[numeric.index[-1]] = out.sum(axis=0)
            mask.loc[numeric.index[-1]] = False
        out = out.astype(float)
        masked = out.astype(object)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                v = out.iat[i, j]
                masked.iat[i, j] = self.label if mask.iat[i, j] else (int(v) if float(v).is_integer() else v)
        return masked, mask, records, omitted

    def _complementary_plan(
        self, body: pd.DataFrame, primary: np.ndarray, total_row: bool, total_col: bool
    ) -> tuple[np.ndarray, list[SuppressionRecord]]:
        """Mask and records after complementary suppression of ``body`` given its primary mask."""
        values = body.to_numpy(dtype=float)
        m = primary.copy()
        records: list[SuppressionRecord] = []
        if not self.complementary or not (total_row or total_col):
            return m, records
        n_rows, n_cols = body.shape
        changed = True
        while changed:
            changed = False
            lines: list[tuple[int, int]] = []
            if total_row:
                lines += [(1, j) for j in range(n_cols)]  # columns are protected against the total row
            if total_col:
                lines += [(0, i) for i in range(n_rows)]  # rows against the total column
            for axis, k in lines:
                cells = [(i, k) for i in range(n_rows)] if axis == 1 else [(k, j) for j in range(n_cols)]
                if sum(1 for c in cells if m[c]) != 1:
                    continue
                candidates = [c for c in cells if not m[c] and values[c] > 0]
                if not candidates:
                    continue

                # prefer a cell whose other line already holds a suppressed cell, then the smallest
                def _key(c: tuple[int, int], axis: int = axis) -> tuple[int, float]:
                    other = m[:, c[1]] if axis == 0 else m[c[0], :]
                    return (0 if other.any() else 1, values[c])

                pick = min(candidates, key=_key)
                m[pick] = True
                records.append(SuppressionRecord(body.index[pick[0]], body.columns[pick[1]], "complementary"))
                changed = True
        return m, records

    def mask_derived(self, derived: pd.DataFrame, count_mask: pd.DataFrame) -> pd.DataFrame:
        """Blank out statistics computed from suppressed counts (same shape as ``count_mask``)."""
        out = derived.astype(object)
        out[count_mask.to_numpy()] = self.label
        return out

    def percent(self, count: float, denominator: float, decimals: int = 1) -> object:
        """A percentage, or the marker when the count or denominator is too small to show."""
        if pd.isna(count) or pd.isna(denominator) or denominator <= 0:
            return np.nan
        if denominator < self.denominator:
            return self.label
        if count < self.min_count and (count != 0 or self.suppress_zero):
            return self.label
        return round(100.0 * count / denominator, decimals)

    def allows(self, n: float) -> bool:
        """Whether a statistic over ``n`` records may be shown."""
        return not pd.isna(n) and n >= self.denominator
