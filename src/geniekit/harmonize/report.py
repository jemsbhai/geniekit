"""Provenance of every harmonisation action."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = ["HarmonizationReport", "Record"]


@dataclass(frozen=True)
class Record:
    cohort: str
    table: str
    rule_id: str
    kind: str
    source: str
    target: str
    n_rows: int
    n_changed: int
    status: str  # applied | skipped | note
    detail: str = ""
    caveat: str = ""


@dataclass
class HarmonizationReport:
    records: list[Record] = field(default_factory=list)

    def add(self, record: Record) -> None:
        self.records.append(record)

    def extend(self, other: HarmonizationReport) -> None:
        self.records.extend(other.records)

    def to_frame(self) -> pd.DataFrame:
        cols = [
            "cohort",
            "table",
            "rule_id",
            "kind",
            "source",
            "target",
            "n_rows",
            "n_changed",
            "status",
            "detail",
            "caveat",
        ]
        return pd.DataFrame([r.__dict__ for r in self.records], columns=cols)

    def applied(self) -> pd.DataFrame:
        f = self.to_frame()
        return f[f["status"] == "applied"].reset_index(drop=True)

    def skipped(self) -> pd.DataFrame:
        f = self.to_frame()
        return f[f["status"] == "skipped"].reset_index(drop=True)

    def unmapped(self) -> pd.DataFrame:
        """Rows whose detail reports values the crosswalk did not know."""
        f = self.to_frame()
        return f[f["detail"].str.startswith("unmapped", na=False)].reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        f = self.to_frame()
        if f.empty:
            return pd.DataFrame(columns=["cohort", "kind", "status", "rules", "rows_changed"])
        return (
            f.groupby(["cohort", "kind", "status"], sort=False)
            .agg(rules=("rule_id", "nunique"), rows_changed=("n_changed", "sum"))
            .reset_index()
        )

    def renames(self, cohort: str | None = None) -> dict[str, str]:
        """Original column name -> harmonised column name (applied renames only)."""
        out: dict[str, str] = {}
        for r in self.records:
            if r.kind == "rename" and r.status == "applied" and (cohort is None or r.cohort == cohort):
                out[r.source] = r.target
        return out

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        f = self.to_frame()
        n_applied = int((f["status"] == "applied").sum()) if len(f) else 0
        return f"HarmonizationReport({len(self.records)} records, {n_applied} applied)"
