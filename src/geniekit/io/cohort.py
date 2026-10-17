"""The :class:`CohortData` container: one BPC release loaded into memory."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..releases import Release, get_release, normalize_cohort

__all__ = ["CLINICAL_TABLES", "GENOMIC_TABLES", "CohortData"]

#: Clinical tables in the order the data guides present them.
CLINICAL_TABLES: tuple[str, ...] = (
    "patient",
    "cancer_index",
    "cancer_non_index",
    "regimen",
    "imaging",
    "pathology",
    "med_onc",
    "tumor_marker",
    "radiation",
    "cpt",
)

#: Genomic tables (cBioPortal-format files).
GENOMIC_TABLES: tuple[str, ...] = (
    "mutations",
    "fusions",
    "sv",
    "cna",
    "cna_seg",
    "gene_matrix",
    "genomic_information",
    "tmb",
)


def _normalize_version(version: str | None) -> str | None:
    if version is None:
        return None
    v = str(version).strip()
    if not v:
        return None
    return v if v.startswith("v") else f"v{v}"


@dataclass
class CohortData:
    """One cohort release as a dictionary of pandas frames.

    ``tables`` maps logical table names (``"patient"``, ``"cancer_index"``,
    ``"regimen"``, ``"mutations"``, ...) to frames exactly as read from the
    release files; no harmonisation has been applied. ``sources`` records where
    each table came from (a path or a Synapse id). ``synthetic`` marks data
    produced by :mod:`geniekit.synthetic`.
    """

    cohort: str
    version: str | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    synthetic: bool = False
    validation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cohort = normalize_cohort(self.cohort)
        self.version = _normalize_version(self.version)

    # -- mapping protocol ----------------------------------------------------
    def __getitem__(self, name: str) -> pd.DataFrame:
        try:
            return self.tables[name]
        except KeyError:
            raise KeyError(
                f"{self.label} has no table {name!r}; available: {', '.join(self.names)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self.tables

    def __iter__(self) -> Iterator[str]:
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)

    def get(self, name: str, default: pd.DataFrame | None = None) -> pd.DataFrame | None:
        return self.tables.get(name, default)

    # -- properties ----------------------------------------------------------
    @property
    def label(self) -> str:
        return f"{self.cohort} {self.version}" if self.version else self.cohort

    @property
    def names(self) -> list[str]:
        """Table names in canonical order (clinical, then genomic, then others)."""
        known = [*CLINICAL_TABLES, *GENOMIC_TABLES]
        ordered = [n for n in known if n in self.tables]
        return ordered + sorted(n for n in self.tables if n not in known)

    @property
    def clinical(self) -> dict[str, pd.DataFrame]:
        return {n: self.tables[n] for n in CLINICAL_TABLES if n in self.tables}

    @property
    def genomic(self) -> dict[str, pd.DataFrame]:
        return {n: self.tables[n] for n in GENOMIC_TABLES if n in self.tables}

    @property
    def release(self) -> Release | None:
        """The registry entry for this cohort/version, if it is a known public release."""
        try:
            return get_release(self.cohort, self.version)
        except KeyError:
            return None

    def patient_ids(self) -> pd.Index:
        """Distinct ``record_id`` values, from the patient table when present."""
        if "patient" in self.tables and "record_id" in self.tables["patient"].columns:
            return pd.Index(self.tables["patient"]["record_id"].astype(str).unique())
        ids: set[str] = set()
        for frame in self.clinical.values():
            if "record_id" in frame.columns:
                ids.update(frame["record_id"].astype(str).unique())
        return pd.Index(sorted(ids))

    def summary(self) -> pd.DataFrame:
        """One row per table: rows, columns, number of patients, source."""
        rows = []
        for name in self.names:
            frame = self.tables[name]
            n_pat = frame["record_id"].nunique() if "record_id" in frame.columns else None
            rows.append(
                {
                    "table": name,
                    "rows": len(frame),
                    "columns": frame.shape[1],
                    "patients": n_pat,
                    "source": self.sources.get(name, ""),
                }
            )
        return pd.DataFrame(rows, columns=["table", "rows", "columns", "patients", "source"])

    def to_dir(self, path: str | Path, compress: bool = False) -> Path:
        """Write every table as CSV (``name.csv`` or ``name.csv.gz``) into ``path``."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in self.tables.items():
            target = out / (f"{name}.csv.gz" if compress else f"{name}.csv")
            frame.to_csv(target, index=False)
        return out

    def __repr__(self) -> str:
        flag = ", synthetic" if self.synthetic else ""
        return f"CohortData({self.label}: {len(self.tables)} tables{flag})"
