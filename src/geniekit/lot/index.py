"""Index events: the point from which lines of therapy are counted.

The data user picks the event. Built-in events, all expressed in days from
the cancer's own diagnosis:

- ``"diagnosis"``: day 0 for every cancer.
- ``"advanced"``: stage IV at diagnosis (day 0) or, for a stage I-III cancer,
  the first indication of distant metastasis (``dx_to_dmets_days``); cancers
  that never became advanced have no index and are left out. This is the
  origin GENIE BPC uses for its "from advanced disease" endpoints.
- ``"first_progression"``: the first radiologist or oncologist assessment of
  progressing/worsening/enlarging disease on or after ``after_days``.
- ``"first_systemic"``: the start of the first regimen with a line-defining
  agent (see :mod:`geniekit.lot.rules`), whenever it started.
- a custom table with ``record_id``, ``ca_seq`` and ``index_days`` (optionally
  ``index_source``), or a callable taking the :class:`~geniekit.io.CohortData`
  and returning one.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from geniekit.io.cohort import CohortData

from .regimens import KEYS, cancer_table, progression_events

__all__ = ["INDEX_EVENTS", "index_events"]

INDEX_EVENTS: tuple[str, ...] = ("diagnosis", "advanced", "first_progression", "first_systemic")
IndexSpec = str | pd.DataFrame | Callable[[CohortData], pd.DataFrame]


def _frame(cancers: pd.DataFrame) -> pd.DataFrame:
    out = cancers[KEYS].copy()
    out["record_id"] = out["record_id"].astype(str)
    out["index_days"] = np.nan
    out["index_source"] = pd.Series([pd.NA] * len(out), dtype="string")
    return out


def _advanced(cancers: pd.DataFrame) -> pd.DataFrame:
    out = _frame(cancers)
    stage = cancers.get("stage_dx_iv", pd.Series(pd.NA, index=cancers.index)).astype("string").str.strip()
    stage_iv = stage.eq("Stage IV").fillna(False).to_numpy()
    flag_col = next((c for c in ("dmets_post_dx", "dmets_stage_i_iii") if c in cancers.columns), None)
    flag = (
        pd.to_numeric(cancers[flag_col], errors="coerce").eq(1).fillna(False).to_numpy()
        if flag_col
        else np.zeros(len(cancers), dtype=bool)
    )
    to_mets = pd.to_numeric(cancers.get("dx_to_dmets_days"), errors="coerce").to_numpy(dtype=float)
    later = (~stage_iv) & flag & ~np.isnan(to_mets)
    out.loc[stage_iv, "index_days"] = 0.0
    out.loc[stage_iv, "index_source"] = "stage IV at diagnosis"
    out.loc[later, "index_days"] = to_mets[later]
    out.loc[later, "index_source"] = "first distant metastasis"
    return out


def _first_progression(
    data: CohortData, cancers: pd.DataFrame, after_days: float, sources: tuple[str, ...]
) -> pd.DataFrame:
    out = _frame(cancers)
    events = progression_events(data, cancers, sources=sources)
    if events.empty:
        return out
    events = events[events["days"] >= after_days]
    first = events.sort_values([*KEYS, "days"]).groupby(KEYS, as_index=False).first()
    first["record_id"] = first["record_id"].astype(str)
    merged = out.drop(columns=["index_days", "index_source"]).merge(first, on=KEYS, how="left")
    out["index_days"] = merged["days"].to_numpy(dtype=float)
    out["index_source"] = ("first progression (" + merged["source"].astype("string") + ")").to_numpy()
    return out


def _first_systemic(cancers: pd.DataFrame, regimens: pd.DataFrame, defining: pd.Series) -> pd.DataFrame:
    out = _frame(cancers)
    eligible = regimens[defining.to_numpy() & regimens["start_days"].notna().to_numpy()]
    first = eligible.groupby(KEYS, as_index=False)["start_days"].min()
    first["record_id"] = first["record_id"].astype(str)
    merged = out.drop(columns=["index_days", "index_source"]).merge(first, on=KEYS, how="left")
    out["index_days"] = merged["start_days"].to_numpy(dtype=float)
    out.loc[out["index_days"].notna(), "index_source"] = "first systemic therapy"
    return out


def _custom(data: CohortData, cancers: pd.DataFrame, spec: pd.DataFrame | Callable) -> pd.DataFrame:
    table = spec(data) if callable(spec) else spec
    if (
        not isinstance(table, pd.DataFrame)
        or "record_id" not in table.columns
        or "index_days" not in table.columns
    ):
        raise ValueError("a custom index must be a DataFrame with record_id, (ca_seq) and index_days")
    table = table.copy()
    table["record_id"] = table["record_id"].astype(str)
    if "ca_seq" not in table.columns:
        # one index per patient: attach it to every selected cancer of the patient
        table = table.merge(cancers[KEYS].assign(record_id=cancers["record_id"].astype(str)), on="record_id")
    table["ca_seq"] = pd.to_numeric(table["ca_seq"], errors="coerce").astype("Int64")
    if table.duplicated(KEYS).any():
        raise ValueError("the custom index table has more than one row for a cancer")
    out = _frame(cancers)
    cols = ["index_days", *(["index_source"] if "index_source" in table.columns else [])]
    merged = out.drop(columns=["index_days", "index_source"]).merge(
        table[[*KEYS, *cols]], on=KEYS, how="left"
    )
    out["index_days"] = pd.to_numeric(merged["index_days"], errors="coerce").to_numpy(dtype=float)
    if "index_source" in table.columns:
        out["index_source"] = merged["index_source"].astype("string").to_numpy()
    else:
        out.loc[out["index_days"].notna(), "index_source"] = "custom"
    return out


def index_events(
    data: CohortData,
    event: IndexSpec = "advanced",
    cancers: pd.DataFrame | str = "index",
    regimens: pd.DataFrame | None = None,
    defining: pd.Series | None = None,
    after_days: float = 0.0,
    progression_sources: tuple[str, ...] = ("imaging", "med_onc"),
) -> pd.DataFrame:
    """One row per selected cancer: ``record_id``, ``ca_seq``, ``index_days``, ``index_source``.

    ``index_days`` is NaN when the event did not occur for that cancer (for
    example a cancer that never became advanced); such cancers get no lines.
    ``regimens`` and ``defining`` (a boolean Series marking regimens with a
    line-defining agent) are needed for ``"first_systemic"`` only.
    """
    table = cancer_table(data, cancers) if isinstance(cancers, str) else cancers
    if isinstance(event, str):
        if event == "diagnosis":
            out = _frame(table)
            out["index_days"] = 0.0
            out["index_source"] = "diagnosis"
        elif event == "advanced":
            out = _advanced(table)
        elif event == "first_progression":
            out = _first_progression(data, table, after_days, tuple(progression_sources))
        elif event == "first_systemic":
            if regimens is None or defining is None:
                raise ValueError("first_systemic needs the annotated regimens and the defining mask")
            out = _first_systemic(table, regimens, defining)
        else:
            raise ValueError(f"unknown index event {event!r}; expected one of {INDEX_EVENTS} or a table")
        out["index_event"] = event
    else:
        out = _custom(data, table, event)
        out["index_event"] = "custom"
    out["ca_seq"] = out["ca_seq"].astype("Int64")
    return out.reset_index(drop=True)
