"""The patient-level analytic table the aggregate functions summarise.

:func:`analytic_table` joins, per index cancer, the clinical variables the
aggregate layer needs (demographics, stage, metastasis to the liver and to
other sites), the index event and lines of therapy from :mod:`geniekit.lot`,
the first-line failure endpoint, and optionally genomic flags. It is
patient-level and stays with users who hold GENIE access; everything the
aggregate functions return is derived from it under disclosure control.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from geniekit.io.cohort import CohortData
from geniekit.lot import LineRules, LinesOfTherapy, derive
from geniekit.lot.index import IndexSpec
from geniekit.lot.regimens import KEYS, progression_events

from .genomics import gene_flags, msi_mmr_status

__all__ = ["DAYS_PER_MONTH", "analytic_table", "first_line_failure", "metastasis_sites"]

DAYS_PER_MONTH = 30.4  # the GENIE BPC convention for the *_mos variables
DAYS_PER_YEAR = 365.25


def metastasis_sites(cancers: pd.DataFrame) -> list[str]:
    """The metastasis site flags present in a cancer table (``dmets_*`` or ``dist_mets_*``)."""
    sites = []
    for col in cancers.columns:
        m = re.fullmatch(r"(?:dmets|dist_mets)_(.+)", col)
        if m and not col.startswith("dx_to_") and m.group(1) not in ("stage_i_iii", "post_dx"):
            sites.append(col)
    return sites


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").astype(float)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def first_line_failure(
    lines: LinesOfTherapy, cancers: pd.DataFrame, events: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Time from first-line start to treatment failure, per cancer with a first line.

    Failure is the earliest of: a progression assessment (imaging or
    oncologist note) after the first-line start, the start of the next line,
    and death. Cancers without an event are censored at the last follow-up
    (``tt_os_dx_days``). ``events`` are the progression assessments (from
    :func:`geniekit.lot.progression_events`); when None, the events held by
    ``lines`` are used, which are empty unless the derivation needed them.
    Returns ``record_id``, ``ca_seq``, ``first_line_start_days``,
    ``failure_days`` (from first-line start), ``failure_event`` (1 or 0) and
    ``failure_type``.
    """
    first = lines.lines[lines.lines["line_number"] == 1][[*KEYS, "start_days", "next_line_start_days"]].copy()
    first["record_id"] = first["record_id"].astype(str)
    fu = cancers[[*KEYS, "tt_os_dx_days", "os_dx_status"]].copy()
    fu["record_id"] = fu["record_id"].astype(str)
    fu["ca_seq"] = pd.to_numeric(fu["ca_seq"], errors="coerce").astype("Int64")
    first = first.merge(fu, on=KEYS, how="left")
    first["follow_up"] = pd.to_numeric(first["tt_os_dx_days"], errors="coerce")
    first["died"] = pd.to_numeric(first["os_dx_status"], errors="coerce").eq(1)
    events = lines.events if events is None else events
    prog_after: dict[tuple[str, int], float] = {}
    if len(events):
        ev = events.copy()
        ev["record_id"] = ev["record_id"].astype(str)
        starts = first.set_index(KEYS)["start_days"]
        ev = ev.merge(starts.rename("line_start"), left_on=KEYS, right_index=True, how="inner")
        ev = ev[ev["days"] > ev["line_start"]]
        prog_after = ev.groupby(KEYS)["days"].min().to_dict()
    rows = []
    for row in first.itertuples(index=False):
        key = (row.record_id, int(row.ca_seq))
        start = float(row.start_days)
        candidates: list[tuple[float, str]] = []
        prog = prog_after.get(key)
        if prog is not None and not math.isnan(prog):
            candidates.append((prog - start, "progression"))
        if not pd.isna(row.next_line_start_days):
            candidates.append((float(row.next_line_start_days) - start, "next line"))
        if bool(row.died) and not pd.isna(row.follow_up):
            candidates.append((float(row.follow_up) - start, "death"))
        if candidates:
            days, kind = min(candidates, key=lambda c: c[0])
            rows.append((row.record_id, int(row.ca_seq), start, max(days, 0.0), 1, kind))
        else:
            fu_days = float(row.follow_up) - start if not pd.isna(row.follow_up) else math.nan
            rows.append(
                (
                    row.record_id,
                    int(row.ca_seq),
                    start,
                    max(fu_days, 0.0) if not math.isnan(fu_days) else math.nan,
                    0,
                    "censored",
                )
            )
    out = pd.DataFrame(
        rows, columns=[*KEYS, "first_line_start_days", "failure_days", "failure_event", "failure_type"]
    )
    out["ca_seq"] = out["ca_seq"].astype("Int64")
    return out


def analytic_table(
    data: CohortData,
    index: IndexSpec = "advanced",
    rules: LineRules | str | None = None,
    lines: LinesOfTherapy | None = None,
    genes: Iterable[str] = (),
    variants: Mapping[str, str] | None = None,
    combined: Mapping[str, Iterable[str]] | None = None,
    msi: bool = False,
) -> pd.DataFrame:
    """One row per index cancer with the variables the aggregate layer summarises.

    Columns: identifiers (``cohort``, ``record_id``, ``ca_seq``,
    ``institution``), ``age_dx_years``, ``sex``, ``race_ethnicity`` (when the
    cohort was harmonised), ``stage_dx``, ``mets_at_dx``, the index event
    (``index_days``, ``index_source``), liver metastasis (``liver_mets``,
    ``liver_mets_days``, ``liver_mets_timing`` relative to the index, and
    ``liver_mets_at_index``), every other released site flag as
    ``mets_<site>``, lines of therapy (``n_lines``, ``first_line``,
    ``first_line_start_days``, ``first_line_days_from_index``,
    ``first_line_agents``), follow-up (``follow_up_days``, ``died``,
    ``os_from_index_days``), the first-line failure endpoint
    (``failure_days``, ``failure_event``, ``failure_type``) and, when
    requested, gene flags (``<GENE>_mut`` and the named variants and
    combinations, with ``sequenced``) and ``msi_mmr``.
    """
    lines = lines if lines is not None else derive(data, index=index, rules=rules)
    cancers = data["cancer_index"].copy()
    cancers["record_id"] = cancers["record_id"].astype(str)
    cancers["ca_seq"] = pd.to_numeric(cancers["ca_seq"], errors="coerce").astype("Int64")
    out = cancers[KEYS].copy()
    out.insert(0, "cohort", data.cohort)
    out["institution"] = (
        cancers["institution"].astype("string") if "institution" in cancers.columns else pd.NA
    )
    out["age_dx_years"] = _num(cancers, "dob_ca_dx_days") / DAYS_PER_YEAR
    if "patient" in data:
        pat = data["patient"].copy()
        pat["record_id"] = pat["record_id"].astype(str)
        cols = [c for c in ("naaccr_sex_code", "race_ethnicity") if c in pat.columns]
        merged = out[["record_id"]].merge(
            pat[["record_id", *cols]].drop_duplicates("record_id"), on="record_id", how="left"
        )
        out["sex"] = (
            merged["naaccr_sex_code"].astype("string").to_numpy() if "naaccr_sex_code" in cols else pd.NA
        )
        out["race_ethnicity"] = (
            merged["race_ethnicity"].astype("string").to_numpy() if "race_ethnicity" in cols else pd.NA
        )
    else:
        out["sex"] = pd.NA
        out["race_ethnicity"] = pd.NA
    out["stage_dx"] = cancers["stage_dx_iv"].astype("string") if "stage_dx_iv" in cancers.columns else pd.NA
    out["mets_at_dx"] = cancers["ca_dmets_yn"].astype("string") if "ca_dmets_yn" in cancers.columns else pd.NA
    # index and lines
    cs = lines.cancers.copy()
    cs["record_id"] = cs["record_id"].astype(str)
    keep = [
        "index_days",
        "index_source",
        "n_lines",
        "first_line_start_days",
        "first_line_days_from_index",
        "first_line_agents",
        "n_pre_index",
    ]
    merged = out[KEYS].merge(cs[[*KEYS, *keep]], on=KEYS, how="left")
    for col in keep:
        out[col] = merged[col].to_numpy()
    out["n_lines"] = out["n_lines"].fillna(0).astype(int)
    out["n_pre_index"] = out["n_pre_index"].fillna(0).astype(int)
    out["first_line"] = out["n_lines"] >= 1
    out["first_line_agents"] = out["first_line_agents"].fillna("")
    # liver and other metastasis sites
    liver_col = next((c for c in ("dist_mets_liver", "dmets_liver") if c in cancers.columns), None)
    liver_days_col = next(
        (c for c in ("dx_to_dist_mets_liver_days", "dx_to_dmets_liver_days") if c in cancers.columns), None
    )
    flag = _num(cancers, liver_col) if liver_col else pd.Series(np.nan, index=cancers.index)
    out["liver_mets"] = pd.array([None if math.isnan(v) else bool(v == 1) for v in flag], dtype="boolean")
    out["liver_mets_days"] = _num(cancers, liver_days_col) if liver_days_col else np.nan
    timing = []
    at_index = []
    for lm, days, idx in zip(out["liver_mets"], out["liver_mets_days"], out["index_days"], strict=True):
        if pd.isna(lm):
            timing.append("unknown")
            at_index.append(None)
        elif not lm:
            timing.append("none")
            at_index.append(False)
        elif pd.isna(days) or pd.isna(idx):
            timing.append("yes, timing unknown")
            at_index.append(None)
        elif days <= idx:
            timing.append("at or before index")
            at_index.append(True)
        else:
            timing.append("after index")
            at_index.append(False)
    out["liver_mets_timing"] = pd.array(timing, dtype="string")
    out["liver_mets_at_index"] = pd.array(at_index, dtype="boolean")
    for col in metastasis_sites(cancers):
        site = re.sub(r"^(dmets|dist_mets)_", "", col)
        vals = _num(cancers, col)
        out[f"mets_{site}"] = pd.array(
            [None if math.isnan(v) else bool(v == 1) for v in vals], dtype="boolean"
        )
    # follow-up and outcome
    out["follow_up_days"] = _num(cancers, "tt_os_dx_days")
    out["died"] = _num(cancers, "os_dx_status").eq(1)
    out["os_from_index_days"] = out["follow_up_days"] - out["index_days"]
    events = progression_events(data, cancers) if ("imaging" in data or "med_onc" in data) else None
    failure = first_line_failure(lines, cancers, events)
    merged = out[KEYS].merge(
        failure[[*KEYS, "failure_days", "failure_event", "failure_type"]], on=KEYS, how="left"
    )
    out["failure_days"] = merged["failure_days"].to_numpy()
    out["failure_event"] = merged["failure_event"].to_numpy()
    out["failure_type"] = merged["failure_type"].astype("string").to_numpy()
    # genomics
    genes = list(genes)
    if genes or variants or combined:
        flags = gene_flags(data, genes, variants=variants, combined=combined)
        merged = out[["record_id"]].merge(flags, on="record_id", how="left")
        out["sequenced"] = merged["sequenced"].notna().to_numpy() & merged["sequenced"].eq(True).to_numpy()
        out["n_samples"] = (
            pd.to_numeric(merged["n_samples"], errors="coerce").fillna(0).astype(int).to_numpy()
        )
        for col in flags.columns:
            if col in ("record_id", "sequenced", "n_samples"):
                continue
            out[col] = pd.array(merged[col].to_numpy(), dtype="boolean")
    if msi:
        status = msi_mmr_status(data)
        merged = out[["record_id"]].merge(status, on="record_id", how="left")
        out["msi_mmr"] = merged["msi_mmr"].fillna("not tested").astype("string").to_numpy()
    return out.reset_index(drop=True)
