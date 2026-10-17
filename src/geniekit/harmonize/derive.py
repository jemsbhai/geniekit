"""Derived variables referenced by ``derive`` rules in the crosswalk.

Each method takes the table frame (after the rules that precede it in the
crosswalk have been applied) and returns the modified frame plus a detail
string for the report.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

__all__ = ["METHODS", "pfs_time_origin", "race_ethnicity", "seer_summary_stage"]

SEER_SUMMARY_STAGE_LABELS: dict[int, str] = {
    0: "In situ",
    1: "Localized",
    2: "Regional, direct extension only",
    3: "Regional, regional lymph nodes only",
    4: "Regional, direct extension and regional lymph nodes",
    5: "Regional, NOS",
    7: "Distant",
    8: "Not applicable",
    9: "Unstaged",
}

_SEER_TEXT_TO_CODE: dict[str, int] = {
    "in situ": 0,
    "localized": 1,
    "local": 1,
    "regional, direct extension only": 2,
    "regional direct extension only": 2,
    "regional by direct extension only": 2,
    "regional, regional lymph nodes only": 3,
    "regional lymph nodes only": 3,
    "regional, direct extension and regional lymph nodes": 4,
    "regional by both direct extension and lymph nodes": 4,
    "regional, nos": 5,
    "regional nos": 5,
    "regional": 5,
    "distant": 7,
    "not applicable": 8,
    "unstaged": 9,
    "unknown": 9,
    "unknown if extension or metastasis": 9,
}

RACE_ETHNICITY_LEVELS: tuple[str, ...] = (
    "Hispanic/Latinx",
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "AAAPI (Asian, Asian American, and Pacific Islander)",
    "Other",
    "Unknown",
)

_HISPANIC = {"spanish/hispanic"}
_NON_HISPANIC = {"non-spanish/non-hispanic"}
_RACE_TO_GROUP = {
    "white": "Non-Hispanic White",
    "black": "Non-Hispanic Black",
    "asian": "AAAPI (Asian, Asian American, and Pacific Islander)",
    "pacific islander": "AAAPI (Asian, Asian American, and Pacific Islander)",
    "native american": "Other",
    "other": "Other",
}

PFS_ORIGIN_DIAGNOSIS = "diagnosis"
PFS_ORIGIN_FIRST_METASTASIS = "first distant metastasis"


def _norm(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "<na>"):
        return None
    return " ".join(text.split()).lower()


def seer_summary_stage(frame: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, str]:
    """Normalise ``naaccr_seer_sum_stage`` to the NAACCR #759 code plus a label column."""
    col = "naaccr_seer_sum_stage"
    if col not in frame.columns:
        return frame, "skipped: column absent"
    if f"{col}_label" in frame.columns:
        return frame, "skipped: already derived"
    raw = frame[col]
    codes: list[object] = []
    unmatched: list[str] = []
    for value in raw:
        n = _norm(value)
        if n is None:
            codes.append(pd.NA)
            continue
        try:
            number = float(n)
        except ValueError:
            number = None
        if number is not None and number.is_integer() and int(number) in SEER_SUMMARY_STAGE_LABELS:
            codes.append(int(number))
            continue
        # "1 = Localized" style text
        head = n.split("=")[0].strip()
        if head.isdigit() and int(head) in SEER_SUMMARY_STAGE_LABELS:
            codes.append(int(head))
            continue
        if n in _SEER_TEXT_TO_CODE:
            codes.append(_SEER_TEXT_TO_CODE[n])
            continue
        codes.append(pd.NA)
        unmatched.append(str(value))
    code_series = pd.array(codes, dtype="Int64")
    out = frame.copy()
    pos = out.columns.get_loc(col)
    if unmatched:
        out.insert(
            pos + 1, f"{col}_detail", raw.astype(object).where(pd.Series(codes, index=raw.index).isna(), None)
        )
    out[col] = code_series
    labels = pd.Series(code_series, index=out.index).map(
        lambda c: SEER_SUMMARY_STAGE_LABELS.get(int(c)) if pd.notna(c) else None
    )
    out.insert(pos + 1, f"{col}_label", labels.astype(object))
    detail = f"{int(pd.notna(code_series).sum())} coded"
    if unmatched:
        detail += f"; unmapped {len(unmatched)} value(s): " + ", ".join(sorted(set(unmatched))[:8])
    return out, detail


def race_ethnicity(frame: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, str]:
    """Derive a six-level race/ethnicity from the collapsed NAACCR codes.

    Must run after the ethnicity and race collapses so that generation-1 codes
    are already in the generation-2 vocabulary.
    """
    eth_col, race_col, out_col = "naaccr_ethnicity_code", "naaccr_race_code_primary", "race_ethnicity"
    if eth_col not in frame.columns or race_col not in frame.columns:
        return frame, "skipped: NAACCR columns absent"
    if f"{out_col}_released" in frame.columns or (
        out_col in frame.columns
        and set(frame[out_col].dropna()) <= set(RACE_ETHNICITY_LEVELS)
        and "naaccr_ethnicity_code_detail" in frame.columns
    ):
        return frame, "skipped: already derived"
    out = frame.copy()
    released = None
    if out_col in out.columns:
        released = out[out_col].copy()
        out = out.rename(columns={out_col: f"{out_col}_released"})
    derived: list[str] = []
    for eth, race in zip(out[eth_col], out[race_col], strict=True):
        e, r = _norm(eth), _norm(race)
        if e in _HISPANIC:
            derived.append("Hispanic/Latinx")
        elif r in _RACE_TO_GROUP:
            derived.append(_RACE_TO_GROUP[r])
        else:
            derived.append("Unknown")
    pos = out.columns.get_loc(race_col) + 1
    out.insert(pos, out_col, pd.array(derived, dtype=object))
    detail = "derived for all rows"
    if released is not None:
        comparable = released.notna()
        rel = (
            released.astype(object)
            .where(comparable, None)
            .map(
                lambda v: "AAAPI (Asian, Asian American, and Pacific Islander)"
                if _norm(v) == "asian indian or pakistani nos"
                else v
            )
        )
        agree = int(
            (rel[comparable].to_numpy() == np.array(derived, dtype=object)[comparable.to_numpy()]).sum()
        )
        detail += f"; agreement with released value {agree}/{int(comparable.sum())}"
    return out, detail


def pfs_time_origin(frame: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, str]:
    """Add ``pfs_cohort_detail`` (where available) and ``pfs_time_origin``."""
    col = "pfs_cohort"
    if col not in frame.columns:
        return frame, "skipped: column absent"
    if "pfs_time_origin" in frame.columns:
        return frame, "skipped: already derived"
    out = frame.copy()
    detail_col = f"{col}_detail"
    coarse = out[col]
    if detail_col not in out.columns:
        # cohorts without the stage IV split: detail known only for stage I-III rows
        detail_vals = coarse.astype(object).where(coarse.map(_norm) == "stage i-iii with distant mets", None)
        out.insert(out.columns.get_loc(col) + 1, detail_col, detail_vals)
    origin: list[object] = []
    for value, det in zip(out[col], out[detail_col], strict=True):
        n, d = _norm(value), _norm(det)
        if d == "stage iv with distant mets at diagnosis":
            origin.append(PFS_ORIGIN_DIAGNOSIS)
        elif d in ("stage iv with distant mets post-diagnosis", "stage i-iii with distant mets"):
            origin.append(PFS_ORIGIN_FIRST_METASTASIS)
        elif n == "stage iv":
            origin.append(PFS_ORIGIN_DIAGNOSIS)
        elif n == "stage i-iii with distant mets":
            origin.append(PFS_ORIGIN_FIRST_METASTASIS)
        else:
            origin.append(None)
    out.insert(out.columns.get_loc(detail_col) + 1, "pfs_time_origin", pd.array(origin, dtype=object))
    n_set = sum(o is not None for o in origin)
    return out, f"{n_set} rows with a PFS time origin"


METHODS: dict[str, Callable[[pd.DataFrame, str], tuple[pd.DataFrame, str]]] = {
    "seer_summary_stage": seer_summary_stage,
    "race_ethnicity": race_ethnicity,
    "pfs_time_origin": pfs_time_origin,
}
