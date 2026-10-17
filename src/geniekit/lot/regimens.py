"""Regimen annotation and the clinical event tables the line derivation needs.

Everything is expressed in days from the diagnosis of the cancer the
regimen is associated with (``ca_seq``), which is the time origin of the
regimen table's ``dx_*`` intervals. Imaging and oncologist assessments are
recorded against the patient's first BPC Project Cancer (``dx_scan_days``,
``dx_md_visit_days``); they are moved onto a cancer's own axis with the
day-from-birth intervals (``image_scan_int``, ``md_onc_visit_int``,
``dob_ca_dx_days``), which the released files carry in every cohort.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from geniekit.io.cohort import CohortData

from .vocabulary import INVESTIGATIONAL, DrugVocabulary, load_vocabulary

__all__ = [
    "LOCAL_ROUTES",
    "PROGRESSION_LEVEL",
    "annotate_regimens",
    "cancer_table",
    "progression_events",
    "regimen_drugs",
]

LOCAL_ROUTES: tuple[str, ...] = (
    "Intravesical (bladder installation)",
    "Intraperitoneal (direct into the abdominal cavity)",
    "Intrapleural (direct into the pleural space around the lung)",
    "Intralesional (direct into a lesion)",
    "Intrathecal (by spinal catheter)",
)
PROGRESSION_LEVEL = "Progressing/Worsening/Enlarging"
KEYS = ["record_id", "ca_seq"]


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").astype(float)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def cancer_table(data: CohortData, cancers: str = "index") -> pd.DataFrame:
    """The cancer rows to derive lines for: ``"index"``, ``"non_index"`` or ``"all"``."""
    parts = []
    if cancers in ("index", "all") and "cancer_index" in data:
        parts.append(data["cancer_index"])
    if cancers in ("non_index", "all") and "cancer_non_index" in data:
        parts.append(data["cancer_non_index"])
    if cancers not in ("index", "non_index", "all"):
        raise ValueError("cancers must be 'index', 'non_index' or 'all'")
    if not parts:
        raise KeyError(f"no cancer table for cancers={cancers!r} in {data.label}")
    frame = pd.concat(parts, ignore_index=True, sort=False)
    frame = frame.drop_duplicates(KEYS)
    frame["ca_seq"] = pd.to_numeric(frame["ca_seq"], errors="coerce").astype("Int64")
    return frame.reset_index(drop=True)


def regimen_drugs(data: CohortData, vocabulary: DrugVocabulary | None = None) -> pd.DataFrame:
    """One row per regimen-drug slot with the vocabulary applied.

    Columns: the regimen keys, ``slot``, ``label`` (as released, synonyms
    removed), ``agents`` (canonical, ``;``-joined), ``drug_class``,
    ``drug_subclass``, ``lot_role``, ``known`` and the drug-level intervals
    ``drug_start_days``, ``drug_end_days`` and ``drug_end_or_last_days``.
    """
    vocab = vocabulary or load_vocabulary()
    reg = data["regimen"]
    rows = []
    for k in range(1, 6):
        col = f"drugs_drug_{k}"
        if col not in reg.columns:
            continue
        mask = reg[col].notna()
        if not mask.any():
            continue
        part = reg.loc[mask, [*KEYS, "regimen_number"]].copy()
        part["slot"] = k
        entries = [vocab.classify(v) for v in reg.loc[mask, col]]
        part["label"] = [e.label for e in entries]
        part["agents"] = [e.agent for e in entries]
        part["drug_class"] = [e.drug_class for e in entries]
        part["drug_subclass"] = [e.drug_subclass for e in entries]
        part["lot_role"] = [e.lot_role for e in entries]
        part["known"] = [e.known for e in entries]
        part["drug_start_days"] = _num(reg.loc[mask], f"dx_drug_start_int_{k}")
        part["drug_end_days"] = _num(reg.loc[mask], f"dx_drug_end_int_{k}")
        part["drug_end_or_last_days"] = _num(reg.loc[mask], f"dx_drug_end_or_lastadm_int_{k}")
        rows.append(part)
    if not rows:
        return pd.DataFrame(
            columns=[
                *KEYS,
                "regimen_number",
                "slot",
                "label",
                "agents",
                "drug_class",
                "drug_subclass",
                "lot_role",
                "known",
                "drug_start_days",
                "drug_end_days",
                "drug_end_or_last_days",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out["ca_seq"] = pd.to_numeric(out["ca_seq"], errors="coerce").astype("Int64")
    out["regimen_number"] = pd.to_numeric(out["regimen_number"], errors="coerce").astype("Int64")
    return out.sort_values([*KEYS, "regimen_number", "slot"]).reset_index(drop=True)


def annotate_regimens(
    data: CohortData,
    vocabulary: DrugVocabulary | None = None,
    local_routes: Iterable[str] = LOCAL_ROUTES,
) -> pd.DataFrame:
    """One row per regimen with its agents, roles, route and intervals.

    ``agents`` lists the canonical agents of the regimen (``;``-joined, sorted);
    ``roles`` lists the role of each agent in the same order, after the route
    override: a regimen whose ``drugs_admin`` (CRC and BLADDER only) names one
    of ``local_routes`` has every agent's role set to ``local``. Each
    "Investigational Drug" slot becomes its own agent token
    ``"Investigational Drug [regimen N]"`` so that two investigational
    regimens are never taken for the same therapy.
    """
    reg = data["regimen"].copy()
    drugs = regimen_drugs(data, vocabulary)
    local = set(local_routes)
    out = pd.DataFrame(
        {
            "cohort": data.cohort,
            "record_id": reg["record_id"].astype(str),
            "ca_seq": pd.to_numeric(reg["ca_seq"], errors="coerce").astype("Int64"),
            "regimen_number": pd.to_numeric(reg["regimen_number"], errors="coerce").astype("Int64"),
            "regimen_number_within_cancer": pd.to_numeric(
                reg.get("regimen_number_within_cancer"), errors="coerce"
            ).astype("Int64")
            if "regimen_number_within_cancer" in reg.columns
            else pd.array([pd.NA] * len(reg), dtype="Int64"),
            "index_cancer": reg["redcap_ca_index"].astype(str).str.lower().eq("yes")
            if "redcap_ca_index" in reg.columns
            else True,
            "start_days": _num(reg, "dx_reg_start_int"),
            "end_days": _num(reg, "dx_reg_end_all_int"),
            "end_any_days": _num(reg, "dx_reg_end_any_int"),
            "last_admin_days": pd.concat(
                [_num(reg, f"dx_drug_end_or_lastadm_int_{k}") for k in range(1, 6)], axis=1
            ).max(axis=1),
            "discontinued": reg["drugs_dc_ynu"].astype(str) if "drugs_dc_ynu" in reg.columns else "",
            "clinical_trial": reg["drugs_ct_yn"].astype(str).str.lower().eq("yes")
            if "drugs_ct_yn" in reg.columns
            else False,
            "route": reg["drugs_admin"]
            if "drugs_admin" in reg.columns
            else pd.Series(pd.NA, index=reg.index),
            "n_drugs": pd.to_numeric(reg.get("drugs_num"), errors="coerce").astype("Int64")
            if "drugs_num" in reg.columns
            else pd.array([pd.NA] * len(reg), dtype="Int64"),
            "regimen_drugs": reg["regimen_drugs"] if "regimen_drugs" in reg.columns else "",
        }
    )
    # end dates are only released when the regimen was discontinued
    dc = out["discontinued"].str.lower().eq("yes")
    out.loc[~dc, "end_days"] = np.nan
    # agents per regimen
    agent_lists: dict[tuple[str, int], list[tuple[str, str, str, bool]]] = {}
    inv_slots: dict[tuple[str, int], set[int]] = {}
    for row in drugs.itertuples(index=False):
        key = (str(row.record_id), int(row.regimen_number))
        for agent in str(row.agents).split(";"):
            if agent == INVESTIGATIONAL:
                agent = f"{INVESTIGATIONAL} [regimen {int(row.regimen_number)}]"
                inv_slots.setdefault(key, set()).add(int(row.slot))
            agent_lists.setdefault(key, []).append((agent, row.lot_role, row.drug_class, bool(row.known)))
    agents_col, roles_col, classes_col, n_inv, unknown_col, n_local = [], [], [], [], [], []
    for rec, num, route in zip(out["record_id"], out["regimen_number"], out["route"], strict=True):
        items = agent_lists.get((str(rec), int(num)) if not pd.isna(num) else ("", -1), [])
        items = sorted(set(items))
        is_local = isinstance(route, str) and route in local
        agents_col.append(";".join(a for a, _, _, _ in items))
        roles_col.append(";".join("local" if is_local else r for _, r, _, _ in items))
        classes_col.append(";".join(c for _, _, c, _ in items))
        n_inv.append(len(inv_slots.get((str(rec), int(num)) if not pd.isna(num) else ("", -1), ())))
        unknown_col.append(";".join(a for a, _, _, k in items if not k))
        n_local.append(int(is_local))
    out["agents"] = agents_col
    out["roles"] = roles_col
    out["classes"] = classes_col
    out["n_investigational"] = n_inv
    out["unknown_labels"] = unknown_col
    out["local_route"] = [bool(x) for x in n_local]
    # a regimen with an investigational drug has its end date masked (end = start)
    inv = out["n_investigational"] > 0
    out.loc[inv, "end_days"] = np.nan
    out["end_masked"] = inv
    order = ["start_days", "regimen_number"]
    return out.sort_values([*KEYS, *order], kind="stable").reset_index(drop=True)


def progression_events(
    data: CohortData,
    cancers: pd.DataFrame | None = None,
    sources: Iterable[str] = ("imaging", "med_onc"),
) -> pd.DataFrame:
    """Progression assessments per cancer, in days from that cancer's diagnosis.

    One row per assessment with ``record_id``, ``ca_seq``, ``days``, ``source``
    (``imaging`` for a radiologist assessment of progressing/worsening/
    enlarging, ``med_onc`` for the oncologist's assessment) and ``level``.
    Assessments are patient-level events; they are attached to every cancer of
    the patient in ``cancers`` and dated against each cancer's diagnosis.
    """
    cancers = cancer_table(data) if cancers is None else cancers
    anchors = cancers[[*KEYS, "dob_ca_dx_days"]].copy() if "dob_ca_dx_days" in cancers.columns else None
    parts = []
    specs = {
        "imaging": ("imaging", "image_overall", "image_scan_int", "dx_scan_days"),
        "med_onc": ("med_onc", "md_ca_status", "md_onc_visit_int", "dx_md_visit_days"),
    }
    for source in sources:
        if source not in specs:
            raise ValueError(f"unknown progression source {source!r}; expected imaging or med_onc")
        table, status_col, dob_col, dx_col = specs[source]
        if table not in data or status_col not in data[table].columns:
            continue
        frame = data[table]
        hit = frame[frame[status_col].astype(str) == PROGRESSION_LEVEL]
        if hit.empty:
            continue
        ev = pd.DataFrame({"record_id": hit["record_id"].astype(str)})
        ev["dob_days"] = _num(hit, dob_col).to_numpy()
        ev["dx_days"] = _num(hit, dx_col).to_numpy()
        ev["source"] = source
        ev["level"] = PROGRESSION_LEVEL
        parts.append(ev)
    if not parts:
        return pd.DataFrame(columns=[*KEYS, "days", "source", "level"])
    events = pd.concat(parts, ignore_index=True)
    keyed = cancers[KEYS].copy()
    keyed["record_id"] = keyed["record_id"].astype(str)
    merged = events.merge(keyed, on="record_id", how="inner")
    if anchors is not None:
        anchors = anchors.copy()
        anchors["record_id"] = anchors["record_id"].astype(str)
        merged = merged.merge(anchors, on=KEYS, how="left")
        days = merged["dob_days"] - pd.to_numeric(merged["dob_ca_dx_days"], errors="coerce")
        merged["days"] = days.where(days.notna(), merged["dx_days"])
    else:
        merged["days"] = merged["dx_days"]
    out = merged.dropna(subset=["days"])[[*KEYS, "days", "source", "level"]]
    out = out.drop_duplicates().sort_values([*KEYS, "days"]).reset_index(drop=True)
    return out
