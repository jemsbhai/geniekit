from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geniekit import io, lot, synthetic
from geniekit.io.cohort import CohortData
from geniekit.lot import engine, vocabulary

FIXTURE = Path(__file__).parent / "fixtures" / "nsclc_sim"


# ----------------------------------------------------------------------------- helpers
def _regimen(record, ca_seq, number, drugs, start, end=None, dc="Yes", route=None, ct="No"):
    """One regimen row in the released layout (labels with synonym lists)."""
    row = {
        "cohort": "CRC",
        "record_id": record,
        "institution": "DFCI",
        "ca_seq": ca_seq,
        "regimen_number": number,
        "regimen_number_within_cancer": number,
        "redcap_ca_index": "Yes",
        "drugs_num": len(drugs),
        "drugs_ct_yn": ct,
        "drugs_dc_ynu": dc,
        "regimen_drugs": ", ".join(vocabulary.base_label(d) for d in drugs),
        "dx_reg_start_int": start,
        "dx_reg_end_any_int": end,
        "dx_reg_end_all_int": end,
        "drugs_admin": route,
    }
    for k in range(1, 6):
        d = drugs[k - 1] if k <= len(drugs) else None
        row[f"drugs_drug_{k}"] = d
        row[f"dx_drug_start_int_{k}"] = start if d else None
        row[f"dx_drug_end_int_{k}"] = end if d else None
        row[f"dx_drug_end_or_lastadm_int_{k}"] = (end if end is not None else start + 30) if d else None
    return row


FOLFOX = ["Fluorouracil(5FU,Adrucil)", "Leucovorin Calcium(Wellcovorin)", "Oxaliplatin(Eloxatin)"]
BEV = "Bevacizumab(Avastin)"
FOLFIRI = [
    "Fluorouracil(5FU,Adrucil)",
    "Leucovorin Calcium(Wellcovorin)",
    "Irinotecan Hydrochloride(Camptosar)",
]
INV = "Investigational Drug"


def _cohort(regimens, cancers, imaging=None, med_onc=None):
    reg = pd.DataFrame(regimens)
    tables = {"regimen": reg, "cancer_index": pd.DataFrame(cancers)}
    if imaging is not None:
        tables["imaging"] = pd.DataFrame(imaging)
    if med_onc is not None:
        tables["med_onc"] = pd.DataFrame(med_onc)
    return CohortData(cohort="CRC", version="v2.0-public", tables=tables, synthetic=True)


def _cancer(record, ca_seq, stage="Stage IV", dmets=None, to_mets=None, os_days=1000, dead=0, dob_dx=20000):
    return {
        "record_id": record,
        "ca_seq": ca_seq,
        "stage_dx_iv": stage,
        "dmets_stage_i_iii": dmets,
        "dx_to_dmets_days": to_mets,
        "tt_os_dx_days": os_days,
        "os_dx_status": dead,
        "dob_ca_dx_days": dob_dx,
    }


@pytest.fixture(scope="module")
def toy():
    """Six cancers that exercise every rule."""
    regs = [
        # P1: stage IV; FOLFOX, bevacizumab added after 10 days (combination), then 5-FU maintenance,
        # then FOLFOX again after a long holiday (continuation), then FOLFIRI (new line)
        _regimen("P1", 0, 1, FOLFOX, 10, 100),
        _regimen("P1", 0, 2, [BEV], 20, 100),
        _regimen("P1", 0, 3, ["Fluorouracil(5FU,Adrucil)", "Leucovorin Calcium(Wellcovorin)"], 110, 200),
        _regimen("P1", 0, 4, FOLFOX, 400, 500),
        _regimen("P1", 0, 5, [*FOLFIRI, BEV], 520, None, dc="No"),
        # P2: stage I-III, adjuvant FOLFOX, metastasis on day 600, then capecitabine,
        # then two investigational regimens
        _regimen("P2", 0, 1, FOLFOX, 30, 200),
        _regimen("P2", 0, 2, ["Capecitabine(Xeloda)"], 610, 800),
        _regimen("P2", 0, 3, [INV], 850, 850),
        _regimen("P2", 0, 4, [INV, INV], 950, 950),
        # P3: never advanced: adjuvant therapy only
        _regimen("P3", 0, 1, FOLFOX, 40, 200),
        # P4: stage IV; intraperitoneal mitomycin (local route) then FOLFIRI;
        # leuprolide alone later (backbone only)
        _regimen(
            "P4",
            0,
            1,
            ["Mitomycin(Mutamycin)"],
            5,
            5,
            route="Intraperitoneal (direct into the abdominal cavity)",
        ),
        _regimen("P4", 0, 2, FOLFIRI, 30, 300),
        _regimen("P4", 0, 3, ["Leuprolide Acetate(Lupron)"], 320, 500),
        # P5: stage IV, regimen with a label that is not in the vocabulary
        _regimen("P5", 0, 1, ["Mystery Compound(XY-1)"], 15, 60),
        # P6: stage IV, same agents resumed after progression on day 250 (imaging),
        # new agent on day 400 without progression
        _regimen("P6", 0, 1, FOLFOX, 10, 150),
        _regimen("P6", 0, 2, FOLFOX, 300, 380),
        _regimen("P6", 0, 3, ["Regorafenib(Stivarga)"], 400, 450),
        # P6 second cancer (non-index) ignored by default
    ]
    cancers = [
        _cancer("P1", 0),
        _cancer("P2", 0, stage="Stage I-III", dmets=1, to_mets=600),
        _cancer("P3", 0, stage="Stage I-III", dmets=0),
        _cancer("P4", 0),
        _cancer("P5", 0),
        _cancer("P6", 0, dead=1, os_days=470),
    ]
    imaging = [
        {
            "record_id": "P6",
            "image_overall": "Progressing/Worsening/Enlarging",
            "image_scan_int": 20250,
            "dx_scan_days": 250,
        },
        {
            "record_id": "P6",
            "image_overall": "Stable/No change",
            "image_scan_int": 20420,
            "dx_scan_days": 420,
        },
        {
            "record_id": "P1",
            "image_overall": "Progressing/Worsening/Enlarging",
            "image_scan_int": 20510,
            "dx_scan_days": 510,
        },
    ]
    med_onc = [
        {
            "record_id": "P2",
            "md_ca_status": "Progressing/Worsening/Enlarging",
            "md_onc_visit_int": 20840,
            "dx_md_visit_days": 840,
        },
    ]
    return _cohort(regs, cancers, imaging, med_onc)


# ----------------------------------------------------------------------------- vocabulary
def test_vocabulary_is_complete_and_consistent():
    vocab = lot.load_vocabulary()
    assert len(vocab) == 205
    frame = vocab.frame
    assert set(frame["lot_role"]) <= set(vocabulary.ROLES)
    assert (frame["agents"] != "").all() and frame["label"].is_unique
    # both generations' spellings resolve to one agent
    for a, b in [
        ("Gemcitabine Hydrochloride(Gemzar)", "Gemcitabine HCL(FF10832,Gemzar)"),
        ("Leucovorin Calcium(Wellcovorin)", "Leucovorin(Wellcovorin)"),
        ("Tamoxifen Citrate(Nolvadex)", "Tamoxifen(Nolvadex)"),
        ("Leuprolide Acetate(Lupron)", "Leuprolide(Lupron)"),
    ]:
        assert vocab.lookup(a).agents == vocab.lookup(b).agents
    assert vocab.lookup("Goserlin Acetate(Zoladex,ZDX)").agents == ("Goserelin",)
    assert vocab.lookup("Pertuzumab-Trastuzumab-Hyaluronidase-ZZXF(Phesgo)").agents == (
        "Pertuzumab",
        "Trastuzumab",
    )
    assert vocab.lookup("BCG Vaccine(TICE)").lot_role == "local"
    assert vocab.lookup("Leucovorin").lot_role == "modulator"
    assert vocab.lookup("Degarelix(Firmagon)").lot_role == "backbone"
    assert "Leuprolide" in vocab.agents(role="backbone") and "Oxaliplatin" in vocab.agents(
        drug_class="chemotherapy"
    )
    assert "DrugVocabulary(205 labels" in repr(vocab)


def test_label_parsing_and_unknown_labels():
    assert lot.base_label("Docetaxel (Taxotere,RP56976)") == "Docetaxel"
    assert lot.base_label("  Gemcitabine   HCL(x)") == "Gemcitabine HCL"
    assert lot.synonyms("Cisplatin(CDDP, DDP,Platinol)") == ["CDDP", "DDP", "Platinol"]
    assert lot.synonyms("Investigational Drug") == []
    vocab = lot.load_vocabulary()
    assert vocab.lookup("Mystery Compound(XY-1)") is None and "Mystery Compound" not in vocab
    entry = vocab.classify("Mystery Compound(XY-1)")
    assert (
        entry.known is False and entry.drug_class == "unclassified" and entry.agents == ("Mystery Compound",)
    )
    with pytest.raises(ValueError, match="lacks columns"):
        lot.DrugVocabulary(pd.DataFrame({"label": ["a"]}))


# ----------------------------------------------------------------------------- annotation
def test_regimen_drugs_and_annotation(toy):
    drugs = lot.regimen_drugs(toy)
    assert set(drugs.columns) >= {
        "record_id",
        "ca_seq",
        "regimen_number",
        "slot",
        "label",
        "agents",
        "lot_role",
        "known",
    }
    assert len(drugs) == 36  # every filled drug slot: P1 13, P2 7, P3 3, P4 5, P5 1, P6 7
    regs = lot.annotate_regimens(toy)
    p1 = regs[regs.record_id == "P1"].set_index("regimen_number")
    assert p1.loc[1, "agents"] == "Fluorouracil;Leucovorin;Oxaliplatin"
    assert p1.loc[1, "roles"] == "systemic;modulator;systemic"
    # end dates are kept only for discontinued regimens, and masked for investigational ones
    assert np.isnan(p1.loc[5, "end_days"]) and p1.loc[4, "end_days"] == 500
    p2 = regs[regs.record_id == "P2"].set_index("regimen_number")
    assert p2.loc[4, "agents"] == "Investigational Drug [regimen 4]" and p2.loc[4, "n_investigational"] == 2
    assert p2.loc[4, "end_masked"] and np.isnan(p2.loc[4, "end_days"])
    p4 = regs[regs.record_id == "P4"].set_index("regimen_number")
    assert p4.loc[1, "local_route"] and p4.loc[1, "roles"] == "local"
    assert regs[regs.record_id == "P5"]["unknown_labels"].iloc[0] == "Mystery Compound"


def test_progression_events_use_each_cancers_diagnosis(toy):
    ev = lot.progression_events(toy)
    assert set(ev.columns) == {"record_id", "ca_seq", "days", "source", "level"}
    p6 = ev[ev.record_id == "P6"]
    assert p6["days"].tolist() == [250.0] and p6["source"].tolist() == ["imaging"]
    assert ev[ev.record_id == "P2"]["source"].tolist() == ["med_onc"]
    assert lot.progression_events(toy, sources=["imaging"])["source"].eq("imaging").all()
    with pytest.raises(ValueError, match="unknown progression source"):
        lot.progression_events(toy, sources=["pathology"])


# ----------------------------------------------------------------------------- index events
def test_index_events(toy):
    adv = lot.index_events(toy, "advanced")
    adv = adv.set_index("record_id")
    assert adv.loc["P1", "index_days"] == 0 and adv.loc["P1", "index_source"] == "stage IV at diagnosis"
    assert adv.loc["P2", "index_days"] == 600 and adv.loc["P2", "index_source"] == "first distant metastasis"
    assert np.isnan(adv.loc["P3", "index_days"])
    dx = lot.index_events(toy, "diagnosis")
    assert (dx["index_days"] == 0).all() and (dx["index_event"] == "diagnosis").all()
    prog = lot.index_events(toy, "first_progression").set_index("record_id")
    assert (
        prog.loc["P6", "index_days"] == 250
        and prog.loc["P6", "index_source"] == "first progression (imaging)"
    )
    assert prog.loc["P2", "index_days"] == 840 and prog.loc["P1", "index_days"] == 510
    # after_days moves the first eligible progression
    later = lot.index_events(toy, "first_progression", after_days=300).set_index("record_id")
    assert np.isnan(later.loc["P6", "index_days"])
    custom = pd.DataFrame(
        {"record_id": ["P1", "P3"], "index_days": [50, 70], "index_source": ["trial entry", "trial entry"]}
    )
    cu = lot.index_events(toy, custom).set_index("record_id")
    assert cu.loc["P1", "index_days"] == 50 and cu.loc["P3", "index_source"] == "trial entry"
    assert np.isnan(cu.loc["P2", "index_days"]) and (cu["index_event"] == "custom").all()
    with pytest.raises(ValueError, match="unknown index event"):
        lot.index_events(toy, "birthday")
    with pytest.raises(ValueError, match="custom index must be"):
        lot.index_events(toy, pd.DataFrame({"record_id": ["P1"]}))
    with pytest.raises(ValueError, match="more than one row"):
        lot.index_events(
            toy, pd.DataFrame({"record_id": ["P1", "P1"], "ca_seq": [0, 0], "index_days": [1, 2]})
        )
    with pytest.raises(ValueError, match="first_systemic needs"):
        lot.index_events(toy, "first_systemic")


# ----------------------------------------------------------------------------- rules
def test_rules_presets_and_validation():
    assert lot.PRESETS["prostate"].backbone_agents == ("Bicalutamide", "Flutamide", "Nilutamide")
    assert lot.PRESETS["progression"].progression_aware
    rules = lot.LineRules().with_(restart_gap_days=90)
    assert rules.restart_gap_days == 90 and rules.describe()["combination_window_days"] == 28
    for bad in [
        {"pre_index": "drop"},
        {"investigational": "merge"},
        {"combination_window_days": -1},
        {"restart_gap_days": -5},
    ]:
        with pytest.raises(ValueError):
            lot.LineRules(**bad)
    with pytest.raises(ValueError, match="unknown rules preset"):
        lot.derive(synthetic.generate("CRC", n_patients=5, seed=0, genomic=False), rules="nope")


# ----------------------------------------------------------------------------- derivation
def test_derive_default_rules(toy):
    res = lot.derive(toy, index="advanced")
    assert res.index_event == "advanced" and res.rules.name == "default"
    assert res.unknown_labels == ["Mystery Compound"]
    reg = res.regimens.set_index(["record_id", "regimen_number"])
    lines = res.lines.set_index(["record_id", "line_number"])
    # P1: combination within the window, then 5-FU alone and FOLFOX without bevacizumab (both subsets of the
    # line's agents, hence maintenance) and FOLFIRI with bevacizumab as a new line
    assert reg.loc[("P1", 1), "line_role"] == "initial" and reg.loc[("P1", 2), "line_role"] == "combination"
    assert (
        reg.loc[("P1", 3), "line_role"] == "maintenance" and reg.loc[("P1", 4), "line_role"] == "maintenance"
    )
    assert (
        reg.loc[("P1", 5), "line_number"] == 2 and "new agent(s): Irinotecan" in reg.loc[("P1", 5), "reason"]
    )
    assert lines.loc[("P1", 1), "agents"] == "Bevacizumab;Fluorouracil;Oxaliplatin"
    assert lines.loc[("P1", 1), "backbone_agents"] == "Leucovorin"
    assert lines.loc[("P1", 1), "regimen_numbers"] == "1,2,3,4" and lines.loc[("P1", 1), "end_days"] == 500
    assert (
        lines.loc[("P1", 1), "end_reason"] == "next line started"
        and lines.loc[("P1", 1), "next_line_start_days"] == 520
    )
    assert lines.loc[("P1", 2), "end_reason"] == "ongoing or end unknown" and np.isnan(
        lines.loc[("P1", 2), "end_days"]
    )
    # P2: adjuvant therapy before the metastasis is excluded; two investigational regimens are two lines
    assert reg.loc[("P2", 1), "status"] == "pre_index" and pd.isna(reg.loc[("P2", 1), "line_number"])
    assert reg.loc[("P2", 2), "line_number"] == 1 and reg.loc[("P2", 3), "line_number"] == 2
    assert reg.loc[("P2", 4), "line_number"] == 3
    assert (
        lines.loc[("P2", 2), "agents"] == "Investigational Drug"
        and lines.loc[("P2", 3), "n_investigational"] == 2
    )
    assert lines.loc[("P2", 2), "investigational"] and lines.loc[("P2", 1), "days_from_index"] == 10
    # P3 has no index: regimens get no line and the cancer counts zero lines
    assert reg.loc[("P3", 1), "status"] == "no_index"
    # P4: the intraperitoneal regimen is excluded, leuprolide alone is backbone kept with line 1
    assert reg.loc[("P4", 1), "status"] == "excluded" and reg.loc[("P4", 3), "status"] == "backbone_only"
    assert (
        reg.loc[("P4", 3), "line_number"] == 1
        and lines.loc[("P4", 1), "backbone_agents"] == "Leucovorin;Leuprolide"
    )
    # P5: unknown label still defines a line
    assert lines.loc[("P5", 1), "agents"] == "Mystery Compound"
    # P6: without progression awareness the FOLFOX resumption continues line 1
    assert reg.loc[("P6", 2), "line_role"] == "continuation" and reg.loc[("P6", 3), "line_number"] == 2
    cancers = res.cancers.set_index("record_id")
    assert (
        cancers.loc["P1", "n_lines"] == 2
        and cancers.loc["P2", "n_pre_index"] == 1
        and cancers.loc["P3", "n_lines"] == 0
    )
    assert (
        cancers.loc["P2", "first_line_agents"] == "Capecitabine"
        and cancers.loc["P2", "first_line_days_from_index"] == 10
    )
    assert cancers.loc["P6", "died"] and cancers.loc["P6", "follow_up_days"] == 470
    summary = res.summary()
    assert (
        summary.loc[0, "cancers"] == 6
        and summary.loc[0, "with_index"] == 5
        and summary.loc[0, "with_lines"] == 5
    )
    assert "LinesOfTherapy(CRC, index='advanced'" in repr(res)


def test_derive_rule_variants(toy):
    # progression-aware: P6 resumption after progression is a new line;
    # regorafenib without progression continues it
    res = lot.derive(toy, index="advanced", rules="progression")
    reg = res.regimens.set_index(["record_id", "regimen_number"])
    assert (
        reg.loc[("P6", 2), "line_number"] == 2
        and "after progression on day 250" in reg.loc[("P6", 2), "reason"]
    )
    assert (
        reg.loc[("P6", 3), "line_role"] == "switch_without_progression"
        and reg.loc[("P6", 3), "line_number"] == 2
    )
    lines = res.lines.set_index(["record_id", "line_number"])
    assert lines.loc[("P6", 2), "agents"] == "Fluorouracil;Oxaliplatin;Regorafenib"
    assert lines.loc[("P1", 1), "progression_before_next_days"] == 510
    # restart gap: P1's FOLFOX resumption after a 200-day holiday starts a new line
    res = lot.derive(toy, index="advanced", rules=lot.LineRules(restart_gap_days=90))
    reg = res.regimens.set_index(["record_id", "regimen_number"])
    assert reg.loc[("P1", 4), "line_number"] == 2 and "gap of 200 days" in reg.loc[("P1", 4), "reason"]
    # maintenance off: every later regimen with agents is a new line
    res = lot.derive(toy, index="advanced", rules=lot.LineRules(maintenance_same_line=False))
    assert res.regimens.set_index(["record_id", "regimen_number"]).loc[("P1", 3), "line_number"] == 2
    # pre-index therapy as line 0, with a tolerance
    res = lot.derive(toy, index="advanced", rules=lot.LineRules(pre_index="line_zero"))
    reg = res.regimens.set_index(["record_id", "regimen_number"])
    assert reg.loc[("P2", 1), "line_number"] == 0 and reg.loc[("P2", 1), "line_role"] == "pre_index"
    assert (res.lines.query("record_id == 'P2'")["line_number"].tolist()) == [0, 1, 2, 3]
    assert res.cancers.set_index("record_id").loc["P2", "n_pre_index"] == 1
    # investigational excluded: P2's investigational regimens are not lines
    res = lot.derive(toy, index="advanced", rules=lot.LineRules(investigational="exclude"))
    assert res.cancers.set_index("record_id").loc["P2", "n_lines"] == 1
    # extra excluded and backbone agents by name
    res = lot.derive(
        toy,
        index="advanced",
        rules=lot.LineRules(backbone_agents=("Bevacizumab",), excluded_agents=("Capecitabine",)),
    )
    lines = res.lines.set_index(["record_id", "line_number"])
    assert (
        "Bevacizumab" in lines.loc[("P1", 1), "backbone_agents"]
        and "Bevacizumab" not in lines.loc[("P1", 1), "agents"]
    )
    assert res.cancers.set_index("record_id").loc["P2", "first_line_agents"] == "Investigational Drug"
    # tolerance: a regimen 5 days before the index is treated as starting at the index
    res = lot.derive(
        toy,
        index=pd.DataFrame({"record_id": ["P1"], "index_days": [15]}),
        rules=lot.LineRules(pre_index_tolerance_days=7),
    )
    assert res.regimens.set_index(["record_id", "regimen_number"]).loc[("P1", 1), "line_number"] == 1


def test_derive_other_index_events(toy):
    res = lot.derive(toy, index="first_systemic")
    c = res.cancers.set_index("record_id")
    assert (
        c.loc["P2", "index_days"] == 30 and c.loc["P4", "index_days"] == 30
    )  # P4's local regimen does not count
    assert c.loc["P3", "n_lines"] == 1
    res = lot.derive(toy, index="first_progression", progression_sources=("imaging",))
    c = res.cancers.set_index("record_id")
    assert c.loc["P6", "index_days"] == 250 and np.isnan(c.loc["P2", "index_days"])
    assert res.cancers.set_index("record_id").loc["P6", "n_lines"] == 2  # FOLFOX resumption, then regorafenib
    res = lot.derive(
        toy, index=lambda data: pd.DataFrame({"record_id": ["P1"], "ca_seq": [0], "index_days": [450]})
    )
    assert res.index_event == "custom" and res.cancers.set_index("record_id").loc["P1", "n_lines"] == 1
    with pytest.raises(ValueError, match="cancers must be"):
        lot.derive(toy, cancers="some")


def test_derive_all_and_to_dir(toy, tmp_path):
    crc = synthetic.generate("CRC", n_patients=40, seed=3, genomic=False)
    res = lot.derive_all([toy, crc], index="advanced")
    assert res.cohort == "CRC,CRC" and set(res.cancers["cohort"]) == {"CRC"}
    assert len(res.lines) == len(lot.derive(toy).lines) + len(lot.derive(crc).lines)
    folder = res.to_dir(tmp_path / "lot", compress=True)
    assert {p.name for p in folder.iterdir()} == {
        "lot_regimens.csv.gz",
        "lot_lines.csv.gz",
        "lot_cancers.csv.gz",
        "lot_events.csv.gz",
        "lot_settings.json",
    }
    settings = json.loads((folder / "lot_settings.json").read_text(encoding="utf-8"))
    assert settings["index_event"] == "advanced" and settings["rules"]["combination_window_days"] == 28
    with pytest.raises(ValueError, match="no cohorts"):
        lot.derive_all([])


def test_synthetic_and_fixture_cohorts_derive():
    for cohort in ("NSCLC", "Prostate", "BLADDER"):
        data = synthetic.generate(cohort, n_patients=60, seed=1, genomic=False)
        res = lot.derive(data, index="advanced", rules="prostate" if cohort == "Prostate" else None)
        assert (res.regimens["status"] != "").all()
        assert (
            res.lines[res.lines.line_number > 0].groupby(["record_id", "ca_seq"]).size()
            == res.cancers.set_index(["record_id", "ca_seq"]).n_lines.reindex(
                res.lines[res.lines.line_number > 0].groupby(["record_id", "ca_seq"]).size().index
            )
        ).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx = io.load_release(FIXTURE, genomic=False)
    res = lot.derive(fx, index="advanced")
    assert res.unknown_labels == [] and res.summary().loc[0, "with_lines"] > 50
    assert (
        res.lines["start_days"]
        >= res.lines.merge(res.cancers[["record_id", "ca_seq", "index_days"]], on=["record_id", "ca_seq"])[
            "index_days"
        ]
    ).all()


def test_engine_helpers():
    assert (
        engine._display({"Investigational Drug [regimen 3]", "Cisplatin"}) == "Cisplatin;Investigational Drug"
    )
    assert (
        np.isnan(engine._progressed(np.array([]), 0, 10))
        and engine._progressed(np.array([5.0, 8.0]), 4, 10) == 5.0
    )
    assert np.isnan(engine._nan_max([np.nan])) and engine._nan_max([1.0, np.nan, 3.0]) == 3.0


# ----------------------------------------------------------------------------- real data (optional)
@pytest.mark.skipif(not os.environ.get("GENIEKIT_DATA_DIR"), reason="GENIEKIT_DATA_DIR not set")
def test_real_releases_satisfy_invariants():
    root = Path(os.environ["GENIEKIT_DATA_DIR"])
    folders = sorted(p for p in root.iterdir() if p.is_dir() and (p / "clinical").exists())
    assert folders, f"no release folders under {root}"
    for folder in folders:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = io.load_release(
                folder,
                tables=["patient", "cancer_index", "cancer_non_index", "regimen", "imaging", "med_onc"],
                genomic=False,
            )
        res = lot.derive(data, index="advanced", rules="prostate" if data.cohort == "Prostate" else None)
        assert res.unknown_labels == [], (data.cohort, res.unknown_labels)
        assert (res.regimens["status"] != "").all()
        lines = res.lines[res.lines.line_number > 0]
        for _, g in lines.groupby(["record_id", "ca_seq"]):
            g = g.sort_values("line_number")
            assert g["line_number"].tolist() == list(range(1, len(g) + 1))
            assert g["start_days"].is_monotonic_increasing
