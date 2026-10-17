from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from geniekit import dictionary, harmonize, io, releases, synthetic
from geniekit.harmonize import derive, genomic
from geniekit.harmonize.derive import AAAPI

FIXTURE = Path(__file__).parent / "fixtures" / "nsclc_sim"


@pytest.fixture(scope="module")
def cw() -> harmonize.Crosswalk:
    return harmonize.load_crosswalk()


@pytest.fixture(scope="module")
def raw() -> dict[str, io.CohortData]:
    return synthetic.generate_all(n_patients=40, seed=11)


@pytest.fixture(scope="module")
def harmonised(raw) -> dict[str, harmonize.HarmonizedCohort]:
    return {c: harmonize.harmonize(d) for c, d in raw.items()}


# --------------------------------------------------------------------- crosswalk
def test_crosswalk_loads_and_validates_against_dictionary(cw):
    assert cw.version == 1
    assert cw.generations == {1: ("NSCLC", "CRC"), 2: ("BrCa", "PANC", "Prostate", "BLADDER")}
    assert cw.validate() == []
    assert len(cw.rules) >= 20
    assert {r.kind for r in cw.rules} == set(harmonize.crosswalk.RULE_KINDS)
    assert cw.generation("nsclc") == 1 and cw.generation("BLADDER") == 2
    assert "rules=" in repr(cw)
    frame = cw.to_frame()
    assert set(frame.columns) >= {"id", "kind", "table", "cohorts", "variables", "basis"}
    assert frame["basis"].str.len().gt(20).all()


def test_rules_for_filters(cw):
    nsclc_cancer = cw.rules_for("NSCLC", "cancer_index")
    assert any(r.id == "cigarette-rename" for r in nsclc_cancer)
    assert not any(r.id == "cigarette-rename" for r in cw.rules_for("BLADDER", "cancer_index"))
    assert all(r.kind == "rename" for r in cw.rules_for("CRC", kinds=("rename",)))
    partial = harmonize.Crosswalk(version=1, generated="", generations={1: ("NSCLC",)}, rules=())
    with pytest.raises(KeyError):
        partial.generation("CRC")


def test_validate_detects_bad_rules(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
version: 9
generations: {1: [NSCLC, CRC], 2: [BrCa, PANC, Prostate, BLADDER]}
rules:
  - id: x
    kind: rename
    table: cancer
    cohorts: [NSCLC]
    from: nope
    to: ca_cigarette
    basis: t
  - id: x
    kind: recode
    table: patient
    cohorts: [CRC]
    variable: naaccr_sex_code
    map: {Alien: Other}
    basis: t
  - id: y
    kind: rename
    table: cancer
    cohorts: [NSCLC]
    from: ca_lung_cigarette
    to: not_declared_anywhere
    basis: t
""",
        encoding="utf-8",
    )
    custom = harmonize.load_crosswalk(bad)
    problems = custom.validate()
    assert any("duplicate rule id" in p for p in problems)
    assert any("does not declare cancer.nope" in p for p in problems)
    assert any("has no level(s) ['Alien']" in p for p in problems)
    assert any("not declared by any other cohort" in p for p in problems)


# --------------------------------------------------------------------- harmonize
def test_generation1_columns_are_renamed(harmonised):
    for c in ("NSCLC", "CRC"):
        ci = harmonised[c]["cancer_index"]
        assert "ca_cigarette" in ci.columns and "ca_lung_cigarette" not in ci.columns
        assert "dmets_post_dx" in ci.columns and "dmets_stage_i_iii" not in ci.columns
        for site in ("liver", "breast", "brain_cns"):
            assert f"dist_mets_{site}" in ci.columns
            assert f"dx_to_dist_mets_{site}_days" in ci.columns
        # regions without a generation-2 equivalent keep their names
        assert {"dmets_abdomen", "dmets_thorax", "dmets_extremity"} <= set(ci.columns)
        assert harmonised[c].report.renames()["dmets_brain"] == "dist_mets_brain_cns"


def test_values_match_generation2_vocabulary(harmonised):
    dd = dictionary.load()
    g2_eth = set(dd.values("naaccr_ethnicity_code", "patient")["BLADDER"])
    g2_race = set(dd.values("naaccr_race_code_primary", "patient")["BLADDER"])
    g2_sex = set(dd.values("naaccr_sex_code", "patient")["BLADDER"])
    for c in ("NSCLC", "CRC"):
        pt = harmonised[c]["patient"]
        assert set(pt["naaccr_ethnicity_code"].dropna()) <= g2_eth
        assert set(pt["naaccr_race_code_primary"].dropna()) <= g2_race
        assert set(pt["naaccr_sex_code"].dropna()) <= g2_sex
        # the detailed generation-1 values are preserved
        assert "naaccr_ethnicity_code_detail" in pt.columns
        assert set(pt["naaccr_ethnicity_code_detail"].dropna()) <= set(
            dd.values("naaccr_ethnicity_code", "patient")[c]
        )
    ci = harmonised["NSCLC"]["cancer_index"]
    assert "Former user (quit <1 year)" not in set(ci["ca_cigarette"].dropna())


def test_heme_type_and_pfs_cohort_collapse(harmonised):
    for c in ("BrCa", "PANC", "Prostate"):
        ci = harmonised[c]["cancer_index"]
        vals = set(ci["ca_heme_type"].dropna())
        assert not vals & {
            "Leukemia NOS",
            "Non-Hodgkin's Lymphoma",
            "Other Hematopoietic or lymphoid neoplasm",
        }
        assert "ca_heme_type_detail" in ci.columns
    for c in ("Prostate", "BLADDER"):
        ci = harmonised[c]["cancer_index"]
        assert set(ci["pfs_cohort"].dropna()) <= {"Stage I-III with Distant Mets", "Stage IV"}
        detail = set(ci["pfs_cohort_detail"].dropna())
        assert detail <= {
            "Stage I-III with Distant Mets",
            "Stage IV with Distant Mets at Diagnosis",
            "Stage IV with Distant Mets Post-Diagnosis",
        }
    for c, h in harmonised.items():
        ci = h["cancer_index"]
        origins = set(ci["pfs_time_origin"].dropna())
        assert origins <= {"diagnosis", "first distant metastasis"}, c
        stage_iv = ci["pfs_cohort"] == "Stage IV"
        if c not in ("Prostate", "BLADDER"):
            assert (ci.loc[stage_iv, "pfs_time_origin"] == "diagnosis").all()
            assert ci.loc[stage_iv, "pfs_cohort_detail"].isna().all()
        assert (
            ci.loc[ci["pfs_cohort"] == "Stage I-III with Distant Mets", "pfs_time_origin"]
            == "first distant metastasis"
        ).all()


def test_seer_summary_stage_and_cpt_year(harmonised):
    for c, h in harmonised.items():
        ci = h["cancer_index"]
        assert str(ci["naaccr_seer_sum_stage"].dtype) == "Int64", c
        assert "naaccr_seer_sum_stage_label" in ci.columns
        coded = ci["naaccr_seer_sum_stage"].dropna()
        assert set(coded.astype(int)) <= set(derive.SEER_SUMMARY_STAGE_LABELS), c
        labels = ci.loc[ci["naaccr_seer_sum_stage"].notna(), "naaccr_seer_sum_stage_label"]
        assert labels.notna().all()
        assert str(h["cpt"]["cpt_seq_date"].dtype) == "Int64", c
        assert h["cpt"]["cpt_seq_date"].dropna().between(2000, 2030).all()


def test_seer_summary_stage_parses_text_forms():
    frame = pd.DataFrame(
        {"naaccr_seer_sum_stage": ["1", "Distant", "Regional, NOS", "7 = Distant", 9.0, None, "mystery"]}
    )
    out, detail = derive.seer_summary_stage(frame, "NSCLC")
    assert out["naaccr_seer_sum_stage"].tolist()[:5] == [1, 7, 5, 7, 9]
    assert pd.isna(out["naaccr_seer_sum_stage"].iloc[5]) and pd.isna(out["naaccr_seer_sum_stage"].iloc[6])
    assert out["naaccr_seer_sum_stage_label"].iloc[1] == "Distant"
    assert out["naaccr_seer_sum_stage_detail"].iloc[6] == "mystery"
    assert "unmapped 1 value(s): mystery" in detail


def test_race_ethnicity_rules():
    # the seven cases that separate the released GENIE rule from the conservative one
    frame = pd.DataFrame(
        {
            "naaccr_ethnicity_code": [
                "Spanish/Hispanic",
                "Non-Spanish/non-Hispanic",
                "Unknown",
                "Non-Spanish/non-Hispanic",
                None,
                "Spanish/Hispanic",
                "Unknown",
                "Spanish/Hispanic",
                None,
            ],
            "naaccr_race_code_primary": [
                "White",
                "Asian",
                "Black",
                "Native American",
                None,
                "Other",
                "Asian",
                "Unknown",
                "White",
            ],
            "race_ethnicity": [
                "Hispanic/Latinx",
                "Asian Indian or Pakistani NOS",
                "Hispanic/Latinx",
                "Other",
                None,
                "Other",
                AAAPI,
                "Unknown",
                None,
            ],
        }
    )
    out, detail = derive.race_ethnicity(frame, "NSCLC")
    assert out["race_ethnicity"].tolist() == [
        "Hispanic/Latinx",
        AAAPI,
        "Hispanic/Latinx",  # unknown origin with a Black race code counts as Hispanic in the released rule
        "Other",
        "Unknown",
        "Other",  # race Other takes precedence over a Hispanic origin
        AAAPI,
        "Unknown",
        "Unknown",  # a missing origin code is not an origin code
    ]
    assert "race_ethnicity_released" in out.columns
    assert "(rule: genie); agreement with released value 7/7" in detail
    out2, detail2 = derive.race_ethnicity(frame, "NSCLC", rule="conservative")
    assert out2["race_ethnicity"].tolist() == [
        "Hispanic/Latinx",
        AAAPI,
        "Unknown",
        "Other",
        "Unknown",
        "Hispanic/Latinx",
        AAAPI,
        "Hispanic/Latinx",
        "Unknown",
    ]
    assert "agreement with released value 4/7" in detail2
    with pytest.raises(ValueError, match="rule must be one of"):
        derive.race_ethnicity(frame, "NSCLC", rule="other")
    # idempotence: a generation-1 frame is recognised by race_ethnicity_released, a generation-2 frame
    # (where the dictionary does not declare race_ethnicity) by the presence of the derived column
    assert derive.race_ethnicity(out, "NSCLC")[1] == "skipped: already derived"
    gen2 = frame.drop(columns="race_ethnicity")
    derived, _ = derive.race_ethnicity(gen2, "BrCa")
    assert "race_ethnicity_released" not in derived.columns
    assert derive.race_ethnicity(derived, "BrCa")[1] == "skipped: already derived"
    # the released generation-1 column is not mistaken for a derivation
    assert derive.race_ethnicity(frame, "NSCLC")[1].startswith("derived")


def test_race_ethnicity_rule_option_flows_from_the_crosswalk(raw):
    cw = harmonize.load_crosswalk()
    rule = next(r for r in cw.rules if r.id == "race-ethnicity-derived")
    assert rule.params["options"] == {"rule": "genie"}
    rule.params["options"] = {"rule": "conservative"}
    h = harmonize.harmonize(raw["BrCa"], crosswalk=cw)
    rec = h.report.to_frame().query("rule_id == 'race-ethnicity-derived'").iloc[0]
    assert rec["status"] == "applied" and "(rule: conservative)" in rec["detail"]


def test_report_contents(harmonised):
    rep = harmonised["NSCLC"].report
    frame = rep.to_frame()
    assert set(frame["status"]) == {"applied", "skipped", "note"}
    assert rep.unmapped().empty
    assert (frame[frame["rule_id"] == "dmets-post-dx"]["caveat"] == "population").all()
    assert len(rep.applied()) > 30
    summary = rep.summary()
    assert {"cohort", "kind", "status", "rules", "rows_changed"} <= set(summary.columns)
    assert "HarmonizationReport(" in repr(rep)
    notes = frame[frame["kind"] == "note"]
    assert "tables-by-cohort" in set(notes["rule_id"]) and (notes["table"] == "*").sum() == 1


def test_unknown_values_are_reported(raw):
    data = synthetic.generate("CRC", n_patients=10, seed=3)
    data["patient"].loc[0, "naaccr_sex_code"] = "Martian"
    h = harmonize.harmonize(data)
    um = h.report.unmapped()
    assert len(um) == 1 and um.iloc[0]["source"] == "naaccr_sex_code" and "Martian" in um.iloc[0]["detail"]


def test_keep_original_and_idempotence(raw):
    h = harmonize.harmonize(raw["NSCLC"], keep_original=True)
    ci = h["cancer_index"]
    assert "ca_lung_cigarette_orig" in ci.columns and "ca_cigarette" in ci.columns
    again = harmonize.harmonize(h)
    skipped = again.report.skipped()
    assert (skipped["detail"].str.contains("absent|already present|already derived")).all()
    assert again["cancer_index"].shape[1] >= ci.shape[1]


def test_genomic_unification(harmonised):
    for h in harmonised.values():
        assert "sv" in h and "fusions" not in h
        assert list(h["sv"].columns) == genomic.SV_COLUMNS
        assert h["mutations"].columns[0] == "sample_id"
        assert list(h["cna_long"].columns) == ["sample_id", "hugo_symbol", "cna"]
        assert (h["cna_long"]["cna"] != 0).all()
    nsclc_sv = harmonised["NSCLC"]["sv"]
    if len(nsclc_sv):
        assert set(nsclc_sv["source_format"]) == {"fusions"}
        assert nsclc_sv["gene_1"].notna().all() and nsclc_sv["gene_2"].notna().all()
    bladder_sv = harmonised["BLADDER"]["sv"]
    if len(bladder_sv):
        assert set(bladder_sv["source_format"]) == {"sv"} and set(bladder_sv["sv_class"]) == {"FUSION"}


def test_structural_variants_helpers():
    fus = pd.DataFrame(
        {
            "Hugo_Symbol": ["ALK", "EML4", "ROS1"],
            "Tumor_Sample_Barcode": ["S1", "S1", "S2"],
            "Fusion": ["EML4-ALK fusion", "EML4-ALK fusion", "ROS1 rearrangement"],
            "Frame": ["in-frame", "in-frame", None],
        }
    )
    out = genomic.structural_variants(fus)
    assert len(out) == 2
    assert out.iloc[0][["gene_1", "gene_2"]].tolist() == ["EML4", "ALK"]
    assert out.iloc[1]["gene_1"] == "ROS1" and pd.isna(out.iloc[1]["gene_2"])
    sv = pd.DataFrame(
        {
            "Sample_Id": ["S3"],
            "Site1_Hugo_Symbol": ["TMPRSS2"],
            "Site2_Hugo_Symbol": ["ERG"],
            "Event_Info": ["TMPRSS2-ERG"],
            "Class": ["FUSION"],
        }
    )
    out2 = genomic.structural_variants(sv)
    assert (
        out2.iloc[0]["gene_2"] == "ERG"
        and pd.isna(out2.iloc[0]["frame"])
        and out2.iloc[0]["source_format"] == "sv"
    )
    assert genomic.structural_variants(pd.DataFrame(columns=["Sample_Id"])).empty
    with pytest.raises(ValueError):
        genomic.structural_variants(pd.DataFrame({"x": [1]}))
    cna = pd.DataFrame({"Hugo_Symbol": ["EGFR", "PTEN"], "S1": [2, 0], "S2": [0, -2]})
    long = genomic.cna_to_long(cna)
    assert long.to_dict("records") == [
        {"sample_id": "S1", "hugo_symbol": "EGFR", "cna": 2.0},
        {"sample_id": "S2", "hugo_symbol": "PTEN", "cna": -2.0},
    ]


# ------------------------------------------------------------------------- stack
def test_stack_six_cohorts(harmonised):
    stacked = harmonize.stack(harmonised.values())
    assert stacked.cohorts == list(releases.COHORTS)
    ci = stacked["cancer_index"]
    assert sorted(ci["cohort"].unique()) == sorted(releases.COHORTS)
    assert set(ci["schema_generation"]) == {1, 2}
    assert ci["ca_cigarette"].notna().any()
    av = stacked.availability("cancer_index")
    assert av.loc["ca_cigarette"].all() and av.loc["dist_mets_liver"].all()
    assert av.loc["dmets_abdomen"].tolist() == [True, True, False, False, False, False]
    assert av.loc["dist_mets_adrenal"].tolist() == [False, False, True, True, True, True]
    assert "ca_cigarette" in stacked.common_columns("cancer_index")
    assert "dmets_abdomen" not in stacked.common_columns("cancer_index")
    assert "cna" not in stacked and "cna_long" in stacked
    assert sorted(stacked["tumor_marker"]["cohort"].unique()) == ["BrCa", "CRC", "PANC", "Prostate"]
    assert sorted(stacked["radiation"]["cohort"].unique()) == ["BLADDER", "PANC", "Prostate"]
    summary = stacked.summary()
    assert set(summary["table"]) == set(stacked.names)
    assert "StackedData(" in repr(stacked)
    assert len(stacked.report) == sum(len(h.report) for h in harmonised.values())


def test_stack_auto_harmonizes_and_validates(raw):
    stacked = harmonize.stack([raw["NSCLC"], raw["BLADDER"]], tables=["patient", "regimen"])
    assert set(stacked.names) == {"patient", "regimen"}
    assert "race_ethnicity" in stacked["patient"].columns
    with pytest.raises(ValueError, match="duplicate cohorts"):
        harmonize.stack([raw["NSCLC"], raw["NSCLC"]])
    with pytest.raises(TypeError):
        harmonize.stack([raw["NSCLC"]], auto_harmonize=False)


def test_fixture_nsclc_harmonizes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = io.load_release(FIXTURE)
    h = harmonize.harmonize(data)
    ci = h["cancer_index"]
    assert "ca_cigarette" in ci.columns and "dist_mets_liver" in ci.columns
    assert str(ci["naaccr_seer_sum_stage"].dtype) == "Int64"
    assert ci["naaccr_seer_sum_stage"].notna().sum() > 0
    assert str(h["cpt"]["cpt_seq_date"].dtype) == "Int64"
    assert "sv" in h and set(h["sv"]["source_format"]) == {"fusions"}
    assert h.report.unmapped().empty
    stacked = harmonize.stack([h, synthetic.generate("CRC", 10, seed=1)])
    assert sorted(stacked["patient"]["cohort"].unique()) == ["CRC", "NSCLC"]
