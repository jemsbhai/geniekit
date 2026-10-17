from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from geniekit import dictionary, io, releases, synthetic


@pytest.fixture(scope="module")
def cohorts() -> dict[str, io.CohortData]:
    return synthetic.generate_all(n_patients=40, seed=7)


def test_generate_all_covers_six_cohorts(cohorts):
    assert list(cohorts) == list(releases.COHORTS)
    for c, data in cohorts.items():
        assert data.cohort == c and data.synthetic and data.version == releases.get_release(c).version
        assert len(data["patient"]) == 40


def test_clinical_columns_match_dictionary_exactly(cohorts):
    dd = dictionary.load()
    for c, data in cohorts.items():
        for name, frame in data.clinical.items():
            dict_table = "cancer" if name.startswith("cancer") else name
            declared = dd.variables(c, dict_table)["variable"].tolist()
            if name == "cancer_non_index":
                assert set(frame.columns) <= set(declared), (c, name)
            else:
                assert list(frame.columns) == declared, (c, name)


def test_optional_tables_follow_release_structure(cohorts):
    for c, data in cohorts.items():
        rel = releases.get_release(c)
        assert ("tumor_marker" in data) == rel.has("tumor_marker"), c
        assert ("radiation" in data) == rel.has("radiation"), c
        assert ("fusions" in data) == rel.has("fusions"), c
        assert ("sv" in data) == rel.has("sv"), c


def test_generation_specific_variable_names(cohorts):
    g1 = {"ca_lung_cigarette", "dmets_stage_i_iii", "dmets_liver", "dx_to_dmets_liver_days"}
    g2 = {"ca_cigarette", "dmets_post_dx", "dist_mets_liver", "dx_to_dist_mets_liver_days"}
    for c, data in cohorts.items():
        cols = set(data["cancer_index"].columns)
        if releases.get_release(c).schema_generation == 1:
            assert g1 <= cols and not (g2 & cols), c
        else:
            assert g2 <= cols and not (g1 & cols), c


def test_patient_level_coherence(cohorts):
    for c, data in cohorts.items():
        pt = data["patient"]
        assert pt["record_id"].is_unique
        assert pt["record_id"].str.match(r"^GENIE-(DFCI|MSK|UHN|VICC)-\d{6}$").all()
        assert set(pt["institution"]) <= set(releases.get_release(c).institutions)
        dead = pt["hybrid_death_int"].notna()
        assert (pt.loc[dead, "hybrid_death_int"] == pt.loc[dead, "dob_lastalive_int"]).all()
        assert pt.loc[dead, "hybrid_death_source"].notna().all()
        assert pt.loc[~dead, "hybrid_death_source"].isna().all()
        if c == "Prostate":
            assert (pt["naaccr_sex_code"] == "Male").all()
        if c == "BrCa":
            assert (data["cancer_index"]["age_dx"] <= 56).all()
        assert (
            pt["n_regimens_pt"]
            == data["regimen"].groupby("record_id").size().reindex(pt["record_id"]).fillna(0).to_numpy()
        ).all()


def test_cancer_and_metastasis_coherence(cohorts):
    for data in cohorts.values():
        ci = data["cancer_index"]
        assert ci["record_id"].is_unique and (ci["redcap_ca_index"] == "Yes").all()
        assert set(ci["stage_dx"]) <= {"Stage I", "Stage II", "Stage III", "Stage IV"}
        iv = ci["stage_dx"] == "Stage IV"
        assert (ci.loc[iv, "stage_dx_iv"] == "Stage IV").all() and (
            ci.loc[~iv, "stage_dx_iv"] == "Stage I-III"
        ).all()
        flag = "dmets_stage_i_iii" if "dmets_stage_i_iii" in ci.columns else "dmets_post_dx"
        assert ci.loc[iv, flag].isna().all()
        recur = ci[flag] == 1
        assert ci.loc[recur, "dx_to_dmets_days"].gt(0).all()
        assert ci.loc[~recur, "dx_to_dmets_days"].isna().all()
        liver = "dmets_liver" if "dmets_liver" in ci.columns else "dist_mets_liver"
        liver_t = "dx_to_dmets_liver_days" if liver == "dmets_liver" else "dx_to_dist_mets_liver_days"
        assert ci.loc[ci[liver] == 1, liver_t].notna().all()
        assert ci.loc[ci[liver] == 0, liver_t].isna().all()
        # survival: event indicator agrees with the patient table, times are integer days
        pt = data["patient"].set_index("record_id")
        dead = pt.loc[ci["record_id"], "hybrid_death_int"].notna().to_numpy()
        assert (ci["os_dx_status"].to_numpy() == dead.astype(float)).all()
        assert (ci["tt_os_dx_days"] == np.round(ci["tt_os_dx_days"])).all()
        adv = ci["pfs_cohort"].notna()
        assert adv.equals(ci["tt_pfs_i_or_m_adv_days"].notna())
        assert (ci.loc[iv, "pfs_cohort"].str.contains("Stage IV")).all()
        assert (ci.loc[adv, "tt_pfs_i_or_m_adv_days"] <= ci.loc[adv, "tt_os_adv_days"]).all()


def test_regimen_coherence(cohorts):
    for data in cohorts.values():
        rg = data["regimen"]
        ci = data["cancer_index"].set_index("record_id")
        assert rg["record_id"].isin(ci.index).all()
        assert (rg["dx_reg_start_int"] <= rg["dx_reg_end_all_int"]).all()
        assert (rg["dx_reg_end_any_int"] <= rg["dx_reg_end_all_int"]).all()
        assert (rg["dx_reg_start_int"] >= 0).all()
        assert (rg["drugs_num"] == rg["regimen_drugs"].str.count(",") + 1).all()
        assert (rg["drugs_num"] == rg[[f"drugs_drug_{k}" for k in range(1, 6)]].notna().sum(axis=1)).all()
        # regimens are numbered consecutively per patient and never overlap in time
        for _, g in rg.sort_values("regimen_number").groupby("record_id"):
            assert g["regimen_number"].tolist() == list(range(1, len(g) + 1))
            assert (g["dx_reg_start_int"].to_numpy()[1:] >= g["dx_reg_end_all_int"].to_numpy()[:-1]).all()
        # regimen end never exceeds the patient's last follow-up
        end = ci.loc[rg["record_id"], "tt_os_dx_days"].to_numpy()
        assert (rg["dx_reg_end_all_int"].to_numpy() <= end + 0.5).all()
        assert (rg["tt_os_g_days"] + rg["dx_reg_start_int"] - end).abs().max() < 1.0
        assert (rg["ttnt_any_ca_days"] <= rg["tt_os_g_days"] + 0.5).all()
        trial = rg["regimen_drugs"].str.contains("Investigational Drug")
        assert (rg.loc[trial, "drugs_ct_yn"] == "Yes").all()


def test_longitudinal_tables_and_samples(cohorts):
    for c, data in cohorts.items():
        pt = data["patient"].set_index("record_id")
        for name, col in (
            ("imaging", "dx_scan_days"),
            ("med_onc", "dx_md_visit_days"),
            ("pathology", "dx_path_proc_days"),
        ):
            t = data[name]
            assert t["record_id"].isin(pt.index).all()
            end = (
                pt.loc[t["record_id"], "dob_lastalive_int"].to_numpy()
                - pt.loc[t["record_id"], "dob_lastalive_int"].to_numpy()
            )
            _ = end
            assert (
                t[col]
                <= data["cancer_index"].set_index("record_id").loc[t["record_id"], "tt_os_dx_days"].to_numpy()
                + 5
            ).all()
        img = data["imaging"]
        first = img[img["scan_number"] == 1]
        assert first["image_overall"].isna().all()
        progressing = img["image_overall"].astype(object).fillna("").str.contains("Progressing")
        assert progressing.any()
        cpt = data["cpt"]
        assert cpt["cpt_genie_sample_id"].is_unique
        mut = data["mutations"]
        assert set(mut["Tumor_Sample_Barcode"]) <= set(cpt["cpt_genie_sample_id"])
        assert list(data["cna"].columns[1:]) == cpt["cpt_genie_sample_id"].tolist()
        sv = data["fusions"] if "fusions" in data else data["sv"]
        key = "Tumor_Sample_Barcode" if "Tumor_Sample_Barcode" in sv.columns else "Sample_Id"
        assert set(sv[key]) <= set(cpt["cpt_genie_sample_id"])
        if "tumor_marker" in data:
            tm = data["tumor_marker"]
            assert set(tm["tm_type"]) <= {t for t, _, _ in synthetic.COHORT_SPECS[c].tumor_markers}
            assert (tm["tm_num_result"] > 0).all()
        if "radiation" in data:
            rt = data["radiation"]
            assert (rt["rt_total_dose"] == rt["rt_fractions"] * 200).all()


def test_cohort_specific_biomarkers(cohorts):
    crc = cohorts["CRC"]["pathology"]
    assert set(crc.loc[crc["msi_testing"] == "Yes", "msi_result"].dropna().str.contains("MS")) == {True}
    assert crc.loc[crc["msi_testing"] == "No", "msi_result"].isna().all()
    assert "msi_testing" not in cohorts["NSCLC"]["pathology"].columns
    assert "msi_testing" not in cohorts["BLADDER"]["pathology"].columns
    assert "mmr_result" in cohorts["BLADDER"]["pathology"].columns
    nsclc = cohorts["NSCLC"]["pathology"]
    assert nsclc.loc[nsclc["pdl1_testing"] == "Yes", "pdl1_perc"].notna().all()
    brca = cohorts["BrCa"]["cancer_index"]
    assert brca["ca_bca_er"].notna().all() and brca["ca_bca_her_summ"].notna().all()
    crc_mut = cohorts["CRC"]["mutations"]
    assert {"APC", "TP53", "KRAS"} <= set(crc_mut["Hugo_Symbol"])
    assert crc_mut.loc[crc_mut["Hugo_Symbol"] == "BRAF", "HGVSp_Short"].str.startswith("p.").all()


def test_seed_determinism_and_arguments():
    a = synthetic.generate("panc", n_patients=25, seed=11)
    b = synthetic.generate("PANC", n_patients=25, seed=11)
    for name in a.names:
        pd.testing.assert_frame_equal(a[name], b[name])
    c = synthetic.generate("PANC", n_patients=25, seed=12)
    assert not a["regimen"].equals(c["regimen"])
    small = synthetic.generate("NSCLC", n_patients=5, seed=1, tables=["patient", "regimen"], genomic=False)
    assert set(small.names) == {"patient", "regimen"}
    with pytest.raises(ValueError):
        synthetic.generate("NSCLC", n_patients=0)
    with pytest.raises(ValueError):
        synthetic.generate("melanoma")


def test_round_trip_through_loader_matches_dictionary(tmp_path):
    data = synthetic.generate("BLADDER", n_patients=12, seed=2)
    folder = data.to_dir(tmp_path / "BLADDER_v1.0-public")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no column-mismatch warning expected
        back = io.load_release(folder)
    assert back.cohort == "BLADDER" and back.version == "v1.0-public"
    assert back.validation["columns"].get("patient") is None
    assert set(back.names) == set(data.names)
