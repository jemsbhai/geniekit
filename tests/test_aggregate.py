from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geniekit import aggregate, io, lot, synthetic
from geniekit.aggregate import DisclosureControl, stats
from geniekit.aggregate.summaries import _p
from geniekit.io.cohort import CohortData

FIXTURE = Path(__file__).parent / "fixtures" / "nsclc_sim"


# ----------------------------------------------------------------------------- statistics (hand-checked)
def test_wilson_interval_matches_hand_computation():
    # 7 of 20 at z = 1.96: centre (0.35 + 0.09604) / 1.19208 = 0.3742, half-width 0.1930 (Wilson 1927)
    lo, hi = stats.wilson_interval(7, 20)
    assert (round(lo, 3), round(hi, 3)) == (0.181, 0.567)
    assert stats.wilson_interval(0, 10)[0] == 0.0 and round(stats.wilson_interval(10, 10)[1], 12) == 1.0
    assert all(math.isnan(v) for v in stats.wilson_interval(1, 0))


def test_risk_difference():
    d, lo, hi = stats.risk_difference(30, 100, 20, 100)
    assert round(d, 3) == 0.1 and round(lo, 3) == -0.019 and round(hi, 3) == 0.219
    assert all(math.isnan(v) for v in stats.risk_difference(1, 0, 1, 5))


def test_fisher_exact_matches_known_values():
    # the classic tea-tasting table [[3, 1], [1, 3]]: two-sided p = 0.4857
    assert round(stats.fisher_exact(3, 1, 1, 3), 4) == 0.4857
    # [[10, 2], [3, 15]] and [[12, 7], [2, 9]]: two-sided p = 0.000537 and 0.02589 (SciPy fisher_exact)
    assert round(stats.fisher_exact(10, 2, 3, 15), 6) == 0.000537
    assert round(stats.fisher_exact(12, 7, 2, 9), 5) == 0.02589
    assert stats.fisher_exact(0, 0, 0, 0) != stats.fisher_exact(0, 0, 0, 0)  # NaN
    assert stats.fisher_exact(5, 5, 5, 5) == 1.0


def test_chi_square_statistic():
    stat, dof, p = stats.chi_square([[10, 20], [30, 40]])
    # expected 12, 18, 28, 42 -> chi2 = 4/12 + 4/18 + 4/28 + 4/42 = 0.7937
    assert round(stat, 4) == 0.7937 and dof == 1
    if p is not None:
        assert round(p, 3) == 0.373
    assert stats.chi_square([[1, 2]])[1] == 0 and math.isnan(stats.chi_square([[1, 2]])[0])


def test_kaplan_meier_hand_example():
    # times 1, 2+, 3, 4, 5+ (plus = censored):
    # S(1) = 4/5, S(3) = 4/5 * 2/3 = 0.5333, S(4) = 0.5333 * 1/2 = 0.2667
    km = stats.kaplan_meier([1, 2, 3, 4, 5], [1, 0, 1, 1, 0])
    assert km.n == 5 and km.n_events == 3
    assert [round(v, 4) for v in km.table["survival"]] == [0.8, 0.5333, 0.2667]
    assert km.table["at_risk"].tolist() == [5, 3, 2] and km.median == 4.0
    assert km.at(0.5) == (1.0, 1.0, 1.0) and round(km.at(3.5)[0], 4) == 0.5333
    assert km.at_risk(2.5) == 3 and km.at_risk(10) == 0
    # log-log limits at t = 1: Greenwood variance 1/(5*4) = 0.05; se of log(-log S) = sqrt(0.05)/|log 0.8|
    _, lo, hi = km.at(1)
    se = math.sqrt(0.05) / abs(math.log(0.8))
    assert round(lo, 4) == round(0.8 ** math.exp(1.96 * se), 4) and round(hi, 4) == round(
        0.8 ** math.exp(-1.96 * se), 4
    )
    assert math.isnan(stats.kaplan_meier([1, 2], [0, 0]).median)
    # survival hits zero when the last subject has the event
    km0 = stats.kaplan_meier([1, 2], [1, 1])
    assert km0.table["survival"].iloc[-1] == 0 and km0.table["lower"].iloc[-1] == 0


def test_p_formatting():
    assert (
        _p(0.00001) == "<0.0001"
        and _p(0.123456) == 0.1235
        and np.isnan(_p(None))
        and np.isnan(_p(float("nan")))
    )


# ----------------------------------------------------------------------------- disclosure control
def test_disclosure_control_validation_and_percent():
    ctl = DisclosureControl(min_count=5)
    assert ctl.label == "<5" and ctl.denominator == 5 and ctl.describe()["marker"] == "<5"
    assert DisclosureControl(marker="x").label == "x"
    with pytest.raises(ValueError):
        DisclosureControl(min_count=0)
    with pytest.raises(ValueError):
        DisclosureControl(min_denominator=0)
    assert ctl.percent(3, 100) == "<5" and ctl.percent(0, 100) == 0.0 and ctl.percent(50, 100) == 50.0
    assert ctl.percent(2, 3) == "<5" and np.isnan(ctl.percent(1, 0))
    assert DisclosureControl(suppress_zero=True).percent(0, 100) == "<5"
    assert ctl.allows(5) and not ctl.allows(4) and not ctl.allows(float("nan"))


def test_mask_counts_policy():
    ctl = DisclosureControl(min_count=5)
    # one small cell in a 2x2 with both margins: complementary suppression would hide 20 + 40 records,
    # omitting the small cell's row hides 23 -> the row is omitted and the totals recomputed
    counts = pd.DataFrame(
        {"a": [20, 10, 30], "b": [3, 40, 43], "Total": [23, 50, 73]}, index=["x", "y", "Total"]
    )
    masked, mask, records, omitted = ctl.mask_counts(counts, total_row=True, total_col=True)
    assert omitted == {"rows": [("x", 23)], "columns": []} and records == []
    assert masked.index.tolist() == ["y", "Total"] and masked.loc["Total", "Total"] == 50
    assert masked.loc["y", "a"] == 10 and masked.loc["Total", "b"] == 40
    # a tiny column next to large cells is omitted for the same reason
    counts = pd.DataFrame(
        {"a": [200, 100, 300], "b": [3, 1, 4], "c": [50, 60, 110], "Total": [253, 161, 414]},
        index=["x", "y", "Total"],
    )
    masked, mask, records, omitted = ctl.mask_counts(counts, total_row=True, total_col=True)
    assert omitted["columns"] == [("b", 4)] and not records
    assert list(masked.columns) == ["a", "c", "Total"] and masked.loc["Total", "Total"] == 410
    # complementary suppression wins when it is cheaper:
    # only a total row, partner cell of 5 versus a column of 8
    counts = pd.DataFrame({"a": [3, 5, 8], "b": [500, 500, 1000]}, index=["x", "y", "Total"])
    masked, mask, records, omitted = ctl.mask_counts(counts, total_row=True)
    kinds = {(r.row, r.column): r.kind for r in records}
    assert kinds == {("x", "a"): "primary", ("y", "a"): "complementary"} and not omitted["columns"]
    assert masked.loc["x", "a"] == "<5" and masked.loc["y", "a"] == "<5" and masked.loc["Total", "a"] == 8
    assert masked.loc["x", "b"] == 500 and mask.loc["y", "a"] and not mask.loc["Total", "a"]
    # a single column of counts: the small row is omitted (3 records) rather than a partner suppressed (20)
    counts = pd.DataFrame({"Total": [40, 3, 20, 63]}, index=["x", "y", "z", "Total"])
    masked, mask, records, omitted = ctl.mask_counts(counts, total_row=True)
    assert omitted["rows"] == [("y", 3)] and masked.index.tolist() == ["x", "z", "Total"]
    assert masked.loc["Total", "Total"] == 60 and not records
    # no totals: only primary suppression, nothing omitted
    counts = pd.DataFrame({"a": [2, 10], "b": [7, 1]}, index=["x", "y"])
    masked, mask, records, omitted = ctl.mask_counts(counts)
    assert masked.loc["x", "a"] == "<5" and masked.loc["y", "b"] == "<5" and masked.loc["x", "b"] == 7
    assert all(r.kind == "primary" for r in records) and omitted == {"rows": [], "columns": []}
    # zero is never suppressed by default
    counts = pd.DataFrame({"Total": [0, 30, 30]}, index=["x", "y", "Total"])
    masked, *_ = ctl.mask_counts(counts, total_row=True)
    assert masked.loc["x", "Total"] == 0
    # with complementary suppression off, the primary cell alone is hidden
    counts = pd.DataFrame({"a": [3, 5, 8], "b": [500, 500, 1000]}, index=["x", "y", "Total"])
    masked, mask, records, omitted = DisclosureControl(complementary=False).mask_counts(
        counts, total_row=True
    )
    assert masked.loc["x", "a"] == "<5" and masked.loc["y", "a"] == 5 and len(records) == 1


# ----------------------------------------------------------------------------- summaries on a toy frame
@pytest.fixture(scope="module")
def toy_frame():
    rng = np.random.default_rng(0)
    n = 120
    frame = pd.DataFrame(
        {
            "group": rng.choice(["A", "B"], size=n, p=[0.6, 0.4]),
            "sex": rng.choice(["Female", "Male"], size=n),
            "age": rng.normal(60, 10, size=n).round(1),
            "flag": rng.random(n) < 0.3,
            "flag2": rng.random(n) < 0.05,
            "sequenced": rng.random(n) < 0.9,
            "failure_days": rng.exponential(300, size=n).round(),
            "failure_event": (rng.random(n) < 0.7).astype(int),
        }
    )
    frame.loc[:2, "group"] = "C"  # a group of three records
    frame.loc[3:4, "sex"] = None
    return frame


def test_frequency_table(toy_frame):
    t = aggregate.frequency_table(toy_frame, "sex", by="group", control=DisclosureControl(5))
    assert isinstance(t, aggregate.SafeTable) and t.n == 120
    assert t.frame.index.tolist() == ["Female", "Male", "Total"] and "Total n" in t.frame.columns
    assert t.frame.loc["Total", "Total %"] == 100.0 and t.frame.loc["Total", "Total n"] == 115
    # the three-record group C and the two records without a sex are omitted rather than blanking large cells
    assert t.omitted == {"rows": [("Missing", 2)], "columns": [("C", 3)]} and "C n" not in t.frame.columns
    assert sum("omitted" in n for n in t.notes) == 2 and "<5" in t.notes[-1]
    single = aggregate.frequency_table(toy_frame, "sex", control=DisclosureControl(5), dropna=True)
    assert single.frame.columns.tolist() == ["Total n", "Total %"] and "Missing" not in single.frame.index
    assert single.frame["Total n"].iloc[:-1].sum() == single.frame.loc["Total", "Total n"] == 118
    assert "SafeTable(" in repr(single) and single.n_omitted == 0


def test_numeric_summary(toy_frame):
    t = aggregate.numeric_summary(toy_frame, "age", by="group", control=DisclosureControl(5))
    assert t.frame.columns.tolist() == ["n", "median", "q1", "q3"] and "min" not in t.frame.columns
    assert "C" not in t.frame.index and t.frame.loc["Total", "n"] == 117
    assert any("not shown: C" in n for n in t.notes)
    small = aggregate.numeric_summary(toy_frame.head(4), "age", control=DisclosureControl(5))
    assert small.frame.loc["Total", "median"] == "<5" and small.frame.loc["Total", "n"] == 4


def test_proportion_table(toy_frame):
    t = aggregate.proportion_table(toy_frame, "flag", by="group", control=DisclosureControl(5), reference="A")
    f = t.frame
    assert f.index.tolist() == ["A", "B", "Total"]
    assert {
        "n",
        "events",
        "%",
        "ci_lower",
        "ci_upper",
        "risk_difference",
        "rd_lower",
        "rd_upper",
        "p_fisher",
    } <= set(f.columns)
    assert np.isnan(f.loc["A", "risk_difference"]) and isinstance(f.loc["B", "p_fisher"], float | str)
    assert f.loc["Total", "n"] == 117
    # a rare flag: the small event count is suppressed with everything derived from it
    r = aggregate.proportion_table(toy_frame, "flag2", by="group", control=DisclosureControl(10))
    assert "<10" in set(r.frame["%"].astype(str)) and "<10" in set(r.frame["p_fisher"].astype(str))


def test_flag_frequency_table(toy_frame):
    t = aggregate.flag_frequency_table(
        toy_frame, ["flag", "flag2"], by="group", control=DisclosureControl(5), subset="sequenced"
    )
    f = t.frame
    assert f.index.tolist() == ["flag", "flag2"] and "p" in f.columns
    # the total covers sequenced records of the shown groups (group C, three records, is omitted)
    shown = toy_frame[toy_frame["group"] != "C"]
    assert f.loc["flag", "Total n"] == int(shown["sequenced"].sum())
    assert any("Fisher" in n for n in t.notes) and any("sequenced" in n for n in t.notes)
    three = aggregate.flag_frequency_table(
        toy_frame.assign(group=lambda d: d.index % 3), ["flag"], by="group"
    )
    assert any("chi-square" in n for n in three.notes)


def test_km_and_early_failure_tables(toy_frame):
    t = aggregate.km_table(
        toy_frame, "failure_days", "failure_event", by="group", at_months=[6], control=DisclosureControl(5)
    )
    f = t.frame
    assert f.index.tolist() == ["A", "B", "Total"] and {
        "n",
        "events",
        "6 mo %",
        "6 mo ci",
        "6 mo at risk",
        "median months",
    } <= set(f.columns)
    assert 0 <= f.loc["A", "6 mo %"] <= 100
    fail = aggregate.km_table(toy_frame, "failure_days", "failure_event", at_months=[6], failure=True)
    surv = aggregate.km_table(toy_frame, "failure_days", "failure_event", at_months=[6])
    assert round(fail.frame.loc["Total", "6 mo %"] + surv.frame.loc["Total", "6 mo %"], 1) == 100.0
    assert surv.n == 120 and t.frame.loc["Total", "n"] == 117  # the grouped total excludes the omitted group
    e = aggregate.early_failure_table(
        toy_frame, months=(6, 9), by="group", control=DisclosureControl(5), reference="A"
    )
    f = e.frame
    assert f.index.tolist() == ["A", "B", "Total"] and f.loc["Total", "started"] == 117
    assert f.loc["Total", "6 mo evaluable"] <= f.loc["Total", "started"]
    assert abs(f.loc["Total", "6 mo failed %"] - f.loc["Total", "6 mo km failed %"]) < 15
    assert np.isnan(f.loc["A", "6 mo risk_difference"]) and e.settings["months"] == [6, 9]


# ----------------------------------------------------------------------------- genomic builders and table
def _toy_cohort() -> CohortData:
    cancers = pd.DataFrame(
        {
            "record_id": ["P1", "P2", "P3"],
            "ca_seq": [0, 0, 0],
            "institution": ["DFCI", "MSK", "MSK"],
            "dob_ca_dx_days": [20000, 25000, 18000],
            "stage_dx_iv": ["Stage IV", "Stage I-III", "Stage IV"],
            "ca_dmets_yn": ["Yes", None, "Yes"],
            "dmets_stage_i_iii": [None, 1, None],
            "dx_to_dmets_days": [None, 400, None],
            "dmets_liver": [1, 1, 0],
            "dx_to_dmets_liver_days": [0, 500, None],
            "dmets_bone": [0, None, 1],
            "tt_os_dx_days": [800, 900, 300],
            "os_dx_status": [1, 0, 1],
        }
    )
    patient = pd.DataFrame({"record_id": ["P1", "P2", "P3"], "naaccr_sex_code": ["Female", "Male", "Male"]})
    reg = pd.DataFrame(
        {
            "record_id": ["P1", "P1", "P2", "P3"],
            "ca_seq": [0, 0, 0, 0],
            "regimen_number": [1, 2, 1, 1],
            "regimen_number_within_cancer": [1, 2, 1, 1],
            "redcap_ca_index": ["Yes"] * 4,
            "drugs_num": [2, 1, 1, 1],
            "drugs_ct_yn": ["No"] * 4,
            "drugs_dc_ynu": ["Yes", "No", "Yes", "Yes"],
            "regimen_drugs": ["Fluorouracil, Oxaliplatin", "Capecitabine", "Capecitabine", "Fluorouracil"],
            "drugs_drug_1": [
                "Fluorouracil(5FU)",
                "Capecitabine(Xeloda)",
                "Capecitabine(Xeloda)",
                "Fluorouracil(5FU)",
            ],
            "drugs_drug_2": ["Oxaliplatin(Eloxatin)", None, None, None],
            "dx_reg_start_int": [10, 200, 450, 20],
            "dx_reg_end_all_int": [150, None, 600, 100],
            "dx_drug_end_or_lastadm_int_1": [150, 700, 600, 100],
            "dx_drug_end_or_lastadm_int_2": [150, None, None, None],
        }
    )
    imaging = pd.DataFrame(
        {
            "record_id": ["P1", "P3"],
            "image_overall": ["Progressing/Worsening/Enlarging", "Stable/No change"],
            "image_scan_int": [20120, 18050],
            "dx_scan_days": [120, 50],
        }
    )
    cpt = pd.DataFrame(
        {
            "record_id": ["P1", "P2", "P2"],
            "ca_seq": [0, 0, 0],
            "cpt_genie_sample_id": ["S1", "S2", "S3"],
            "cpt_seq_assay_id": ["PANEL-A", "PANEL-B", "PANEL-B"],
            "sample_type": ["Primary", "Primary", "Metastasis"],
        }
    )
    mutations = pd.DataFrame(
        {
            "Hugo_Symbol": ["KRAS", "BRAF", "TP53", "APC", "APC"],
            "Tumor_Sample_Barcode": ["S1", "S2", "S3", "S1", "S9"],
            "Variant_Classification": [
                "Missense_Mutation",
                "Missense_Mutation",
                "Silent",
                "Nonsense_Mutation",
                "Missense_Mutation",
            ],
            "HGVSp_Short": ["p.G12D", "p.V600E", "p.R175H", "p.R1450*", "p.E1309fs"],
        }
    )
    pathology = pd.DataFrame(
        {
            "record_id": ["P1", "P2", "P2", "P3"],
            "msi_result": ["MSS: STABLE", None, "MSI-H: HIGH", "Indeterminate/Not stated"],
            "mmr_result": [None, "No loss of nuclear expression of MMR proteins: proficient", None, None],
        }
    )
    return CohortData(
        cohort="CRC",
        version="v2.0-public",
        tables={
            "patient": patient,
            "cancer_index": cancers,
            "regimen": reg,
            "imaging": imaging,
            "cpt": cpt,
            "mutations": mutations,
            "pathology": pathology,
        },
        synthetic=True,
    )


def test_gene_flags_and_msi_status():
    data = _toy_cohort()
    flags = aggregate.gene_flags(
        data,
        ["KRAS", "TP53", "APC"],
        variants={"BRAF_V600E": "BRAF p.V600E"},
        combined={"RAS": ["KRAS", "NRAS"]},
    )
    f = flags.set_index("record_id")
    assert f.index.tolist() == ["P1", "P2"] and f["n_samples"].tolist() == [1, 2]
    assert bool(f.loc["P1", "KRAS_mut"]) and not bool(f.loc["P2", "KRAS_mut"])
    assert not bool(f.loc["P2", "TP53_mut"])  # silent variants do not count
    assert bool(f.loc["P1", "APC_mut"]) and not bool(f.loc["P2", "APC_mut"])  # S9 is not a cpt sample
    assert bool(f.loc["P2", "BRAF_V600E"]) and bool(f.loc["P1", "RAS"]) and not bool(f.loc["P2", "RAS"])
    status = aggregate.msi_mmr_status(data).set_index("record_id")["msi_mmr"]
    assert status.to_dict() == {"P1": "MSS or pMMR", "P2": "MSI-H or dMMR", "P3": "indeterminate"}
    samples = aggregate.sample_patients(data)
    assert samples["sample_id"].tolist() == ["S1", "S2", "S3"]
    with pytest.raises(KeyError):
        aggregate.gene_flags(CohortData("CRC", tables={"cpt": data["cpt"]}), ["KRAS"])


def test_analytic_table_and_first_line_failure():
    data = _toy_cohort()
    table = aggregate.analytic_table(data, genes=["KRAS"], variants={"BRAF_V600E": "BRAF p.V600E"}, msi=True)
    t = table.set_index("record_id")
    assert len(t) == 3 and t.loc["P1", "sex"] == "Female" and round(t.loc["P1", "age_dx_years"], 1) == 54.8
    assert t.loc["P1", "index_source"] == "stage IV at diagnosis" and t.loc["P2", "index_days"] == 400
    assert (
        t.loc["P1", "liver_mets_timing"] == "at or before index"
        and t.loc["P2", "liver_mets_timing"] == "after index"
    )
    assert t.loc["P3", "liver_mets_timing"] == "none" and bool(t.loc["P3", "mets_bone"])
    assert (
        t["n_lines"].tolist() == [2, 1, 1] and t.loc["P1", "first_line_agents"] == "Fluorouracil;Oxaliplatin"
    )
    # P1: progression on day 120 is the first failure after the line start on day 10 -> 110 days
    assert t.loc["P1", "failure_days"] == 110 and t.loc["P1", "failure_type"] == "progression"
    # P2: no progression, no next line, alive -> censored at follow-up 900 - 450
    assert (
        t.loc["P2", "failure_days"] == 450
        and t.loc["P2", "failure_event"] == 0
        and t.loc["P2", "failure_type"] == "censored"
    )
    # P3: died on day 300, line started on day 20 -> death at 280
    assert t.loc["P3", "failure_days"] == 280 and t.loc["P3", "failure_type"] == "death"
    assert (
        bool(t.loc["P1", "sequenced"])
        and not bool(t.loc["P3", "sequenced"])
        and pd.isna(t.loc["P3", "KRAS_mut"])
    )
    assert t.loc["P2", "msi_mmr"] == "MSI-H or dMMR" and t.loc["P1", "os_from_index_days"] == 800
    assert aggregate.metastasis_sites(data["cancer_index"]) == ["dmets_liver", "dmets_bone"]


# ----------------------------------------------------------------------------- the CRC report
def test_liver_metastasis_report_on_synthetic_crc(tmp_path):
    crc = synthetic.generate("CRC", n_patients=300, seed=7)
    report = aggregate.liver_metastasis_report(crc, control=DisclosureControl(min_count=5))
    assert isinstance(report, aggregate.AggregateReport) and len(report) >= 14
    keys = set(report)
    assert {
        "patients_by_liver_mets",
        "early_failure_by_liver_mets",
        "alterations_by_liver_mets",
        "msi_mmr_by_liver_mets",
    } <= keys
    summary = report.summary()
    assert set(summary.columns) == {"table", "rows", "columns", "n", "suppressed"}
    # every cell of every table is a number, a marker, a short text or missing: never an identifier
    for key in report:
        frame = report[key].frame
        for value in frame.to_numpy().ravel():
            assert not (isinstance(value, str) and value.startswith("GENIE-")), (key, value)
    assert report.settings["index_event"] == "advanced" and report.settings["disclosure"]["min_count"] == 5
    folder = report.to_dir(tmp_path / "crc")
    manifest = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert set(manifest["tables"]) == keys and (folder / "early_failure_by_liver_mets.csv").exists()
    assert "AggregateReport(" in repr(report)
    # a stricter threshold suppresses or omits more, never less
    strict = aggregate.liver_metastasis_report(crc, control=DisclosureControl(min_count=20))
    assert (
        strict.summary()["suppressed"].sum() + sum(t.n_omitted for t in strict.tables.values())
        >= summary["suppressed"].sum()
    )


def test_report_accepts_precomputed_lines_and_other_cohorts():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx = io.load_release(FIXTURE, genomic=True)
    lines = lot.derive(fx, index="advanced")
    report = aggregate.liver_metastasis_report(
        fx, lines=lines, genes=["EGFR", "KRAS"], variants={}, combined={}
    )
    assert report.settings["cohort"] == "NSCLC v2.0-public" and "alterations_by_liver_mets" in set(report)
    assert report["alterations_by_liver_mets"].frame.index.tolist() == ["EGFR_mut", "KRAS_mut"]


@pytest.mark.skipif(not os.environ.get("GENIEKIT_DATA_DIR"), reason="GENIEKIT_DATA_DIR not set")
def test_real_crc_report_is_aggregate_only():
    root = Path(os.environ["GENIEKIT_DATA_DIR"]) / "CRC_v2.0-public"
    if not root.exists():
        pytest.skip("CRC release not downloaded")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crc = io.load_release(root)
    report = aggregate.liver_metastasis_report(crc)
    for key in report:
        frame = report[key].frame
        assert len(frame) <= 25, key
        for value in frame.to_numpy().ravel():
            assert not (isinstance(value, str) and value.startswith("GENIE-")), (key, value)
    ef = report["early_failure_by_liver_mets"].frame
    assert ef.loc["Total", "started"] > 1000


def test_example_notebook_runs_on_synthetic_data(tmp_path, monkeypatch):
    """The notebook's code cells execute in order without a data folder (synthetic cohort)."""
    notebook = Path(__file__).parent.parent / "examples" / "crc_liver_metastasis.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    monkeypatch.delenv("GENIEKIT_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"display": lambda _: None}
    for cell in cells:
        if cell["cell_type"] == "code":
            exec(cell["source"], namespace)
    assert (
        "SYNTHETIC" in str(namespace["source"])
        and (tmp_path / "outputs" / "crc_liver_metastasis" / "report.json").exists()
    )
