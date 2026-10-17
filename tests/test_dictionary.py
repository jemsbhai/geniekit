from __future__ import annotations

import pandas as pd
import pytest

from geniekit import dictionary, releases


def test_load_is_cached_and_well_formed(dd):
    assert dictionary.load() is dd
    assert dd.cohorts == releases.COHORTS
    assert len(dd.frame) == 4771
    assert set(dd.tables()) == set(dictionary.TABLES)
    assert isinstance(dd.frame["values"].iloc[0], list)
    assert dd.frame.duplicated(["cohort", "table", "variable"]).sum() == 0


def test_row_counts_per_cohort_match_synopses(dd):
    counts = dd.frame.groupby("cohort").size().to_dict()
    assert counts == {"NSCLC": 565, "CRC": 740, "BrCa": 872, "PANC": 865, "Prostate": 839, "BLADDER": 890}


def test_schema_generation(dd):
    assert dd.schema_generation("nsclc") == 1
    assert dd.schema_generation("BLADDER") == 2
    with pytest.raises(ValueError):
        dd.schema_generation("kidney")


def test_universal_identifiers_present_everywhere(dd):
    for table in ("patient", "cancer", "regimen", "imaging", "pathology", "med_onc", "cpt"):
        common = dd.common_variables(table)
        assert {"record_id", "institution", "cohort"} <= set(common), table


def test_presence_matrix_shape_and_values(dd):
    mat = dd.presence("patient")
    assert list(mat.columns) == list(releases.COHORTS)
    assert mat.dtypes.eq(bool).all()
    assert mat.loc[("patient", "record_id")].all()
    assert mat.loc[("patient", "race_ethnicity")].tolist() == [True, True, False, False, False, False]
    assert mat.loc[("patient", "muscprop_invasive_pt")].tolist() == [False] * 5 + [True]
    whole = dd.presence()
    assert len(whole) == 1105
    assert whole.all(axis=1).sum() == 505


def test_presence_subset_of_cohorts(dd):
    mat = dd.presence("regimen", cohorts=["crc", "bladder"])
    assert list(mat.columns) == ["CRC", "BLADDER"]
    assert mat.loc[("regimen", "drugs_admin")].all()


def test_compare_rename_pair_smoking(dd):
    old = dd.compare("ca_lung_cigarette", "cancer")
    new = dd.compare("ca_cigarette", "cancer")
    assert old.present_in == ("NSCLC", "CRC")
    assert new.present_in == ("BrCa", "PANC", "Prostate", "BLADDER")
    assert not old.is_universal and not new.is_universal


def test_compare_dtype_conflict(dd):
    cmp = dd.compare("naaccr_seer_sum_stage", "cancer")
    assert cmp.is_universal
    assert cmp.dtype_conflict
    assert cmp.dtypes["NSCLC"] == "Character" and cmp.dtypes["BLADDER"] == "Numeric"
    assert cmp.value_set_conflict
    assert "value sets differ" in cmp.summary() and "dtype conflict" in cmp.summary()


def test_compare_value_text_only_conflict(dd):
    cmp = dd.compare("ca_type_brain", "cancer")
    assert cmp.is_universal
    assert not cmp.value_set_conflict
    assert cmp.value_text_conflict
    # each cohort declares 20 levels; two differ only by case ("Mixed Gliomas" vs "Mixed gliomas")
    assert len(cmp.union_values) == 20
    assert all(len(v) == 20 for v in cmp.values.values())


def test_compare_identical_everywhere(dd):
    cmp = dd.compare("record_id", "patient")
    assert cmp.is_universal
    assert not (cmp.dtype_conflict or cmp.value_set_conflict or cmp.value_text_conflict or cmp.label_conflict)


def test_compare_unknown_variable(dd):
    with pytest.raises(KeyError):
        dd.compare("not_a_variable", "patient")
    with pytest.raises(ValueError):
        dd.compare("record_id", "not_a_table")


def test_values_per_cohort(dd):
    vals = dd.values("institution", "patient")
    assert set(vals) == set(releases.COHORTS)
    assert len(vals["CRC"]) == 3 and len(vals["NSCLC"]) == 4


def test_differences_frame(dd):
    diff = dd.differences()
    assert set(diff.columns) == {"table", "variable", "kind", "present_in", "detail"}
    kinds = set(diff["kind"])
    assert kinds <= {"missing_in_some", "dtype", "value_set", "value_text", "label"}
    assert {"missing_in_some", "dtype", "value_set", "value_text"} <= kinds
    dtype_rows = diff[diff["kind"] == "dtype"]
    assert set(dtype_rows["variable"]) == {"naaccr_seer_sum_stage", "cpt_seq_date"}
    missing = diff[diff["kind"] == "missing_in_some"]
    assert len(missing) == 1105 - 505
    only_dtype = dd.differences(kinds=["dtype"])
    assert set(only_dtype["kind"]) == {"dtype"}
    patient_only = dd.differences(table="patient")
    assert set(patient_only["table"]) == {"patient"}


def test_normalize_level():
    n = dictionary.normalize_level
    assert n("Mixed Gliomas") == n("Mixed gliomas")
    assert n("Mammogram - Use for Breast Cancer only") == n(
        "Mammogram " + chr(0x2013) + " Use for Breast Cancer only"
    )
    assert n("Former user (quit < 1 year)") != n("Former user (quit <1 year)")


def test_lookup_without_table_spans_tables(dd):
    rows = dd.lookup("record_id")
    assert set(rows["table"]) == set(dictionary.TABLES)
    assert dd.lookup("record_id", "cpt")["cohort"].tolist() == list(releases.COHORTS)


def test_constructor_validates_columns():
    with pytest.raises(ValueError, match="lacks columns"):
        dictionary.DataDictionary(pd.DataFrame({"cohort": []}))
