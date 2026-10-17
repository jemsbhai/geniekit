from __future__ import annotations

import gzip
import shutil
import warnings
from pathlib import Path

import pandas as pd
import pytest

from geniekit import io
from geniekit.io import local

FIXTURE = Path(__file__).parent / "fixtures" / "nsclc_sim"


@pytest.fixture(scope="module")
def nsclc() -> io.CohortData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return io.load_release(FIXTURE)


def test_filename_map_covers_all_release_files():
    from geniekit import releases

    for rel in releases.list_releases():
        for ref in rel.clinical.values():
            assert local.FILENAME_TO_TABLE[ref.filename] in io.CLINICAL_TABLES, ref
        for name, ref in rel.genomic.items():
            assert local._logical_name(Path(ref.filename)) == name, ref
    assert local.FILENAME_TO_TABLE["patient.csv"] == "patient"
    assert local._logical_name(Path("genie_crc_data_cna_hg19.seg")) == "cna_seg"
    assert local._logical_name(Path("random.csv")) is None


def test_find_release_files_fixture():
    files = io.find_release_files(FIXTURE)
    assert set(files) == {
        "patient",
        "cancer_index",
        "cancer_non_index",
        "regimen",
        "imaging",
        "pathology",
        "med_onc",
        "cpt",
        "mutations",
        "fusions",
        "cna",
    }
    assert files["patient"].name == "patient_level_dataset.csv.gz"
    subset = io.find_release_files(FIXTURE, tables=["patient", "regimen"])
    assert set(subset) == {"patient", "regimen"}


def test_find_release_files_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.find_release_files(tmp_path / "nope")
    (tmp_path / "x.txt").write_text("nothing", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No GENIE BPC release files"):
        io.find_release_files(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for d in ("a", "b"):
        shutil.copy(FIXTURE / "clinical_data" / "patient_level_dataset.csv.gz", tmp_path / d)
    with pytest.raises(ValueError, match="matches 2 files"):
        io.find_release_files(tmp_path)


def test_load_release_infers_cohort_and_version(nsclc):
    assert nsclc.cohort == "NSCLC"
    assert nsclc.version == "v2.0-public"
    assert nsclc.label == "NSCLC v2.0-public"
    assert nsclc.release is not None and nsclc.release.release_folder == "syn27056697"
    assert len(nsclc) == 11 and "patient" in nsclc and "radiation" not in nsclc
    assert nsclc.names[:3] == ["patient", "cancer_index", "cancer_non_index"]
    assert nsclc.names[-3:] == ["mutations", "fusions", "cna"]


def test_loaded_frames_match_fixture_shapes(nsclc):
    assert nsclc["patient"].shape == (200, 36)
    assert nsclc["cancer_index"].shape == (207, 141)
    assert nsclc["regimen"].shape == (473, 102)
    assert nsclc["mutations"].shape == (1945, 64)
    assert nsclc["cna"].shape == (964, 195)
    assert nsclc["fusions"].shape == (70, 9)
    assert nsclc["mutations"].columns[0] == "Hugo_Symbol"  # '#version' line skipped
    assert nsclc.patient_ids().size == 200
    assert nsclc["patient"]["birth_year"].dtype.kind == "i"
    assert nsclc["regimen"]["dx_reg_start_int"].dtype.kind in "if"


def test_na_handling_mirrors_r_package(tmp_path):
    csv = tmp_path / "patient_level_dataset.csv"
    csv.write_text(
        "cohort,record_id,x,y\nNSCLC,GENIE-TEST-1,NA,None\nNSCLC,GENIE-TEST-2,,N/A\n", encoding="utf-8"
    )
    frame = io.read_table(csv)
    assert frame["x"].isna().all()
    # 'None' and 'N/A' are kept as text, as read.csv(na.strings = c("", "NA")) would
    assert frame["y"].tolist() == ["None", "N/A"]


def test_read_table_tab_and_gzip(tmp_path):
    txt = tmp_path / "data_sv.txt.gz"
    with gzip.open(txt, "wt", encoding="utf-8") as fh:
        fh.write("#meta line\n#another\nSample_Id\tSite1_Hugo_Symbol\nS1\tALK\n")
    frame = io.read_table(txt)
    assert list(frame.columns) == ["Sample_Id", "Site1_Hugo_Symbol"] and len(frame) == 1


def test_validation_report_and_warning(nsclc):
    report = nsclc.validation["columns"]
    # the simulated cancer_index lacks the non-index-only columns described by the synopsis
    assert "ca_heme_type" in report["cancer_index"]["missing"]
    assert report["cancer_index"]["extra"] == []
    # the simulated cpt table, like the released one, lacks cpt_report_post_last_alive, which the
    # dictionary records as declared but not released: reported, not warned about
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = io.load_release(FIXTURE, tables=["patient", "cpt", "cancer_index"], genomic=False)
    assert data.validation["columns"].keys() == {"cancer_index", "cpt"}
    assert data.validation["columns"]["cpt"] == {
        "missing": [],
        "extra": [],
        "not_released": ["cpt_report_post_last_alive"],
    }
    # a column that is declared, released and absent is warned about
    data.tables["patient"] = data.tables["patient"].drop(columns=["naaccr_sex_code"])
    report2 = local._validate_columns(data)
    assert report2["patient"]["missing"] == ["naaccr_sex_code"]
    assert local._validation_warning(report2) == "patient: 1 missing, 0 unexpected"
    assert local._validation_warning({"cancer_non_index": {"missing": ["a"], "extra": []}}) is None
    assert (
        local._validation_warning({"patient": {"missing": [], "extra": ["zzz"]}})
        == "patient: 0 missing, 1 unexpected"
    )


def test_load_release_explicit_cohort_and_no_validation(tmp_path):
    (tmp_path / "patient.csv").write_text("record_id,x\nGENIE-TEST-1,1\n", encoding="utf-8")
    data = io.load_release(tmp_path, cohort="crc", version="2.0-public", validate=False)
    assert data.cohort == "CRC" and data.version == "v2.0-public"
    assert data.validation == {}
    with pytest.raises(ValueError, match="Could not infer the cohort"):
        io.load_release(tmp_path, validate=False)


def test_load_release_from_folder_name(tmp_path):
    folder = tmp_path / "PANC_v1.0-public"
    folder.mkdir()
    (folder / "patient_level_dataset.csv").write_text("record_id,x\nGENIE-TEST-1,1\n", encoding="utf-8")
    data = io.load_release(folder, validate=False)
    assert data.cohort == "PANC" and data.version == "v1.0-public"


def test_cohort_data_roundtrip_and_summary(nsclc, tmp_path):
    out = nsclc.to_dir(tmp_path / "out", compress=True)
    again = io.load_release(out, validate=False)
    assert again.cohort == "NSCLC" and set(again.names) == set(nsclc.names)
    pd.testing.assert_frame_equal(again["patient"], nsclc["patient"])
    summary = nsclc.summary()
    assert list(summary.columns) == ["table", "rows", "columns", "patients", "source"]
    assert int(summary.loc[summary["table"] == "regimen", "rows"].iloc[0]) == 473
    assert repr(nsclc) == "CohortData(NSCLC v2.0-public: 11 tables)"
    with pytest.raises(KeyError, match="has no table"):
        nsclc["radiation"]
    assert nsclc.get("radiation") is None
    assert set(nsclc.clinical) | set(nsclc.genomic) == set(nsclc.names)


def test_manifest_is_picked_up(tmp_path):
    folder = tmp_path / "CRC_v2.0-public"
    (folder / "clinical").mkdir(parents=True)
    (folder / "clinical" / "patient_level_dataset.csv").write_text(
        "record_id\nGENIE-TEST-1\n", encoding="utf-8"
    )
    (folder / "manifest.json").write_text('{"cohort": "CRC", "files": []}', encoding="utf-8")
    data = io.load_release(folder, validate=False)
    assert data.validation["manifest"]["cohort"] == "CRC"
