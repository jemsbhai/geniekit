from __future__ import annotations

import json
import re

import pytest

from geniekit import releases

SYN = re.compile(r"^syn\d{6,}$")


def test_six_public_releases_in_canonical_order(all_releases):
    assert [r.cohort for r in all_releases] == list(releases.COHORTS)
    assert all(r.version.endswith("-public") for r in all_releases)
    assert {r.version for r in all_releases} == {"v2.0-public", "v1.0-public"}


def test_registry_metadata_present():
    raw = releases.load_registry()
    assert raw["source"]["synapse_data_releases_folder"] == "syn27056700"
    assert raw["source"]["access_requirement_id"] == "9606645"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", releases.registry_generated_on())


@pytest.mark.parametrize(
    "cohort,folder,n",
    [
        ("NSCLC", "syn27056697", 1846),
        ("CRC", "syn30991637", 1485),
        ("BrCa", "syn71825332", 1130),
        ("PANC", "syn72666404", 1109),
        ("Prostate", "syn73719283", 1116),
        ("BLADDER", "syn75201334", 716),
    ],
)
def test_release_folders_and_patient_counts(cohort, folder, n):
    r = releases.get_release(cohort)
    assert r.release_folder == folder
    assert r.n_patients == n
    assert r.synapse_url.endswith(folder)


def test_every_release_has_core_clinical_and_genomic_files(all_releases):
    core = {
        "patient",
        "cancer_index",
        "cancer_non_index",
        "regimen",
        "imaging",
        "pathology",
        "med_onc",
        "cpt",
    }
    for r in all_releases:
        assert core <= set(r.clinical), r.label
        assert {"mutations", "cna", "cna_seg", "gene_matrix"} <= set(r.genomic), r.label
        assert {"data_guide", "variable_synopsis"} <= set(r.documentation), r.label
        for f in r.files:
            assert SYN.match(f.synapse_id), (r.label, f)


def test_schema_generations_and_optional_tables(all_releases):
    by = {r.cohort: r for r in all_releases}
    assert {c: by[c].schema_generation for c in releases.COHORTS} == {
        "NSCLC": 1,
        "CRC": 1,
        "BrCa": 2,
        "PANC": 2,
        "Prostate": 2,
        "BLADDER": 2,
    }
    # tumor markers: CRC, BrCa, PANC, Prostate; radiation: PANC, Prostate, BLADDER
    assert [c for c in releases.COHORTS if by[c].has("tumor_marker")] == ["CRC", "BrCa", "PANC", "Prostate"]
    assert [c for c in releases.COHORTS if by[c].has("radiation")] == ["PANC", "Prostate", "BLADDER"]
    # structural variants: fusions (generation 1) vs sv (generation 2)
    assert [c for c in releases.COHORTS if by[c].has("fusions")] == ["NSCLC", "CRC"]
    assert [c for c in releases.COHORTS if by[c].has("sv")] == ["BrCa", "PANC", "Prostate", "BLADDER"]


def test_institutions_match_known_contributors(all_releases):
    by = {r.cohort: r for r in all_releases}
    assert by["CRC"].institutions == ("DFCI", "MSK", "VICC")
    assert by["BrCa"].institutions == ("DFCI", "MSK", "VICC")
    for c in ("NSCLC", "PANC", "Prostate", "BLADDER"):
        assert by[c].institutions == ("DFCI", "MSK", "UHN", "VICC")


def test_cbioportal_ids(all_releases):
    for r in all_releases:
        assert r.cbioportal_study_id == f"{r.cohort.lower()}_public_genie_bpc"
        assert r.cbioportal_url.endswith(r.cbioportal_study_id)


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("nsclc", "NSCLC"),
        ("Non-Small Cell Lung Cancer", "NSCLC"),
        ("BRCA", "BrCa"),
        ("breast cancer", "BrCa"),
        ("pancreas", "PANC"),
        ("PROSTATE", "Prostate"),
        ("bladder", "BLADDER"),
        ("  crc ", "CRC"),
    ],
)
def test_normalize_cohort(alias, expected):
    assert releases.normalize_cohort(alias) == expected


def test_normalize_cohort_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown cohort"):
        releases.normalize_cohort("RENAL")


def test_get_release_errors():
    with pytest.raises(KeyError, match="No public release"):
        releases.get_release("NSCLC", "v9.9-public")
    with pytest.raises(ValueError):
        releases.get_release("melanoma")


def test_file_lookup_and_groups():
    r = releases.get_release("PANC")
    f = r.file("patient")
    assert f.filename == "patient_level_dataset.csv" and f.group == "clinical"
    assert r.file("mutations").filename == "data_mutations_extended.txt"
    assert r.file("sv", group="genomic").filename == "data_sv.txt"
    assert r.file("data_guide").group == "documentation"
    assert "sequencing" in r.files_in("cbioportal_timeline")
    assert any(p.name.startswith("UHN") for p in r.files_in("gene_panels").values())
    with pytest.raises(KeyError):
        r.file("fusions")
    with pytest.raises(ValueError):
        r.files_in("nope")
    assert f.synapse_url == f"https://www.synapse.org/Synapse:{f.synapse_id}"


def test_to_frame_is_one_row_per_file(all_releases):
    frame = releases.to_frame()
    assert len(frame) == sum(len(r.files) for r in all_releases)
    assert frame["synapse_id"].is_unique
    assert set(frame["group"]) <= set(releases.FILE_GROUPS)


def test_to_dict_round_trips_through_json():
    r = releases.get_release("CRC")
    d = json.loads(json.dumps(r.to_dict()))
    assert d["cohort"] == "CRC" and len(d["files"]) == len(r.files)


def test_geniebpc_alias_table_covers_all_clinical_tables(all_releases):
    names = {name for r in all_releases for name in r.clinical}
    assert names <= set(releases.GENIEBPC_TABLE_NAMES)
