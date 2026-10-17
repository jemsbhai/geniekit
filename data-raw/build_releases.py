"""Build ``src/geniekit/resources/releases.json`` from the public Synapse entity tree.

The AACR Project GENIE BPC "Data Releases" folder (syn27056700) exposes its
folder and file *metadata* without authentication through the Synapse REST
API. Downloading the files themselves requires an approved GENIE access
requirement, but names and Synapse IDs are public, which is all this registry
needs.

Run from the repository root::

    python data-raw/build_releases.py

The script is deterministic for a given state of Synapse. Re-run it when a new
public release appears; the diff of ``releases.json`` is then the review
artefact.

Cohort-level facts that are not available from the entity tree (patient
counts, contributing institutions, cBioPortal study identifiers) are taken from
the AACR Project GENIE cohort pages and recorded with their source URL.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ENDPOINT = "https://repo-prod.prod.sagebase.org/repo/v1"
DATA_RELEASES_FOLDER = "syn27056700"
BPC_PROJECT = "syn27056172"
ACCESS_REQUIREMENT_ID = "9606645"

# Facts from the AACR Project GENIE cohort pages (checked 2026-10-17).
COHORT_FACTS = {
    "NSCLC": {
        "full_name": "Non-Small Cell Lung Cancer",
        "n_patients": 1846,
        "institutions": ["DFCI", "MSK", "UHN", "VICC"],
        "cbioportal_study_id": "nsclc_public_genie_bpc",
        "aacr_page": "https://aacrprojectgenie.org/data/bpc-nsclc-v2-0-public/",
        "data_guide_url": "https://www.aacr.org/wp-content/uploads/2022/05/GENIE-BPC-NSCLC-v2.0-public-Analytic-Data-Guide-1.pdf",
        "variable_synopsis_url": "https://www.aacr.org/wp-content/uploads/2022/05/BPC_NSCLC_v2.0-public_variable_synopsis_posted.xlsx",
    },
    "CRC": {
        "full_name": "Colorectal Cancer",
        "n_patients": 1485,
        "institutions": ["DFCI", "MSK", "VICC"],
        "cbioportal_study_id": "crc_public_genie_bpc",
        "aacr_page": "https://aacrprojectgenie.org/data/bpc-crc-v2-0-public/",
        "data_guide_url": "https://www.aacr.org/wp-content/uploads/2022/11/GENIE-BPC-CRC-v2.0-public-Analytic-Data-Guide-v2.pdf",
        "variable_synopsis_url": "https://www.aacr.org/wp-content/uploads/2022/09/BPC_CRC_v2.0-public_variable_synopsis-xlxm.xlsx",
    },
    "BrCa": {
        "full_name": "Breast Cancer (early onset)",
        "n_patients": 1130,
        "institutions": ["DFCI", "MSK", "VICC"],
        "cbioportal_study_id": "brca_public_genie_bpc",
        "aacr_page": "https://aacrprojectgenie.org/data/bpc-brca-v1-0-public/",
        "data_guide_url": "https://www.aacr.org/wp-content/uploads/2025/12/GENIE-BPC-BrCa_v1.0-public-Analytic-Data-Guide.pdf",
        "variable_synopsis_url": "https://www.aacr.org/wp-content/uploads/2025/12/GENIE-BPC-BrCa_v1.0-public-Variable-Synopsis.xlsx",
    },
    "PANC": {
        "full_name": "Pancreatic Cancer",
        "n_patients": 1109,
        "institutions": ["DFCI", "MSK", "UHN", "VICC"],
        "cbioportal_study_id": "panc_public_genie_bpc",
        "aacr_page": "https://aacrprojectgenie.org/data/panc-1-0-public/",
        "data_guide_url": "https://www.aacr.org/wp-content/uploads/2026/03/GENIE-BPC-PANC-1.0-public-Analytic-Data-Guide.pdf",
        "variable_synopsis_url": "https://www.aacr.org/wp-content/uploads/2026/03/GENIE-BPC-PANC-v1.0-public-Variable-Synopsis.xlsx",
    },
    "Prostate": {
        "full_name": "Prostate Cancer",
        "n_patients": 1116,
        "institutions": ["DFCI", "MSK", "UHN", "VICC"],
        "cbioportal_study_id": "prostate_public_genie_bpc",
        "aacr_page": "https://aacrprojectgenie.org/data/prostate-1-0-public/",
        "data_guide_url": "https://www.aacr.org/wp-content/uploads/2026/03/GENIE-BPC-Prostate-v1.0-public-Analytic-Data-Guide.pdf",
        "variable_synopsis_url": "https://www.aacr.org/wp-content/uploads/2026/03/GENIE-BPC-Prostate-v1.0-public-Variable-Synopsis.xlsx",
    },
    "BLADDER": {
        "full_name": "Bladder Cancer",
        "n_patients": 716,
        "institutions": ["DFCI", "MSK", "UHN", "VICC"],
        "cbioportal_study_id": "bladder_public_genie_bpc",
        "aacr_page": "https://aacrprojectgenie.org/data/bpc-bladder-v1-0-public/",
        "data_guide_url": "https://aacrprojectgenie.org/wp-content/uploads/2026/09/GENIE-BPC-BLADDER-1.0-public-Analytic-Data-Guide.pdf",
        "variable_synopsis_url": "https://aacrprojectgenie.org/wp-content/uploads/2026/08/GENIE-BPC-Bladder-v1.0-public-Variable-Synopsis.xlsx",
    },
}

# Cohorts whose clinical data model predates the 2025 redesign (see the
# dictionary module for the measured differences).
SCHEMA_GENERATION = {"NSCLC": 1, "CRC": 1, "BrCa": 2, "PANC": 2, "Prostate": 2, "BLADDER": 2}

CLINICAL_FILES = {
    "patient_level_dataset.csv": "patient",
    "cancer_level_dataset_index.csv": "cancer_index",
    "cancer_level_dataset_non_index.csv": "cancer_non_index",
    "regimen_cancer_level_dataset.csv": "regimen",
    "imaging_level_dataset.csv": "imaging",
    "pathology_report_level_dataset.csv": "pathology",
    "med_onc_note_level_dataset.csv": "med_onc",
    "tm_level_dataset.csv": "tumor_marker",
    "ca_radtx_dataset.csv": "radiation",
    "cancer_panel_test_level_dataset.csv": "cpt",
}

GENOMIC_FILES = {
    "data_mutations_extended.txt": "mutations",
    "data_fusions.txt": "fusions",
    "data_sv.txt": "sv",
    "data_CNA.txt": "cna",
    "genomic_information.txt": "genomic_information",
    "data_gene_matrix.txt": "gene_matrix",
    "tmb.tsv": "tmb",
}

CBIO_CLINICAL_FILES = {
    "data_clinical_patient.txt": "clinical_patient",
    "data_clinical_sample.txt": "clinical_sample",
    "data_clinical_supp_survival.txt": "supp_survival",
    "data_clinical_supp_survival_treatment.txt": "supp_survival_treatment",
}


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def _children(parent: str) -> list[dict]:
    body = json.dumps(
        {"parentId": parent, "includeTypes": ["folder", "file"], "includeTotalChildCount": True}
    ).encode()
    req = urllib.request.Request(
        f"{REPO_ENDPOINT}/entity/children",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    out: list[dict] = []
    with urllib.request.urlopen(req, timeout=60) as resp:
        page = json.load(resp)
    out.extend(page.get("page", []))
    token = page.get("nextPageToken")
    while token:
        body = json.dumps(
            {"parentId": parent, "includeTypes": ["folder", "file"], "nextPageToken": token}
        ).encode()
        req = urllib.request.Request(
            f"{REPO_ENDPOINT}/entity/children",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            page = json.load(resp)
        out.extend(page.get("page", []))
        token = page.get("nextPageToken")
    return out


def _walk(parent: str) -> list[dict]:
    """Return a flat list of entities with a ``path`` of folder names."""
    result: list[dict] = []

    def rec(pid: str, path: list[str]) -> None:
        for child in _children(pid):
            is_folder = child["type"].endswith("Folder")
            entry = {
                "id": child["id"],
                "name": child["name"],
                "is_folder": is_folder,
                "path": path,
            }
            result.append(entry)
            if is_folder:
                rec(child["id"], [*path, child["name"]])

    rec(parent, [])
    return result


def _classify_version(name: str) -> str:
    """``1.0-public`` or ``2.0-public`` -> ``v1.0-public``."""
    m = re.fullmatch(r"v?(\d+\.\d+)-(public|consortium)", name)
    if not m:
        raise ValueError(f"Unrecognised version folder name: {name!r}")
    return f"v{m.group(1)}-{m.group(2)}"


def build_release(cohort: str, cohort_folder: dict, version_folder: dict, entities: list[dict]) -> dict:
    version = _classify_version(version_folder["name"])
    release_meta = _get(f"{REPO_ENDPOINT}/entity/{version_folder['id']}")
    files_in_release = [
        e
        for e in entities
        if not e["is_folder"] and e["path"][:2] == [cohort_folder["name"], version_folder["name"]]
    ]
    clinical: dict[str, dict] = {}
    genomic: dict[str, dict] = {}
    cbio_clinical: dict[str, dict] = {}
    timelines: dict[str, dict] = {}
    gene_panels: list[dict] = []
    case_lists: list[dict] = []
    meta_files: list[dict] = []
    documentation: dict[str, dict] = {}
    other: list[dict] = []

    for e in files_in_release:
        subfolder = e["path"][2] if len(e["path"]) > 2 else ""
        rec = {"synapse_id": e["id"], "filename": e["name"]}
        name = e["name"]
        if subfolder.endswith("clinical_data") and name in CLINICAL_FILES:
            clinical[CLINICAL_FILES[name]] = rec
        elif subfolder == "cBioPortal_files" or (len(e["path"]) > 3 and e["path"][2] == "cBioPortal_files"):
            if len(e["path"]) > 3 and e["path"][3] == "case_lists":
                case_lists.append(rec)
            elif name in GENOMIC_FILES:
                genomic[GENOMIC_FILES[name]] = rec
            elif name.endswith("data_cna_hg19.seg"):
                genomic["cna_seg"] = rec
            elif name in CBIO_CLINICAL_FILES:
                cbio_clinical[CBIO_CLINICAL_FILES[name]] = rec
            elif name.startswith("data_timeline_"):
                key = name[len("data_timeline_") : -len(".txt")]
                timelines[key] = rec
            elif name.startswith("data_gene_panel_"):
                gene_panels.append(dict(rec, panel=name[len("data_gene_panel_") : -len(".txt")]))
            elif name.startswith("meta_") or name.endswith("meta_cna_hg19.seg"):
                meta_files.append(rec)
            else:
                other.append(dict(rec, folder=subfolder))
        elif subfolder == "Documentation":
            low = name.lower()
            if "data guide" in low or "data_guide" in low:
                documentation["data_guide"] = rec
            elif "synopsis" in low:
                documentation["variable_synopsis"] = rec
            elif "release" in low and "note" in low:
                documentation["release_notes"] = rec
            elif "crosswalk" in low:
                documentation["cbioportal_crosswalk"] = rec
            else:
                other.append(dict(rec, folder=subfolder))
        else:
            other.append(dict(rec, folder=subfolder))

    facts = COHORT_FACTS[cohort]
    return {
        "cohort": cohort,
        "version": version,
        "label": f"{cohort} {version}",
        "full_name": facts["full_name"],
        "schema_generation": SCHEMA_GENERATION[cohort],
        "cohort_folder": cohort_folder["id"],
        "release_folder": version_folder["id"],
        "release_folder_created_on": release_meta.get("createdOn"),
        "release_folder_modified_on": release_meta.get("modifiedOn"),
        "n_patients": facts["n_patients"],
        "institutions": facts["institutions"],
        "cbioportal_study_id": facts["cbioportal_study_id"],
        "sources": {
            "aacr_page": facts["aacr_page"],
            "data_guide_url": facts["data_guide_url"],
            "variable_synopsis_url": facts["variable_synopsis_url"],
        },
        "documentation": documentation,
        "files": {
            "clinical": clinical,
            "genomic": genomic,
            "cbioportal_clinical": cbio_clinical,
            "cbioportal_timeline": timelines,
            "gene_panels": sorted(gene_panels, key=lambda r: r["panel"]),
            "case_lists": sorted(case_lists, key=lambda r: r["filename"]),
            "meta": sorted(meta_files, key=lambda r: r["filename"]),
            "other": other,
        },
    }


def main(out_path: Path) -> None:
    entities = _walk(DATA_RELEASES_FOLDER)
    cohort_folders = [e for e in entities if e["is_folder"] and e["path"] == []]
    releases = []
    for cf in sorted(cohort_folders, key=lambda e: e["name"]):
        cohort = cf["name"]
        if cohort not in COHORT_FACTS:
            print(f"skipping unknown cohort folder {cohort!r} ({cf['id']})", file=sys.stderr)
            continue
        for vf in [e for e in entities if e["is_folder"] and e["path"] == [cohort]]:
            if "public" not in vf["name"]:
                continue
            releases.append(build_release(cohort, cf, vf, entities))

    order = ["NSCLC", "CRC", "BrCa", "PANC", "Prostate", "BLADDER"]
    releases.sort(key=lambda r: (order.index(r["cohort"]), r["version"]))

    payload = {
        "generated_on": dt.date.today().isoformat(),
        "generator": "data-raw/build_releases.py",
        "source": {
            "synapse_data_releases_folder": DATA_RELEASES_FOLDER,
            "synapse_bpc_project": BPC_PROJECT,
            "access_requirement_id": ACCESS_REQUIREMENT_ID,
            "rest_endpoint": REPO_ENDPOINT,
        },
        "releases": releases,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} with {len(releases)} releases")
    for r in releases:
        print(
            f"  {r['label']:22s} release_folder={r['release_folder']:12s} "
            f"clinical={len(r['files']['clinical'])} genomic={len(r['files']['genomic'])} "
            f"timelines={len(r['files']['cbioportal_timeline'])} panels={len(r['files']['gene_panels'])}"
        )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(root / "src" / "geniekit" / "resources" / "releases.json")
