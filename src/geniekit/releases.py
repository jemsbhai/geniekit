"""Registry of the public AACR Project GENIE BPC releases.

The registry (``resources/releases.json``) records, for each of the six public
BPC cohorts, the Synapse identifier of every clinical, genomic, cBioPortal and
documentation file, together with cohort-level facts (patient count,
contributing institutions, cBioPortal study id). It is generated from the
public Synapse entity tree by ``data-raw/build_releases.py``; nothing in it
requires an approved access request, but downloading the files does.

Logical table names used throughout geniekit, with the file they refer to and
the name the R package ``genieBPC`` uses for the same table:

==================  ===================================  ===================
geniekit            file in the release                  genieBPC
==================  ===================================  ===================
``patient``         patient_level_dataset.csv            ``pt_char``
``cancer_index``    cancer_level_dataset_index.csv       ``ca_dx_index``
``cancer_non_index`` cancer_level_dataset_non_index.csv  ``ca_dx_non_index``
``regimen``         regimen_cancer_level_dataset.csv     ``ca_drugs``
``imaging``         imaging_level_dataset.csv            ``prissmm_imaging``
``pathology``       pathology_report_level_dataset.csv   ``prissmm_pathology``
``med_onc``         med_onc_note_level_dataset.csv       ``prissmm_md``
``tumor_marker``    tm_level_dataset.csv                 ``tumor_marker``
``radiation``       ca_radtx_dataset.csv                 ``ca_radtx``
``cpt``             cancer_panel_test_level_dataset.csv  ``cpt``
``mutations``       data_mutations_extended.txt          ``mutations_extended``
``fusions``         data_fusions.txt                     ``fusions``
``sv``              data_sv.txt                          ``sv``
``cna``             data_CNA.txt                         ``cna``
==================  ===================================  ===================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources as importlib_resources
from typing import Any

import pandas as pd

__all__ = [
    "COHORTS",
    "FILE_GROUPS",
    "GENIEBPC_TABLE_NAMES",
    "FileRef",
    "Release",
    "get_release",
    "list_releases",
    "load_registry",
    "normalize_cohort",
    "registry_generated_on",
    "to_frame",
]

#: Canonical cohort labels, in the order GENIE released them.
COHORTS: tuple[str, ...] = ("NSCLC", "CRC", "BrCa", "PANC", "Prostate", "BLADDER")

#: Groups of files in a release, as stored in the registry.
FILE_GROUPS: tuple[str, ...] = (
    "clinical",
    "genomic",
    "cbioportal_clinical",
    "cbioportal_timeline",
    "gene_panels",
    "case_lists",
    "meta",
    "documentation",
)

#: geniekit logical table name -> name used by the R package ``genieBPC``.
GENIEBPC_TABLE_NAMES: dict[str, str] = {
    "patient": "pt_char",
    "cancer_index": "ca_dx_index",
    "cancer_non_index": "ca_dx_non_index",
    "regimen": "ca_drugs",
    "imaging": "prissmm_imaging",
    "pathology": "prissmm_pathology",
    "med_onc": "prissmm_md",
    "tumor_marker": "tumor_marker",
    "radiation": "ca_radtx",
    "cpt": "cpt",
    "mutations": "mutations_extended",
    "fusions": "fusions",
    "sv": "sv",
    "cna": "cna",
}

_COHORT_ALIASES: dict[str, str] = {
    "nsclc": "NSCLC",
    "lung": "NSCLC",
    "non-small cell lung cancer": "NSCLC",
    "non small cell lung cancer": "NSCLC",
    "nonsmall cell lung cancer": "NSCLC",
    "crc": "CRC",
    "colorectal": "CRC",
    "colorectal cancer": "CRC",
    "brca": "BrCa",
    "breast": "BrCa",
    "breast cancer": "BrCa",
    "panc": "PANC",
    "pancreas": "PANC",
    "pancreatic": "PANC",
    "pancreatic cancer": "PANC",
    "prostate": "Prostate",
    "prostate cancer": "Prostate",
    "bladder": "BLADDER",
    "bladder cancer": "BLADDER",
}


def normalize_cohort(name: str) -> str:
    """Return the canonical cohort label for ``name``.

    Accepts the canonical labels in any case plus common long forms
    (``"breast cancer"``, ``"pancreas"``, ...), mirroring the aliases accepted
    by ``genieBPC::pull_data_synapse``.

    >>> normalize_cohort("brca")
    'BrCa'
    >>> normalize_cohort("Non-Small Cell Lung Cancer")
    'NSCLC'
    """
    key = " ".join(str(name).strip().lower().split())
    try:
        return _COHORT_ALIASES[key]
    except KeyError:
        raise ValueError(f"Unknown cohort {name!r}; expected one of {', '.join(COHORTS)}") from None


@dataclass(frozen=True)
class FileRef:
    """A single file inside a release."""

    synapse_id: str
    filename: str
    group: str
    name: str

    @property
    def synapse_url(self) -> str:
        return f"https://www.synapse.org/Synapse:{self.synapse_id}"


@dataclass(frozen=True)
class Release:
    """One public BPC release (one cohort at one version)."""

    cohort: str
    version: str
    label: str
    full_name: str
    schema_generation: int
    cohort_folder: str
    release_folder: str
    n_patients: int
    institutions: tuple[str, ...]
    cbioportal_study_id: str
    release_folder_created_on: str | None
    sources: dict[str, str]
    files: tuple[FileRef, ...]

    # -- lookups -----------------------------------------------------------
    def files_in(self, group: str) -> dict[str, FileRef]:
        """Files of one group keyed by logical name (or panel id / filename)."""
        if group not in FILE_GROUPS:
            raise ValueError(f"Unknown file group {group!r}; expected one of {', '.join(FILE_GROUPS)}")
        return {f.name: f for f in self.files if f.group == group}

    def file(self, name: str, group: str | None = None) -> FileRef:
        """Return the file with logical name ``name``.

        When ``group`` is omitted the clinical, genomic and documentation
        groups are searched in that order.
        """
        groups = (group,) if group else ("clinical", "genomic", "documentation", "cbioportal_clinical")
        for g in groups:
            hit = self.files_in(g).get(name)
            if hit is not None:
                return hit
        raise KeyError(f"{self.label} has no file named {name!r} in group(s) {', '.join(groups)}")

    def has(self, name: str, group: str | None = None) -> bool:
        try:
            self.file(name, group)
        except KeyError:
            return False
        return True

    @property
    def clinical(self) -> dict[str, FileRef]:
        return self.files_in("clinical")

    @property
    def genomic(self) -> dict[str, FileRef]:
        return self.files_in("genomic")

    @property
    def documentation(self) -> dict[str, FileRef]:
        return self.files_in("documentation")

    @property
    def synapse_url(self) -> str:
        return f"https://www.synapse.org/Synapse:{self.release_folder}"

    @property
    def cbioportal_url(self) -> str:
        return f"https://genie.cbioportal.org/study/summary?id={self.cbioportal_study_id}"

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "files"}
        d["institutions"] = list(self.institutions)
        d["files"] = [f.__dict__ for f in self.files]
        return d


def _flatten_files(raw: dict[str, Any]) -> tuple[FileRef, ...]:
    out: list[FileRef] = []
    for group in ("clinical", "genomic", "cbioportal_clinical", "cbioportal_timeline"):
        for name, rec in raw["files"].get(group, {}).items():
            out.append(FileRef(rec["synapse_id"], rec["filename"], group, name))
    for rec in raw["files"].get("gene_panels", []):
        out.append(FileRef(rec["synapse_id"], rec["filename"], "gene_panels", rec["panel"]))
    for group in ("case_lists", "meta"):
        for rec in raw["files"].get(group, []):
            out.append(FileRef(rec["synapse_id"], rec["filename"], group, rec["filename"]))
    for name, rec in raw.get("documentation", {}).items():
        out.append(FileRef(rec["synapse_id"], rec["filename"], "documentation", name))
    return tuple(out)


def _release_from_raw(raw: dict[str, Any]) -> Release:
    return Release(
        cohort=raw["cohort"],
        version=raw["version"],
        label=raw["label"],
        full_name=raw["full_name"],
        schema_generation=int(raw["schema_generation"]),
        cohort_folder=raw["cohort_folder"],
        release_folder=raw["release_folder"],
        n_patients=int(raw["n_patients"]),
        institutions=tuple(raw["institutions"]),
        cbioportal_study_id=raw["cbioportal_study_id"],
        release_folder_created_on=raw.get("release_folder_created_on"),
        sources=dict(raw.get("sources", {})),
        files=_flatten_files(raw),
    )


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    """Return the raw registry as parsed from ``resources/releases.json``."""
    text = (
        importlib_resources.files("geniekit.resources").joinpath("releases.json").read_text(encoding="utf-8")
    )
    return json.loads(text)


@lru_cache(maxsize=1)
def _all_releases() -> tuple[Release, ...]:
    return tuple(_release_from_raw(r) for r in load_registry()["releases"])


def registry_generated_on() -> str:
    """ISO date on which the packaged registry was generated."""
    return str(load_registry()["generated_on"])


def _version_key(version: str) -> tuple[int, ...]:
    core = version.lstrip("v").split("-")[0]
    return tuple(int(p) for p in core.split("."))


def list_releases(cohort: str | None = None, version: str | None = None) -> list[Release]:
    """All known public releases, optionally filtered by cohort and/or version."""
    out = list(_all_releases())
    if cohort is not None:
        c = normalize_cohort(cohort)
        out = [r for r in out if r.cohort == c]
    if version is not None:
        out = [r for r in out if r.version == version]
    return out


def get_release(cohort: str, version: str | None = None) -> Release:
    """Return one release.

    With ``version=None`` the most recent public release of the cohort is
    returned (there is currently exactly one per cohort).
    """
    matches = list_releases(cohort, version)
    if not matches:
        wanted = f"{normalize_cohort(cohort)} {version}" if version else normalize_cohort(cohort)
        known = ", ".join(r.label for r in _all_releases())
        raise KeyError(f"No public release matching {wanted!r}; known releases: {known}")
    return max(matches, key=lambda r: _version_key(r.version))


def to_frame() -> pd.DataFrame:
    """One row per file across all releases."""
    rows = []
    for r in _all_releases():
        for f in r.files:
            rows.append(
                {
                    "cohort": r.cohort,
                    "version": r.version,
                    "schema_generation": r.schema_generation,
                    "group": f.group,
                    "name": f.name,
                    "filename": f.filename,
                    "synapse_id": f.synapse_id,
                }
            )
    return pd.DataFrame(
        rows, columns=["cohort", "version", "schema_generation", "group", "name", "filename", "synapse_id"]
    )
