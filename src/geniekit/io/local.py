"""Load a BPC release from files on disk.

Accepted layouts (all searched recursively from the directory you pass):

* a release folder as downloaded from Synapse, e.g. ``clinical_data/*.csv``
  plus ``cBioPortal_files/*.txt``;
* a folder written by :func:`geniekit.io.download_release`
  (``<cohort>_<version>/clinical/...``, ``genomic/...`` with a manifest);
* a folder written by :meth:`CohortData.to_dir` (``patient.csv`` etc.);
* any flat directory containing the standard file names, optionally
  gzip-compressed (``patient_level_dataset.csv.gz``).

Files are read the way the official R package reads them: CSV with ``""`` and
``"NA"`` as missing, tab-delimited for ``.txt``/``.tsv``/``.seg``. Leading
lines that start with ``#`` (cBioPortal metadata headers) are skipped.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .. import dictionary
from ..releases import COHORTS, list_releases, normalize_cohort
from .cohort import CLINICAL_TABLES, GENOMIC_TABLES, CohortData

__all__ = ["FILENAME_TO_TABLE", "find_release_files", "load_release", "read_table"]

_STATIC_FILENAMES: dict[str, str] = {
    "data_mutations_extended.txt": "mutations",
    "data_fusions.txt": "fusions",
    "data_sv.txt": "sv",
    "data_CNA.txt": "cna",
    "data_cna_hg19.seg": "cna_seg",
    "data_gene_matrix.txt": "gene_matrix",
    "genomic_information.txt": "genomic_information",
    "tmb.tsv": "tmb",
    "data_clinical_patient.txt": "clinical_patient",
    "data_clinical_sample.txt": "clinical_sample",
    "data_clinical_supp_survival.txt": "supp_survival",
    "data_clinical_supp_survival_treatment.txt": "supp_survival_treatment",
}


def _build_filename_map() -> dict[str, str]:
    mapping = dict(_STATIC_FILENAMES)
    for rel in list_releases():
        for name, ref in rel.clinical.items():
            mapping[ref.filename] = name
        for name, ref in rel.genomic.items():
            mapping.setdefault(ref.filename, name)
        for name, ref in rel.files_in("cbioportal_timeline").items():
            mapping.setdefault(ref.filename, f"timeline_{name}")
    for name in (*CLINICAL_TABLES, *GENOMIC_TABLES):
        mapping[f"{name}.csv"] = name
    return mapping


#: Release file name (without ``.gz``) -> logical table name.
FILENAME_TO_TABLE: dict[str, str] = _build_filename_map()

_NA_VALUES = ["", "NA"]


def _logical_name(path: Path) -> str | None:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name in FILENAME_TO_TABLE:
        return FILENAME_TO_TABLE[name]
    # cohort-prefixed seg files, e.g. genie_crc_data_cna_hg19.seg
    if name.endswith("data_cna_hg19.seg"):
        return "cna_seg"
    return None


def find_release_files(root: str | Path, tables: Iterable[str] | None = None) -> dict[str, Path]:
    """Map logical table names to files found under ``root``.

    Raises ``FileNotFoundError`` when nothing recognisable is found and
    ``ValueError`` when one table matches several files (for example two
    releases under one directory): pass the release folder itself instead.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    wanted = set(tables) if tables else None
    found: dict[str, list[Path]] = {}
    candidates = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for path in candidates:
        if any(part.startswith(".") for part in path.relative_to(root).parts[:-1]):
            continue
        name = _logical_name(path)
        if name is None or (wanted is not None and name not in wanted):
            continue
        found.setdefault(name, []).append(path)
    if not found:
        raise FileNotFoundError(
            f"No GENIE BPC release files found under {root}. Expected names such as "
            "patient_level_dataset.csv, cancer_level_dataset_index.csv, regimen_cancer_level_dataset.csv."
        )
    out: dict[str, Path] = {}
    for name, paths in found.items():
        if len(paths) > 1:
            listing = "\n  ".join(str(p) for p in paths)
            raise ValueError(
                f"Table {name!r} matches {len(paths)} files under {root}; "
                f"point to one release folder:\n  {listing}"
            )
        out[name] = paths[0]
    return out


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def _skip_comment_header(path: Path) -> tuple[str, int]:
    """Return (text, n_skipped) with leading '#' lines removed."""
    with _open_text(path) as fh:
        lines = fh.readlines()
    n = 0
    while n < len(lines) and lines[n].startswith("#"):
        n += 1
    return "".join(lines[n:]), n


def read_table(path: str | Path, name: str | None = None) -> pd.DataFrame:
    """Read one release file into a frame.

    ``name`` (the logical table name) is inferred from the file name when
    omitted; it only affects the choice between comma and tab separation for
    files with unusual extensions.
    """
    path = Path(path)
    base = path.name[:-3] if path.name.endswith(".gz") else path.name
    suffix = Path(base).suffix.lower()
    sep = "," if suffix == ".csv" else "\t"
    if suffix == ".csv":
        with _open_text(path) as fh:
            return pd.read_csv(fh, sep=sep, na_values=_NA_VALUES, keep_default_na=False, low_memory=False)
    text, _ = _skip_comment_header(path)
    return pd.read_csv(
        io.StringIO(text), sep=sep, na_values=_NA_VALUES, keep_default_na=False, low_memory=False
    )


def _infer_cohort_version(tables: dict[str, pd.DataFrame], root: Path) -> tuple[str | None, str | None]:
    cohort = version = None
    for name in ("patient", "cancer_index", "regimen", "cpt", "imaging"):
        frame = tables.get(name)
        if frame is None:
            continue
        if cohort is None and "cohort" in frame.columns and frame["cohort"].notna().any():
            cohort = str(frame["cohort"].dropna().iloc[0])
        if version is None and "release_version" in frame.columns and frame["release_version"].notna().any():
            version = str(frame["release_version"].dropna().iloc[0])
        if cohort and version:
            break
    if cohort is None or version is None:
        # fall back to a "<cohort>_<version>" folder name
        parts = root.name.split("_", 1)
        if len(parts) == 2:
            try:
                normalize_cohort(parts[0])
            except ValueError:
                pass
            else:
                cohort = cohort or parts[0]
                version = version or parts[1]
    return cohort, version


def _validate_columns(data: CohortData) -> dict[str, dict[str, list[str]]]:
    """Compare clinical columns with the packaged dictionary for this cohort."""
    if data.cohort not in COHORTS:
        return {}
    dd = dictionary.load()
    report: dict[str, dict[str, list[str]]] = {}
    for name, frame in data.clinical.items():
        dict_table = "cancer" if name in ("cancer_index", "cancer_non_index") else name
        rows = dd.variables(data.cohort, dict_table)
        declared = set(rows["variable"])
        if not declared:
            continue
        known_absent = set(rows.loc[~rows["released"], "variable"])
        actual = set(frame.columns)
        missing = sorted(declared - actual - known_absent)
        not_released = sorted((declared - actual) & known_absent)
        extra = sorted(actual - declared)
        if missing or extra or not_released:
            report[name] = {"missing": missing, "extra": extra, "not_released": not_released}
    return report


def _validation_warning(report: dict[str, dict[str, list[str]]]) -> str | None:
    """Text of the column warning, or None when the differences are expected.

    The synopsis describes the index and non-index cancer files together, so
    each of those files legitimately lacks the other's columns; only
    unexpected columns are reported for them. Variables the dictionary records
    as declared but not released (``not_released``) are not warned about.
    """
    parts = []
    for table, r in report.items():
        missing = [] if table in ("cancer_index", "cancer_non_index") else r["missing"]
        if missing or r["extra"]:
            parts.append(f"{table}: {len(missing)} missing, {len(r['extra'])} unexpected")
    return "; ".join(parts) if parts else None


def load_release(
    path: str | Path,
    cohort: str | None = None,
    version: str | None = None,
    tables: Iterable[str] | None = None,
    genomic: bool = True,
    validate: bool = True,
) -> CohortData:
    """Load every recognisable release file under ``path`` into a :class:`CohortData`.

    Parameters
    ----------
    path:
        A release folder (see module docstring for accepted layouts).
    cohort, version:
        Override or supply the cohort label and release version. When omitted
        they are read from the ``cohort`` and ``release_version`` columns of
        the clinical files, then from a ``<cohort>_<version>`` folder name.
    tables:
        Restrict loading to these logical table names.
    genomic:
        Set ``False`` to skip the (large) genomic files.
    validate:
        Compare the clinical columns with the packaged dictionary and store the
        result in ``CohortData.validation``; a warning is issued when columns
        are missing or unexpected.
    """
    root = Path(path)
    files = find_release_files(root, tables)
    if not genomic:
        files = {n: p for n, p in files.items() if n not in GENOMIC_TABLES}
    frames = {name: read_table(p, name) for name, p in files.items()}
    inferred_cohort, inferred_version = _infer_cohort_version(frames, root)
    cohort = cohort or inferred_cohort
    if cohort is None:
        raise ValueError(
            f"Could not infer the cohort from the files under {root}; pass cohort='NSCLC' (or another label)."
        )
    data = CohortData(
        cohort=cohort,
        version=version or inferred_version,
        tables=frames,
        sources={n: str(p) for n, p in files.items()},
    )
    manifest = root / "manifest.json"
    if manifest.exists():
        with contextlib.suppress(OSError, ValueError):
            data.validation["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    if validate:
        report = _validate_columns(data)
        data.validation["columns"] = report
        details = _validation_warning(report)
        if details:
            warnings.warn(
                f"{data.label}: clinical columns differ from the packaged dictionary ({details}); "
                "see CohortData.validation['columns'].",
                stacklevel=2,
            )
    return data
