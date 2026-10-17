# geniekit

Python tools for the AACR Project GENIE Biopharma Collaborative (BPC)
clinicogenomic cohorts.

- **Harmonise** the six public BPC cohorts (NSCLC, CRC, BrCa, PANC, Prostate,
  BLADDER) to one schema so that cross-cohort analyses do not have to deal
  with per-cohort variable names, types and code lists.
- **Lines of therapy**: derive first and subsequent lines from the regimen
  data relative to an index event that the user chooses (diagnosis,
  metastatic diagnosis, first progression, first systemic therapy, or any
  custom event), with explicit, configurable rules.
- **Aggregate-only analytic layer**: produce cohort-level summaries with
  disclosure control (minimum cell sizes, suppression, no row-level output)
  for users who may not access patient-level records.

geniekit ships metadata only. The data themselves come from Synapse under the
GENIE access requirement (see [Data access](#data-access)).

## Status

Version 0.0.1 is the first public release and contains the foundation: the
release registry, the data dictionary, data loading and download, and the
synthetic cohorts. The harmonisation, line-of-therapy and aggregate layers
follow in the next minor releases.

| Module | Purpose | State |
|---|---|---|
| `geniekit.releases` | Registry of the six public BPC releases with the Synapse id of every clinical, genomic, cBioPortal and documentation file | done |
| `geniekit.dictionary` | The six Variable Synopsis workbooks compiled into one data dictionary, with cross-cohort comparison helpers | done |
| `geniekit.io` | Download from Synapse (personal access token, access-status check, MD5 verification) and load release folders from disk into one `CohortData` object | done |
| `geniekit.synthetic` | Dictionary-driven synthetic cohorts: every declared column, cohort-specific naming, coherent simulated histories | done |
| `geniekit.harmonize` | Crosswalk-driven normalisation of the six cohorts | planned |
| `geniekit.lot` | Line-of-therapy derivation with user-defined index events | planned |
| `geniekit.aggregate` | Aggregate-only analyses with disclosure control | planned |

## Installation

```powershell
pip install geniekit
```

Optional extras: `geniekit[synapse]` (official Synapse client), `geniekit[survival]`
(Kaplan-Meier summaries), `geniekit[notebook]` (Jupyter and plotting),
`geniekit[dev]` (tests and linting).

From a clone:

```powershell
pip install -e ".[dev]"
pytest tests/ -v --cov=geniekit --cov-report=term
```

## Quick start

```python
from geniekit import releases, dictionary

# What is available, and where on Synapse
for r in releases.list_releases():
    print(r.label, r.n_patients, r.release_folder, sorted(r.clinical))

# How the cohorts differ before any harmonisation
dd = dictionary.load()
print(dd.compare("ca_cigarette", table="cancer").summary())
print(dd.differences(table="patient").head())
```

Loading data (after approval, see [Data access](#data-access)):

```python
from geniekit import io

# download one release with a personal access token in SYNAPSE_AUTH_TOKEN
folder = io.download_release("CRC", dest="data/genie_bpc")       # data/genie_bpc/CRC_v2.0-public
crc = io.load_release(folder)                                     # CohortData with 12 tables
crc["regimen"].head()

# or point at a folder you downloaded through the Synapse website
nsclc = io.load_release("downloads/NSCLC_2.0-public")
```

Working without the data:

```python
from geniekit import synthetic

fake = synthetic.generate("BLADDER", n_patients=300, seed=0)     # same tables and columns as the real release
fake["cancer_index"][["record_id", "stage_dx", "dist_mets_liver", "tt_os_dx_days"]].head()
```

Synthetic cohorts carry the exact column set of the corresponding release
(checked against the dictionary in the test suite) and mutually consistent
identifiers, intervals and outcomes, but their values are invented. They
exist so that code can be written and tested before access is granted; never
report numbers computed from them.

Command line:

```powershell
geniekit releases --files
geniekit dictionary lookup naaccr_seer_sum_stage --table cancer --values
geniekit dictionary diff --table cancer --kind dtype --kind value_set
geniekit dictionary presence --table patient --only-partial
geniekit access                                   # token and GENIE approval status
geniekit download CRC --dest data/genie_bpc       # after approval
geniekit inspect data/genie_bpc/CRC_v2.0-public   # table summary and column check
```

## The six public releases

| Cohort | Version | Patients | Institutions | Synapse | Schema generation |
|---|---|---|---|---|---|
| NSCLC | v2.0-public | 1846 | DFCI, MSK, UHN, VICC | syn27056697 | 1 |
| CRC | v2.0-public | 1485 | DFCI, MSK, VICC | syn30991637 | 1 |
| BrCa | v1.0-public | 1130 | DFCI, MSK, VICC | syn71825332 | 2 |
| PANC | v1.0-public | 1109 | DFCI, MSK, UHN, VICC | syn72666404 | 2 |
| Prostate | v1.0-public | 1116 | DFCI, MSK, UHN, VICC | syn73719283 | 2 |
| BLADDER | v1.0-public | 716 | DFCI, MSK, UHN, VICC | syn75201334 | 2 |

"Schema generation" is a geniekit label, not a GENIE term. The two 2022
releases (NSCLC, CRC) and the four 2025 to 2026 releases share most of their
variables, but the later releases renamed or recoded a measurable subset. Of
the 1105 distinct (table, variable) pairs declared across the six Variable
Synopses, 505 are declared in all six cohorts. Examples of what differs:

- smoking history is `ca_lung_cigarette` in generation 1 and `ca_cigarette` in generation 2;
- distant metastasis sites are `dmets_*` / `dx_to_dmets_*` (9 sites) in generation 1 and `dist_mets_*` / `dx_to_dist_mets_*` (17 sites) in generation 2;
- `naaccr_seer_sum_stage` is free text in generation 1 and a coded numeric field in generation 2;
- `cpt_seq_date` is character in generation 1 and numeric in generation 2;
- `ca_grade`, `pfs_cohort`, `ca_heme_type`, `ca_clin_t_stage` and others have different permitted value sets;
- tumor-marker tables exist only for CRC, BrCa, PANC and Prostate; radiation-therapy tables only for PANC, Prostate and BLADDER;
- structural variants are `data_fusions.txt` in generation 1 and `data_sv.txt` in generation 2;
- UHN contributes to NSCLC, PANC, Prostate and BLADDER but not to CRC or BrCa.

`geniekit dictionary diff` lists every such difference.

## Data access

1. Create a Synapse account and request access to the GENIE BPC data: open
   https://www.synapse.org/Synapse:syn27056700 and click "Request Access".
   One access requirement (id 9606645) covers both the BPC cohorts and the
   main GENIE registry.
2. Complete the Sage Bionetworks Access and Compliance Team (ACT) form linked
   from that dialog, using your Synapse username. Both steps are required.
3. After approval, create a personal access token (Synapse profile, Settings,
   Personal Access Tokens) with "view" and "download" scope and expose it as
   `SYNAPSE_AUTH_TOKEN` (see `.env.example`). Never commit the token.
   `geniekit access` reports whether the token works and whether the access
   request is approved, pending or missing.

The per-cohort Analytic Data Guides and Variable Synopses are public and are
linked from https://aacrprojectgenie.org/data/.

## Relationship to the R package genieBPC

The R package [genieBPC](https://github.com/GENIE-BPC/genieBPC) by the MSK
Biostatistics team downloads BPC releases and builds analytic cohorts one
cohort at a time. geniekit uses the same logical table names where they exist
(`releases.GENIEBPC_TABLE_NAMES` maps between the two) so that workflows can
move between the languages, and adds the cross-cohort harmonisation, line-of-
therapy and aggregate-only layers.

## Development

```powershell
git clone https://github.com/jemsbhai/geniekit.git
cd geniekit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest tests/ -v --cov=geniekit --cov-report=term
ruff check src tests data-raw
python scripts/check_vocabulary.py
```

Regenerating the packaged metadata (requires network access, no Synapse login):

```powershell
python data-raw/build_releases.py
python data-raw/build_dictionary.py
```

## License

MIT. See `LICENSE`. See `NOTICE.md` for the provenance of the packaged
metadata and for how to cite the data.
