# Changelog

All notable changes to geniekit are recorded here. The format follows
Keep a Changelog, and the project follows semantic versioning.

## [0.1.0] - 2026-10-17

All three challenge capabilities are in this release: harmonisation of the six
cohorts, lines of therapy from a user-chosen index event, and the
aggregate-only analytic layer, each validated on the six public releases.

### Added
- `geniekit.aggregate`: an aggregate-only analytic layer with disclosure
  control. `DisclosureControl` (minimum cell size, complementary suppression
  or omission of small categories by comparing the records lost, suppression
  of every statistic derived from a suppressed count); summary functions
  (`frequency_table`, `numeric_summary`, `proportion_table`,
  `flag_frequency_table`, `km_table`, `early_failure_table`) that return
  `SafeTable` objects with notes and the suppressions recorded; statistics
  implemented without optional dependencies and checked against hand
  computations (Wilson interval, risk difference, Fisher exact, chi-square,
  Kaplan-Meier with log-log limits); genomic builders (`gene_flags`,
  `msi_mmr_status`); `analytic_table` joining clinical variables, index
  event, lines of therapy and the first-line failure endpoint; and the
  Challenge 10 example `liver_metastasis_report` (patients by liver
  metastasis status, early treatment failure at 6 and 9 months after
  first-line therapy, APC/TP53/KRAS/NRAS/BRAF V600E/SMAD4 and MSI-H or dMMR
  frequencies, comparisons by liver metastasis and genomic subgroup). CLI
  command `aggregate` and the notebook `examples/crc_liver_metastasis.ipynb`.
- `geniekit.lot`: lines of therapy from the regimen tables, counted from an
  index event the user chooses (`diagnosis`, `advanced`, `first_progression`,
  `first_systemic`, or a custom table or callable), with explicit rules
  (`LineRules`: combination window, maintenance and resumption, restart gap,
  progression awareness, pre-index handling, backbone and excluded roles,
  investigational drugs; presets `default`, `prostate`, `progression`),
  three output tables (regimens with the reason for every assignment, lines,
  cancers) and the settings written alongside them. CLI command `lot`.
- Drug vocabulary (`resources/drug_vocabulary.csv`, `lot.load_vocabulary()`):
  all 205 drug labels released across the six cohorts mapped to canonical
  agents, classes and roles; built by `data-raw/build_drug_vocabulary.py`.
- `geniekit.harmonize`: crosswalk (`resources/crosswalk.yaml`, 25 rules with
  their basis in the data guides), `harmonize()` with a provenance report,
  `stack()` with per-column cohort availability, derived `race_ethnicity`,
  `pfs_time_origin` and NAACCR summary-stage code/label, and unification of
  the two structural-variant formats and the copy-number matrix.
- CLI commands `crosswalk` and `harmonize`.
- `DataDictionary.declared_not_released()` and the packaged record
  `resources/released_columns.json`: the 17 variables the Variable Synopses
  declare that the released files do not contain (established on the six
  releases on 2026-10-17). `load_release` reports them under
  `validation["columns"][table]["not_released"]` instead of warning.

### Changed
- `SynapseSession.download_file` retries a refused download (HTTP 403, 408,
  429 or 5xx, or an MD5 mismatch) with a fresh signed URL and exponential
  backoff; Synapse's storage back end refuses a freshly signed URL now and
  then. `download_release` continues past a file that still fails, records
  it under `"failed"` in `manifest.json`, and raises only with `strict=True`.
- The derived `race_ethnicity` now uses the rule behind the released
  generation-1 variable, recovered from the NSCLC and CRC releases where it
  reproduces every released value (1846/1846 and 1485/1485); the earlier rule
  disagreed on 75 and 44 patients. The rule is selectable through the
  crosswalk (`options.rule`: `genie` or `conservative`); see the rule's
  basis and caveat.
- Idempotence of the derivations is decided from the dictionary (whether the
  cohort releases the variable), not from the values.

### Fixed
- The crosswalk and README said `naaccr_seer_sum_stage` is free text in
  generation 1; the released files hold the numeric NAACCR code in all six
  cohorts, and only the synopses differ.
- Column validation on all six real releases: every released clinical column
  is declared in the dictionary; the index and non-index cancer files together
  hold every declared cancer variable.

## [0.0.1] - 2026-10-17

First public release: the foundation modules.

### Added
- `geniekit.releases`: registry of the six public GENIE BPC releases (NSCLC
  v2.0, CRC v2.0, BrCa v1.0, PANC v1.0, Prostate v1.0, BLADDER v1.0) with the
  Synapse id of every clinical, genomic, cBioPortal and documentation file,
  generated from the public Synapse entity tree.
- `geniekit.dictionary`: the six Variable Synopsis workbooks compiled into one
  data dictionary, with presence matrices and cross-cohort comparison of
  types, labels and permitted values.
- `geniekit` command line interface with `releases` and `dictionary`
  subcommands.
- Build scripts in `data-raw/` that regenerate the packaged metadata.
- `geniekit.io`: `CohortData` container, `load_release` for release folders on
  disk (Synapse layout, geniekit download layout, flat folders, gzip), and a
  Synapse client (`SynapseSession`, `download_release`) with token
  resolution, access-status reporting and MD5-verified downloads.
- `geniekit.synthetic`: dictionary-driven synthetic cohorts for all six
  public releases, with coherent patient histories, regimens, imaging and
  oncologist assessments, sequencing samples and genomic tables.
- CLI commands `access`, `download` and `inspect`.
- Test fixture `tests/fixtures/nsclc_sim` derived from the simulated data
  shipped with the R package genieBPC (MIT).
