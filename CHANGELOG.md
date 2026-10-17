# Changelog

All notable changes to geniekit are recorded here. The format follows
Keep a Changelog, and the project follows semantic versioning.

## [Unreleased]

### Added
- `geniekit.harmonize`: crosswalk (`resources/crosswalk.yaml`, 25 rules with
  their basis in the data guides), `harmonize()` with a provenance report,
  `stack()` with per-column cohort availability, derived `race_ethnicity`,
  `pfs_time_origin` and NAACCR summary-stage code/label, and unification of
  the two structural-variant formats and the copy-number matrix.
- CLI commands `crosswalk` and `harmonize`.

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
