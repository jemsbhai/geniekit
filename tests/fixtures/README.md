# Test fixtures

## `nsclc_sim/`

A simulated (fake) NSCLC cohort with the column layout of a GENIE BPC release,
stored under the file names Synapse uses, gzip-compressed:

- `clinical_data/` : patient, index and non-index cancer, regimen, imaging,
  pathology, medical-oncologist and cancer-panel-test tables (200 patients);
- `cBioPortal_files/` : mutations (MAF-like, with a `#version` header line),
  fusions and a copy-number matrix.

Source: the `nsclc_test_data` object shipped with the R package
[genieBPC](https://github.com/GENIE-BPC/genieBPC) version 2.2.0
(MIT licence, copyright 2022 genieBPC authors), which its documentation
describes as "simulated fake GENIE BPC data for function examples and
tests". The tables were exported from R unchanged except that the genomic
tables were reduced to their cBioPortal columns (the R object carries the
cancer-panel-test columns pre-joined). Record and sample identifiers use the
`GENIE-TEST-` prefix; no row corresponds to a real patient.

The fixture is used to test loading, harmonisation and line-of-therapy code
paths. Being a simulation, its value distributions carry no clinical meaning
and must not be used to draw conclusions about the real cohort.
