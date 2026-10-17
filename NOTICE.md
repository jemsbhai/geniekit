# Notice

geniekit is an independent, community-developed tool. It is not produced,
endorsed or maintained by the American Association for Cancer Research (AACR),
AACR Project GENIE, Sage Bionetworks or any GENIE-participating institution.

## Data

geniekit ships no patient-level data. All GENIE and GENIE BPC data are
obtained by the user from Synapse (https://www.synapse.org) under the GENIE
access requirement and the terms of use that accompany each release. Users are
responsible for complying with those terms, including restrictions on
redistribution and on attempts to re-identify individuals.

## Packaged metadata and its sources

The files in `src/geniekit/resources/` are derived from public materials:

- `releases.json` is generated from the public entity tree of the GENIE BPC
  "Data Releases" folder on Synapse (syn27056700) through the Synapse REST API,
  plus cohort facts published on the AACR Project GENIE website
  (https://aacrprojectgenie.org/data/).
- `dictionary.csv.gz` is compiled from the Variable Synopsis workbooks that
  AACR publishes with each public BPC release. The workbooks are linked from
  the cohort pages on https://aacrprojectgenie.org/data/ and are the
  authoritative description of each release; geniekit reproduces variable
  names, field labels, data types and permitted values so that software can
  reason about them. Consult the Analytic Data Guide of each release for the
  full definitions.

## Test fixture

`tests/fixtures/nsclc_sim/` is derived from the `nsclc_test_data` object of
the R package genieBPC (https://github.com/GENIE-BPC/genieBPC, MIT licence,
copyright 2022 genieBPC authors), which its authors describe as simulated fake
data for examples and tests. It contains no real patient records. See
`tests/fixtures/README.md`.

AACR Project GENIE and GENIE BPC are trademarks of their respective owners.

## Citing the data

Users of GENIE data are asked by AACR to cite:

AACR Project GENIE Consortium. AACR Project GENIE: Powering Precision Medicine
through an International Consortium. Cancer Discovery 2017;7(8):818-831.
https://doi.org/10.1158/2159-8290.CD-17-0151

and, for BPC cohorts, the cohort-specific publication listed on the cohort's
page at https://aacrprojectgenie.org/data/.
