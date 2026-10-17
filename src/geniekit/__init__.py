"""geniekit: tools for the AACR Project GENIE BPC clinicogenomic cohorts.

Three capabilities, one package:

* ``geniekit.harmonize``: normalise the six public BPC cohorts (NSCLC, CRC,
  BrCa, PANC, Prostate, BLADDER) to one schema for cross-cohort analysis.
* ``geniekit.lot``: derive lines of therapy from the regimen data relative to a
  user-chosen index event.
* ``geniekit.aggregate``: an aggregate-only analytic layer with disclosure
  control for users who may not see patient-level records.

Supporting modules: ``geniekit.releases`` (registry of public releases and
Synapse ids), ``geniekit.dictionary`` (compiled variable synopses),
``geniekit.io`` (download and load), ``geniekit.synthetic`` (synthetic cohorts
for tests and demos).

The package ships metadata only. Patient-level GENIE data must be obtained
from Synapse under the GENIE access requirement.
"""

from ._version import __version__
from .releases import COHORTS, get_release, list_releases, normalize_cohort

__all__ = [
    "COHORTS",
    "__version__",
    "get_release",
    "list_releases",
    "normalize_cohort",
]
