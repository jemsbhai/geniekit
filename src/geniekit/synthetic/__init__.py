"""Synthetic GENIE BPC cohorts for tests, examples and demonstrations.

>>> from geniekit import synthetic
>>> crc = synthetic.generate("CRC", n_patients=50, seed=1)
>>> crc.synthetic, sorted(crc.names)[:3]
(True, ['cancer_index', 'cancer_non_index', 'cna'])

The data are invented. See :mod:`geniekit.synthetic.generate` for what is and
is not coherent, and never present summaries of synthetic cohorts as findings.
"""

from .generate import generate, generate_all
from .spec import COHORT_SPECS, CohortSpec

__all__ = ["COHORT_SPECS", "CohortSpec", "generate", "generate_all"]
