"""Normalise the six public GENIE BPC cohorts to one schema.

The crosswalk (``geniekit/resources/crosswalk.yaml``) records every rename,
code-list collapse, spelling fix, type coercion and derived variable, with the
basis for each in the Analytic Data Guides. :func:`harmonize` applies it to one
cohort and returns the tables together with a :class:`HarmonizationReport`;
:func:`stack` concatenates harmonised cohorts and tells you which cohorts
contributed each column.

>>> from geniekit import synthetic, harmonize
>>> crc = harmonize.harmonize(synthetic.generate("CRC", 20, seed=0))
>>> crc.report.renames()["ca_lung_cigarette"]
'ca_cigarette'
"""

from .crosswalk import Crosswalk, Rule, load_crosswalk
from .engine import HarmonizedCohort, StackedData, harmonize, stack
from .genomic import cna_to_long, structural_variants
from .report import HarmonizationReport, Record

__all__ = [
    "Crosswalk",
    "HarmonizationReport",
    "HarmonizedCohort",
    "Record",
    "Rule",
    "StackedData",
    "cna_to_long",
    "harmonize",
    "load_crosswalk",
    "stack",
    "structural_variants",
]
