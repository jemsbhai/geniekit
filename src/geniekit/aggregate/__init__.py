"""Aggregate-only analytic layer with disclosure control.

The functions here take patient-level GENIE BPC data (which stays with the
users who hold access) and return only aggregate tables: counts,
percentages, confidence intervals, Kaplan-Meier estimates, risk differences
and p-values, each passed through :class:`DisclosureControl` so that cells
below the minimum size, and every statistic derived from them, are
suppressed. The results can be shared with users who may not see
patient-level records.

Quick start::

    from geniekit import io, aggregate

    crc = io.load_release("data/genie_bpc/CRC_v2.0-public")
    report = aggregate.liver_metastasis_report(crc, control=aggregate.DisclosureControl(min_count=5))
    report.summary()                                   # the tables and how many cells were suppressed
    report["early_failure_by_liver_mets"].frame        # early treatment failure at 6 and 9 months
    report.to_dir("outputs/crc_liver")                 # one CSV per table plus report.json

Building blocks for other questions: :func:`analytic_table` (the
patient-level frame the summaries use), :func:`frequency_table`,
:func:`numeric_summary`, :func:`proportion_table`, :func:`flag_frequency_table`,
:func:`km_table`, :func:`early_failure_table`, and the genomic builders
:func:`gene_flags` and :func:`msi_mmr_status`.
"""

from .cohort import DAYS_PER_MONTH, analytic_table, first_line_failure, metastasis_sites
from .crc import CRC_COMBINED, CRC_GENES, CRC_VARIANTS, liver_metastasis_report
from .disclosure import DisclosureControl, SafeTable, SuppressionRecord
from .genomics import NONSYNONYMOUS, gene_flags, msi_mmr_status, sample_patients
from .stats import KaplanMeier, chi_square, fisher_exact, kaplan_meier, risk_difference, wilson_interval
from .summaries import (
    AggregateReport,
    early_failure_table,
    flag_frequency_table,
    frequency_table,
    km_table,
    numeric_summary,
    proportion_table,
)

__all__ = [
    "CRC_COMBINED",
    "CRC_GENES",
    "CRC_VARIANTS",
    "DAYS_PER_MONTH",
    "NONSYNONYMOUS",
    "AggregateReport",
    "DisclosureControl",
    "KaplanMeier",
    "SafeTable",
    "SuppressionRecord",
    "analytic_table",
    "chi_square",
    "early_failure_table",
    "first_line_failure",
    "fisher_exact",
    "flag_frequency_table",
    "frequency_table",
    "gene_flags",
    "kaplan_meier",
    "km_table",
    "liver_metastasis_report",
    "metastasis_sites",
    "msi_mmr_status",
    "numeric_summary",
    "proportion_table",
    "risk_difference",
    "sample_patients",
    "wilson_interval",
]
