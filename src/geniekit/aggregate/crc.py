"""The GENIE BPC CRC liver-metastasis example as an aggregate-only report.

:func:`liver_metastasis_report` reproduces the analysis the challenge
describes for the CRC cohort, with every output under disclosure control:

a. patients stratified by liver metastasis (ever, and relative to the index
   event), with demographics, stage, institution and treatment;
b. early treatment failure at 6 and 9 months after the start of first-line
   systemic therapy for advanced disease, by liver metastasis status;
c. frequencies of APC, TP53, KRAS, NRAS (and KRAS or NRAS), BRAF V600E,
   SMAD4 alterations and MSI-H or dMMR status, overall and by liver
   metastasis status;
d. comparisons of early failure between liver and non-liver metastasis and
   between genomic subgroups (RAS, BRAF V600E, MSI-H or dMMR).

The same function runs on any cohort that releases a liver metastasis flag
(all six do); the gene list is the CRC default and can be changed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from geniekit.io.cohort import CohortData
from geniekit.lot import LineRules, LinesOfTherapy, derive
from geniekit.lot.index import IndexSpec

from .cohort import analytic_table
from .disclosure import DisclosureControl
from .summaries import (
    AggregateReport,
    early_failure_table,
    flag_frequency_table,
    frequency_table,
    km_table,
    numeric_summary,
)

__all__ = ["CRC_COMBINED", "CRC_GENES", "CRC_VARIANTS", "liver_metastasis_report"]

CRC_GENES: tuple[str, ...] = ("APC", "TP53", "KRAS", "NRAS", "BRAF", "SMAD4")
CRC_VARIANTS: dict[str, str] = {"BRAF_V600E": "BRAF p.V600E"}
CRC_COMBINED: dict[str, tuple[str, ...]] = {"RAS_mut": ("KRAS", "NRAS")}
LIVER_LABELS = {True: "Liver metastasis", False: "No liver metastasis"}


def _lines_bucket(n: int) -> str:
    return "3 or more" if n >= 3 else str(n)


def liver_metastasis_report(
    data: CohortData,
    control: DisclosureControl | None = None,
    index: IndexSpec = "advanced",
    rules: LineRules | str | None = None,
    lines: LinesOfTherapy | None = None,
    months: Iterable[float] = (6, 9),
    genes: Iterable[str] = CRC_GENES,
    variants: Mapping[str, str] | None = None,
    combined: Mapping[str, Iterable[str]] | None = None,
    survival_months: Iterable[float] = (12, 24),
) -> AggregateReport:
    """Build the liver-metastasis report; returns only disclosure-controlled tables.

    ``index`` and ``rules`` are passed to :func:`geniekit.lot.derive` (or give
    a ready ``lines`` result); ``months`` are the early-failure horizons.
    """
    ctl = control if control is not None else DisclosureControl()
    variants = dict(CRC_VARIANTS if variants is None else variants)
    combined = dict(CRC_COMBINED if combined is None else combined)
    lines = lines if lines is not None else derive(data, index=index, rules=rules)
    genomic = "mutations" in data and "cpt" in data
    table = analytic_table(
        data,
        lines=lines,
        genes=genes if genomic else (),
        variants=variants if genomic else None,
        combined=combined if genomic else None,
        msi="pathology" in data,
    )
    months = list(months)
    report = AggregateReport(
        name=f"{data.label} liver metastasis report",
        settings={
            "cohort": data.label,
            "index_event": lines.index_event,
            "rules": lines.rules.describe(),
            "disclosure": ctl.describe(),
            "months": months,
            "genes": list(genes),
            "variants": variants,
            "combined": {k: list(v) for k, v in combined.items()},
        },
        notes=[
            "All tables are aggregate: counts below the minimum cell size and statistics derived from them "
            "are "
            "suppressed; complementary suppression protects suppressed counts against table margins.",
            "Liver metastasis: the released flag (metastasis to the liver at diagnosis for stage IV cancers, "
            "or post-diagnosis), populated for the patient's first BPC Project Cancer.",
            f"Index event: {lines.index_event}; lines of therapy by geniekit.lot with the rules in settings.",
        ],
    )
    # labels for stratification: patients with an unknown liver status appear in the first table only
    table["liver_status"] = table["liver_mets"].map(LIVER_LABELS).astype(object)
    table["liver_status"] = table["liver_status"].where(table["liver_mets"].notna(), "Unknown")
    table["liver"] = table["liver_status"].where(table["liver_mets"].notna(), None)
    n_unknown = int(table["liver_mets"].isna().sum())
    if n_unknown:
        report.notes.append(
            f"{n_unknown} patient(s) with an unknown liver metastasis status are counted in "
            "patients_by_liver_mets only."
        )
    adv = table[table["index_days"].notna()].copy()
    adv["lines_group"] = adv["n_lines"].map(_lines_bucket)

    # a. patients stratified by liver metastasis
    report.add(
        frequency_table(table, "liver_status", control=ctl, name="patients_by_liver_mets"),
        "patients_by_liver_mets",
    )
    report.add(
        frequency_table(table, "liver_mets_timing", control=ctl, name="liver_mets_timing"),
        "liver_mets_timing",
    )
    report.add(
        frequency_table(table, "liver", by="stage_dx", control=ctl, name="liver_mets_by_stage", dropna=True),
        "liver_mets_by_stage",
    )
    report.add(frequency_table(table, "sex", by="liver", control=ctl, dropna=True), "sex_by_liver_mets")
    report.add(numeric_summary(table, "age_dx_years", by="liver", control=ctl), "age_by_liver_mets")
    report.add(
        frequency_table(table, "institution", by="liver", control=ctl, dropna=True),
        "institution_by_liver_mets",
    )
    if table["race_ethnicity"].notna().any():
        report.add(
            frequency_table(table, "race_ethnicity", by="liver", control=ctl, dropna=True),
            "race_ethnicity_by_liver_mets",
        )
    report.add(
        frequency_table(
            adv, "first_line", by="liver", control=ctl, name="first line after index by liver", dropna=True
        ),
        "first_line_by_liver_mets",
    )
    report.add(
        frequency_table(
            adv, "lines_group", by="liver", control=ctl, name="lines of therapy by liver", dropna=True
        ),
        "lines_by_liver_mets",
    )
    # b. early treatment failure after first-line therapy
    report.add(
        early_failure_table(adv, months=months, by="liver", control=ctl, reference="No liver metastasis"),
        "early_failure_by_liver_mets",
    )
    report.add(
        km_table(
            adv,
            "os_from_index_days",
            "died",
            by="liver",
            at_months=survival_months,
            control=ctl,
            name="overall survival from index by liver",
        ),
        "overall_survival_by_liver_mets",
    )
    # c. genomic frequencies
    if genomic:
        flags = [f"{g}_mut" for g in genes] + list(combined) + list(variants)
        report.add(
            flag_frequency_table(
                table,
                flags,
                by="liver",
                control=ctl,
                subset="sequenced",
                name="alterations by liver",
            ),
            "alterations_by_liver_mets",
        )
    if "msi_mmr" in table.columns:
        report.add(
            frequency_table(table, "msi_mmr", by="liver", control=ctl, dropna=True), "msi_mmr_by_liver_mets"
        )
    # d. comparisons between genomic subgroups
    if genomic:
        seq = adv[adv["sequenced"]].copy()
        for col, label in [("RAS_mut", "ras"), ("BRAF_V600E", "braf_v600e")]:
            if col in seq.columns:
                seq[label] = seq[col].map({True: f"{col}: yes", False: f"{col}: no"}).astype(object)
                report.add(
                    early_failure_table(seq, months=months, by=label, control=ctl, reference=f"{col}: no"),
                    f"early_failure_by_{label}",
                )
    if "msi_mmr" in adv.columns and adv["msi_mmr"].isin(["MSI-H or dMMR", "MSS or pMMR"]).any():
        tested = adv[adv["msi_mmr"].isin(["MSI-H or dMMR", "MSS or pMMR"])]
        report.add(
            early_failure_table(tested, months=months, by="msi_mmr", control=ctl, reference="MSS or pMMR"),
            "early_failure_by_msi_mmr",
        )
    return report
