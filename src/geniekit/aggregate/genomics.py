"""Patient-level genomic and biomarker flags for the aggregate layer.

These builders return one row per patient (not per sample) so that the
aggregate functions can count patients. They are inputs to the aggregate
layer, not outputs of it: the tables they return are patient-level and must
stay with users who hold GENIE access.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from geniekit.io.cohort import CohortData

__all__ = [
    "NONSYNONYMOUS",
    "gene_flags",
    "msi_mmr_status",
    "sample_patients",
]

# MAF Variant_Classification values that change the protein (the usual "non-synonymous" filter)
NONSYNONYMOUS: frozenset[str] = frozenset(
    {
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Nonstop_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "In_Frame_Del",
        "In_Frame_Ins",
        "Splice_Site",
        "Translation_Start_Site",
    }
)
MSI_HIGH = "MSI-H: HIGH"
MMR_DEFICIENT_PREFIX = "Loss of nuclear expression"
MMR_PROFICIENT_PREFIX = "No loss of nuclear expression"
MSS = "MSS: STABLE"
MSI_LOW_PREFIX = "MSI-L"


def sample_patients(data: CohortData) -> pd.DataFrame:
    """Sample to patient map from the ``cpt`` table: ``sample_id``, ``record_id``, ``ca_seq``, ``assay``."""
    if "cpt" not in data:
        raise KeyError(f"{data.label} has no cpt table; sample-to-patient links need it")
    cpt = data["cpt"]
    out = pd.DataFrame(
        {
            "sample_id": cpt["cpt_genie_sample_id"].astype(str),
            "record_id": cpt["record_id"].astype(str),
            "ca_seq": pd.to_numeric(cpt.get("ca_seq"), errors="coerce").astype("Int64")
            if "ca_seq" in cpt.columns
            else pd.array([pd.NA] * len(cpt), dtype="Int64"),
            "assay": cpt["cpt_seq_assay_id"].astype("string") if "cpt_seq_assay_id" in cpt.columns else pd.NA,
            "sample_type": cpt["sample_type"].astype("string") if "sample_type" in cpt.columns else pd.NA,
        }
    )
    return out.drop_duplicates(["sample_id", "record_id"]).reset_index(drop=True)


def gene_flags(
    data: CohortData,
    genes: Iterable[str],
    variants: Mapping[str, str] | None = None,
    combined: Mapping[str, Iterable[str]] | None = None,
    classifications: Iterable[str] = NONSYNONYMOUS,
) -> pd.DataFrame:
    """One row per sequenced patient with a boolean per requested gene and variant.

    ``genes`` give columns ``<GENE>_mut`` (any mutation of the listed
    classifications in any of the patient's samples); ``variants`` maps a
    column name to a ``GENE p.Change`` specification, for example
    ``{"BRAF_V600E": "BRAF p.V600E"}``; ``combined`` maps a column name to
    genes any of which counts, for example ``{"RAS_mut": ["KRAS", "NRAS"]}``.
    Patients are those with at least one sample in the ``cpt`` table
    (``sequenced`` is True for every row); the mutation table must be the
    release MAF with ``Hugo_Symbol``, ``Tumor_Sample_Barcode``,
    ``Variant_Classification`` and ``HGVSp_Short``.

    The denominator makes no allowance for panel content: a patient whose
    panel did not cover a gene counts as unmutated for it. The genes in the
    CRC example are on every panel the contributing centres used, but check
    panel coverage (``genomic_information`` in generation-2 releases) before
    applying the flags to rarely covered genes.
    """
    if "mutations" not in data:
        raise KeyError(f"{data.label} has no mutations table")
    samples = sample_patients(data)
    maf = data["mutations"]
    keep = maf["Variant_Classification"].astype(str).isin(set(classifications))
    maf = maf.loc[keep, ["Hugo_Symbol", "Tumor_Sample_Barcode", "HGVSp_Short"]].copy()
    maf["Tumor_Sample_Barcode"] = maf["Tumor_Sample_Barcode"].astype(str)
    hits = maf.merge(
        samples[["sample_id", "record_id"]], left_on="Tumor_Sample_Barcode", right_on="sample_id", how="inner"
    )
    patients = samples["record_id"].drop_duplicates().sort_values().reset_index(drop=True)
    out = pd.DataFrame({"record_id": patients, "sequenced": True})
    out["n_samples"] = out["record_id"].map(samples.groupby("record_id").size()).astype(int)
    genes = list(dict.fromkeys(genes))
    for gene in genes:
        mutated = set(hits.loc[hits["Hugo_Symbol"] == gene, "record_id"])
        out[f"{gene}_mut"] = out["record_id"].isin(mutated)
    for name, spec in (variants or {}).items():
        gene, change = spec.split(None, 1)
        mutated = set(
            hits.loc[(hits["Hugo_Symbol"] == gene) & (hits["HGVSp_Short"].astype(str) == change), "record_id"]
        )
        out[name] = out["record_id"].isin(mutated)
    for name, members in (combined or {}).items():
        mutated = set(hits.loc[hits["Hugo_Symbol"].isin(list(members)), "record_id"])
        out[name] = out["record_id"].isin(mutated)
    return out


def msi_mmr_status(data: CohortData) -> pd.DataFrame:
    """One row per patient with an MSI/MMR status from the pathology reports.

    ``msi_mmr`` is ``"MSI-H or dMMR"`` when any report records MSI-high
    (``msi_result*`` equal to ``"MSI-H: HIGH"``) or loss of nuclear
    expression of an MMR protein (``mmr_result*``), ``"MSS or pMMR"`` when
    every informative result is stable, low or proficient, ``"indeterminate"``
    when the only results are indeterminate, and ``"not tested"`` otherwise.
    Patients without a pathology row are ``"not tested"``.
    """
    if "pathology" not in data:
        raise KeyError(f"{data.label} has no pathology table")
    path = data["pathology"]
    msi_cols = [c for c in path.columns if c == "msi_result" or c.startswith("msi_result_")]
    mmr_cols = [c for c in path.columns if c == "mmr_result" or c.startswith("mmr_result_")]
    rec = path["record_id"].astype(str)
    high = pd.Series(False, index=path.index)
    stable = pd.Series(False, index=path.index)
    indeterminate = pd.Series(False, index=path.index)
    for col in msi_cols:
        v = path[col].astype("string")
        high |= v.eq(MSI_HIGH).fillna(False)
        stable |= (v.eq(MSS) | v.str.startswith(MSI_LOW_PREFIX)).fillna(False)
        indeterminate |= v.str.startswith("Indeterminate").fillna(False)
    for col in mmr_cols:
        v = path[col].astype("string")
        high |= v.str.startswith(MMR_DEFICIENT_PREFIX).fillna(False)
        stable |= v.str.startswith(MMR_PROFICIENT_PREFIX).fillna(False)
        indeterminate |= v.str.startswith("Indeterminate").fillna(False)
    per = pd.DataFrame({"record_id": rec, "high": high, "stable": stable, "indeterminate": indeterminate})
    agg = per.groupby("record_id")[["high", "stable", "indeterminate"]].any()
    status = pd.Series("not tested", index=agg.index, dtype=object)
    status[agg["indeterminate"]] = "indeterminate"
    status[agg["stable"]] = "MSS or pMMR"
    status[agg["high"]] = "MSI-H or dMMR"
    out = pd.DataFrame({"record_id": agg.index.astype(str), "msi_mmr": status.to_numpy()})
    out["msi_mmr_tested"] = out["msi_mmr"] != "not tested"
    return out.reset_index(drop=True)
