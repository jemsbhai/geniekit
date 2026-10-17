"""Harmonise the genomic tables across releases.

* Structural variants: NSCLC and CRC ship ``data_fusions.txt`` (the retired
  cBioPortal fusion format, one row per gene per fusion), the other cohorts
  ``data_sv.txt`` (one row per event with two sites). Both are mapped onto one
  long schema, one row per sample and event:
  ``sample_id, gene_1, gene_2, event, sv_class, frame, source_format``.
* Copy number: the wide gene-by-sample matrix is melted into
  ``sample_id, hugo_symbol, cna`` for stacking; the wide form is kept too.
* Mutations: the MAF-like file has the same layout in every release;
  ``Tumor_Sample_Barcode`` is also exposed as ``sample_id``.
"""

from __future__ import annotations

import re

import pandas as pd

__all__ = ["cna_to_long", "mutations_with_sample_id", "structural_variants"]

SV_COLUMNS = ["sample_id", "gene_1", "gene_2", "event", "sv_class", "frame", "source_format"]

_FUSION_SPLIT = re.compile(r"\s*[-_/]\s*")


def _split_fusion_label(label: object) -> tuple[str | None, str | None]:
    """``'EML4-ALK fusion'`` -> ``('EML4', 'ALK')``; anything else -> (None, None)."""
    if not isinstance(label, str):
        return None, None
    text = label.replace(" fusion", "").replace(" Fusion", "").strip()
    parts = [p for p in _FUSION_SPLIT.split(text) if p]
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def structural_variants(frame: pd.DataFrame, source_format: str | None = None) -> pd.DataFrame:
    """Map a fusions or SV table onto :data:`SV_COLUMNS`.

    ``source_format`` is inferred from the columns when omitted: ``"fusions"``
    when ``Fusion`` and ``Tumor_Sample_Barcode`` are present, ``"sv"`` when
    ``Sample_Id`` is present.
    """
    cols = set(frame.columns)
    if source_format is None:
        if "Fusion" in cols and "Tumor_Sample_Barcode" in cols:
            source_format = "fusions"
        elif "Sample_Id" in cols:
            source_format = "sv"
        else:
            raise ValueError("Cannot tell whether this is a fusions or an SV table from its columns")
    if frame.empty:
        return pd.DataFrame(columns=SV_COLUMNS)
    if source_format == "fusions":
        rows = []
        for _, r in frame.iterrows():
            g1, g2 = _split_fusion_label(r.get("Fusion"))
            gene = r.get("Hugo_Symbol")
            if g1 is None and isinstance(gene, str):
                g1, g2 = gene, None
            rows.append(
                {
                    "sample_id": r.get("Tumor_Sample_Barcode"),
                    "gene_1": g1,
                    "gene_2": g2,
                    "event": r.get("Fusion"),
                    "sv_class": "FUSION",
                    "frame": r.get("Frame"),
                    "source_format": "fusions",
                }
            )
        out = pd.DataFrame(rows, columns=SV_COLUMNS)
        # the fusion format repeats one event per partner gene; keep one row per sample and event
        return out.drop_duplicates(["sample_id", "event"]).reset_index(drop=True)
    if source_format == "sv":

        def col(*names: str) -> pd.Series:
            for n in names:
                if n in frame.columns:
                    return frame[n].astype(object)
            return pd.Series([None] * len(frame), index=frame.index, dtype=object)

        out = pd.DataFrame(
            {
                "sample_id": col("Sample_Id"),
                "gene_1": col("Site1_Hugo_Symbol"),
                "gene_2": col("Site2_Hugo_Symbol"),
                "event": col("Event_Info"),
                "sv_class": col("Class"),
                "frame": col("Site2_Effect_On_Frame", "Frame"),
                "source_format": "sv",
            },
            index=frame.index,
        )
        return out[SV_COLUMNS].reset_index(drop=True)
    raise ValueError(f"unknown source_format {source_format!r}")


def cna_to_long(frame: pd.DataFrame, drop_zero: bool = True) -> pd.DataFrame:
    """Melt a cBioPortal copy-number matrix into ``sample_id, hugo_symbol, cna``."""
    if frame.empty or "Hugo_Symbol" not in frame.columns:
        return pd.DataFrame(columns=["sample_id", "hugo_symbol", "cna"])
    value_cols = [c for c in frame.columns if c not in ("Hugo_Symbol", "Entrez_Gene_Id")]
    long = frame.melt(id_vars=["Hugo_Symbol"], value_vars=value_cols, var_name="sample_id", value_name="cna")
    long = long.rename(columns={"Hugo_Symbol": "hugo_symbol"})
    long["cna"] = pd.to_numeric(long["cna"], errors="coerce")
    if drop_zero:
        long = long[long["cna"].notna() & (long["cna"] != 0)]
    return long[["sample_id", "hugo_symbol", "cna"]].reset_index(drop=True)


def mutations_with_sample_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the MAF with a leading ``sample_id`` column copied from ``Tumor_Sample_Barcode``."""
    if "Tumor_Sample_Barcode" not in frame.columns or "sample_id" in frame.columns:
        return frame
    out = frame.copy()
    out.insert(0, "sample_id", out["Tumor_Sample_Barcode"])
    return out
