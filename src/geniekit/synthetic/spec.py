"""Per-cohort settings for the synthetic generator.

Everything here is arbitrary. The drug menus, gene lists and OncoTree codes
are chosen so that cohort-specific code paths (for example the CRC liver
metastasis example or EGFR-directed therapy in NSCLC) have something to find;
the frequencies are round numbers picked for test coverage and are not
estimates of the GENIE BPC cohorts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["COHORT_SPECS", "CohortSpec"]


@dataclass(frozen=True)
class CohortSpec:
    cohort: str
    ca_type: str
    ca_d_site: str
    oncotree_codes: tuple[str, ...]
    sex_female_fraction: float
    stage_weights: dict[str, float]
    #: regimen menus by line: first-line menus, later-line menus
    first_line: tuple[tuple[str, ...], ...]
    later_line: tuple[tuple[str, ...], ...]
    genes: dict[str, float]
    hotspots: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fusions: tuple[tuple[str, str, str], ...] = ()  # (gene, partner, label)
    cna_genes: dict[str, int] = field(default_factory=dict)  # gene -> typical call (2 amp, -2 deep del)
    tumor_markers: tuple[tuple[str, str, float], ...] = ()  # (tm_type, units, upper limit of normal)
    icdo_sites: tuple[str, ...] = ()
    histology_codes: tuple[int, ...] = ()
    median_os_days: float = 900.0
    median_pfs_days: float = 300.0


COHORT_SPECS: dict[str, CohortSpec] = {
    "NSCLC": CohortSpec(
        cohort="NSCLC",
        ca_type="Non Small Cell Lung Cancer",
        ca_d_site="Lung",
        oncotree_codes=("LUAD", "LUSC", "NSCLC"),
        sex_female_fraction=0.5,
        stage_weights={"Stage I": 0.25, "Stage II": 0.12, "Stage III": 0.23, "Stage IV": 0.40},
        first_line=(
            ("Carboplatin", "Pemetrexed Disodium"),
            ("Carboplatin", "Paclitaxel"),
            ("Cisplatin", "Pemetrexed Disodium"),
            ("Carboplatin", "Pembrolizumab", "Pemetrexed Disodium"),
            ("Pembrolizumab",),
            ("Osimertinib",),
            ("Erlotinib Hydrochloride",),
            ("Investigational Drug",),
        ),
        later_line=(
            ("Docetaxel",),
            ("Nivolumab",),
            ("Osimertinib",),
            ("Docetaxel", "Ramucirumab"),
            ("Gemcitabine Hydrochloride",),
        ),
        genes={
            "TP53": 0.5,
            "KRAS": 0.3,
            "EGFR": 0.2,
            "STK11": 0.15,
            "KEAP1": 0.12,
            "CDKN2A": 0.1,
            "RB1": 0.06,
            "BRAF": 0.04,
            "MET": 0.04,
        },
        hotspots={
            "KRAS": ("G12C", "G12D", "G12V", "G13D"),
            "EGFR": ("L858R", "E746_A750del", "T790M"),
            "BRAF": ("V600E", "G469A"),
        },
        fusions=(
            ("ALK", "EML4", "EML4-ALK fusion"),
            ("ROS1", "CD74", "CD74-ROS1 fusion"),
            ("RET", "KIF5B", "KIF5B-RET fusion"),
        ),
        cna_genes={"MET": 2, "CDKN2A": -2, "EGFR": 2},
        tumor_markers=(),
        icdo_sites=(
            "C34.1    Upper lobe lung",
            "C34.3    Lower lobe lung",
            "C34.9    Lung NOS",
            "C77.1    Intrathoracic lymph nodes",
        ),
        histology_codes=(8140, 8070, 8046, 8255, 8250),
        median_os_days=800,
        median_pfs_days=240,
    ),
    "CRC": CohortSpec(
        cohort="CRC",
        ca_type="Colorectal Cancer",
        ca_d_site="Colon",
        oncotree_codes=("COAD", "READ", "COADREAD"),
        sex_female_fraction=0.47,
        stage_weights={"Stage I": 0.15, "Stage II": 0.2, "Stage III": 0.3, "Stage IV": 0.35},
        first_line=(
            ("Fluorouracil", "Leucovorin Calcium", "Oxaliplatin"),
            ("Bevacizumab", "Fluorouracil", "Leucovorin Calcium", "Oxaliplatin"),
            ("Capecitabine", "Oxaliplatin"),
            ("Fluorouracil", "Irinotecan Hydrochloride", "Leucovorin Calcium"),
            ("Capecitabine",),
            ("Cetuximab", "Fluorouracil", "Irinotecan Hydrochloride", "Leucovorin Calcium"),
        ),
        later_line=(
            ("Bevacizumab", "Fluorouracil", "Irinotecan Hydrochloride", "Leucovorin Calcium"),
            ("Panitumumab",),
            ("Regorafenib",),
            ("Trifluridine", "Tipiracil Hydrochloride"),
            ("Pembrolizumab",),
            ("Investigational Drug",),
        ),
        genes={
            "APC": 0.7,
            "TP53": 0.6,
            "KRAS": 0.45,
            "PIK3CA": 0.18,
            "SMAD4": 0.15,
            "FBXW7": 0.1,
            "BRAF": 0.1,
            "NRAS": 0.05,
            "SOX9": 0.08,
        },
        hotspots={
            "KRAS": ("G12D", "G12V", "G13D", "G12C"),
            "NRAS": ("Q61K", "G12D"),
            "BRAF": ("V600E", "D594G"),
            "PIK3CA": ("E545K", "H1047R"),
        },
        fusions=(("NTRK1", "TPM3", "TPM3-NTRK1 fusion"),),
        cna_genes={"ERBB2": 2, "SMAD4": -2, "MYC": 2},
        tumor_markers=(("CEA", "ng/mL", 5.0), ("CA 19-9", "U/mL", 37.0)),
        icdo_sites=(
            "C18.7    Sigmoid colon",
            "C18.2    Ascending colon",
            "C20.9    Rectum NOS",
            "C22.0    Liver",
        ),
        histology_codes=(8140, 8480, 8263),
        median_os_days=1100,
        median_pfs_days=330,
    ),
    "BrCa": CohortSpec(
        cohort="BrCa",
        ca_type="Breast Cancer",
        ca_d_site="Breast",
        oncotree_codes=("IDC", "ILC", "BRCA"),
        sex_female_fraction=0.98,
        stage_weights={"Stage I": 0.3, "Stage II": 0.35, "Stage III": 0.2, "Stage IV": 0.15},
        first_line=(
            ("Cyclophosphamide", "Doxorubicin Hydrochloride"),
            ("Paclitaxel",),
            ("Tamoxifen Citrate",),
            ("Anastrozole",),
            ("Docetaxel", "Pertuzumab", "Trastuzumab"),
            ("Letrozole", "Palbociclib"),
        ),
        later_line=(
            ("Capecitabine",),
            ("Eribulin Mesylate",),
            ("Fulvestrant",),
            ("Trastuzumab Deruxtecan",),
            ("Investigational Drug",),
        ),
        genes={
            "PIK3CA": 0.35,
            "TP53": 0.35,
            "GATA3": 0.12,
            "CDH1": 0.1,
            "MAP3K1": 0.08,
            "ESR1": 0.05,
            "AKT1": 0.04,
            "PTEN": 0.05,
        },
        hotspots={"PIK3CA": ("H1047R", "E545K", "E542K"), "ESR1": ("D538G", "Y537S"), "AKT1": ("E17K",)},
        fusions=(),
        cna_genes={"ERBB2": 2, "CCND1": 2, "PTEN": -2},
        tumor_markers=(("CA 15-3", "U/mL", 30.0), ("CA 27.29", "U/mL", 38.0), ("CEA", "ng/mL", 5.0)),
        icdo_sites=(
            "C50.4    Upper-outer quadrant of breast",
            "C50.9    Breast NOS",
            "C77.3    Axillary lymph nodes",
            "C22.0    Liver",
        ),
        histology_codes=(8500, 8520, 8522),
        median_os_days=1500,
        median_pfs_days=400,
    ),
    "PANC": CohortSpec(
        cohort="PANC",
        ca_type="Pancreatic Cancer",
        ca_d_site="Pancreas",
        oncotree_codes=("PAAD", "PANET", "PAAC"),
        sex_female_fraction=0.47,
        stage_weights={"Stage I": 0.1, "Stage II": 0.3, "Stage III": 0.2, "Stage IV": 0.4},
        first_line=(
            ("Fluorouracil", "Irinotecan Hydrochloride", "Leucovorin Calcium", "Oxaliplatin"),
            ("Gemcitabine Hydrochloride", "Paclitaxel Albumin-Stabilized Nanoparticle Formulation"),
            ("Gemcitabine Hydrochloride",),
            ("Capecitabine", "Gemcitabine Hydrochloride"),
        ),
        later_line=(
            ("Gemcitabine Hydrochloride", "Paclitaxel Albumin-Stabilized Nanoparticle Formulation"),
            ("Fluorouracil", "Irinotecan Liposome", "Leucovorin Calcium"),
            ("Capecitabine",),
            ("Investigational Drug",),
        ),
        genes={
            "KRAS": 0.9,
            "TP53": 0.7,
            "CDKN2A": 0.4,
            "SMAD4": 0.25,
            "ARID1A": 0.08,
            "BRCA2": 0.04,
            "GNAS": 0.05,
        },
        hotspots={"KRAS": ("G12D", "G12V", "G12R", "Q61H")},
        fusions=(),
        cna_genes={"CDKN2A": -2, "SMAD4": -2, "MYC": 2},
        tumor_markers=(("CA 19-9", "U/mL", 37.0), ("CEA", "ng/mL", 5.0)),
        icdo_sites=(
            "C25.0    Head of pancreas",
            "C25.1    Body of pancreas",
            "C25.9    Pancreas NOS",
            "C22.0    Liver",
        ),
        histology_codes=(8500, 8140, 8246),
        median_os_days=500,
        median_pfs_days=180,
    ),
    "Prostate": CohortSpec(
        cohort="Prostate",
        ca_type="Prostate Cancer",
        ca_d_site="Prostate",
        oncotree_codes=("PRAD", "PRNE", "PROSTATE"),
        sex_female_fraction=0.0,
        stage_weights={"Stage I": 0.15, "Stage II": 0.3, "Stage III": 0.25, "Stage IV": 0.3},
        first_line=(
            ("Leuprolide Acetate",),
            ("Bicalutamide", "Leuprolide Acetate"),
            ("Abiraterone Acetate", "Leuprolide Acetate", "Prednisone"),
            ("Docetaxel", "Leuprolide Acetate"),
            ("Degarelix",),
        ),
        later_line=(
            ("Enzalutamide",),
            ("Abiraterone Acetate", "Prednisone"),
            ("Docetaxel", "Prednisone"),
            ("Cabazitaxel", "Prednisone"),
            ("Radium Ra 223 Dichloride",),
            ("Investigational Drug",),
        ),
        genes={
            "TP53": 0.35,
            "PTEN": 0.2,
            "AR": 0.15,
            "SPOP": 0.1,
            "FOXA1": 0.1,
            "RB1": 0.08,
            "BRCA2": 0.06,
            "ATM": 0.05,
            "CDK12": 0.05,
        },
        hotspots={"SPOP": ("F133V", "W131G"), "AR": ("L702H", "T878A", "H875Y")},
        fusions=(("ERG", "TMPRSS2", "TMPRSS2-ERG fusion"),),
        cna_genes={"AR": 2, "PTEN": -2, "RB1": -2, "MYC": 2},
        tumor_markers=(("PSA", "ng/mL", 4.0), ("Testosterone", "ng/dL", 300.0)),
        icdo_sites=(
            "C61.9    Prostate gland",
            "C77.5    Pelvic lymph nodes",
            "C41.4    Pelvic bones",
            "C22.0    Liver",
        ),
        histology_codes=(8140, 8550, 8041),
        median_os_days=1600,
        median_pfs_days=420,
    ),
    "BLADDER": CohortSpec(
        cohort="BLADDER",
        ca_type="Bladder Cancer",
        ca_d_site="Bladder",
        oncotree_codes=("BLCA", "UTUC", "BLAD"),
        sex_female_fraction=0.25,
        stage_weights={"Stage I": 0.2, "Stage II": 0.3, "Stage III": 0.2, "Stage IV": 0.3},
        first_line=(
            ("Cisplatin", "Gemcitabine Hydrochloride"),
            ("Carboplatin", "Gemcitabine Hydrochloride"),
            ("Cisplatin", "Doxorubicin Hydrochloride", "Methotrexate", "Vinblastine Sulfate"),
            ("Pembrolizumab",),
            ("Atezolizumab",),
        ),
        later_line=(
            ("Enfortumab Vedotin",),
            ("Pembrolizumab",),
            ("Erdafitinib",),
            ("Avelumab",),
            ("Investigational Drug",),
        ),
        genes={
            "TP53": 0.5,
            "KDM6A": 0.25,
            "ARID1A": 0.25,
            "KMT2D": 0.25,
            "PIK3CA": 0.2,
            "CDKN2A": 0.2,
            "FGFR3": 0.15,
            "RB1": 0.15,
            "ERBB2": 0.1,
        },
        hotspots={"FGFR3": ("S249C", "Y373C", "R248C"), "PIK3CA": ("E545K", "E542K")},
        fusions=(("FGFR3", "TACC3", "FGFR3-TACC3 fusion"),),
        cna_genes={"CDKN2A": -2, "ERBB2": 2, "CCND1": 2},
        tumor_markers=(),
        icdo_sites=(
            "C67.9    Bladder NOS",
            "C67.0    Trigone of bladder",
            "C77.5    Pelvic lymph nodes",
            "C22.0    Liver",
        ),
        histology_codes=(8120, 8130, 8070),
        median_os_days=900,
        median_pfs_days=240,
    ),
}
