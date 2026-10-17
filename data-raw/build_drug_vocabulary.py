"""Build ``resources/drug_vocabulary.csv``: one row per released drug label.

The regimen tables name drugs as ``"<label>(<synonyms>)"``, for example
``"Gemcitabine HCL(FF10832,Gemzar,LY188011,dFdCyd)"``. The label before the
parenthesis is the identity geniekit works with. Labels differ between the
two schema generations in salt-form spelling (``Gemcitabine Hydrochloride``
versus ``Gemcitabine HCL``, ``Leucovorin Calcium`` versus ``Leucovorin``,
``Tamoxifen Citrate`` versus ``Tamoxifen``) and in one misspelling
(``Goserlin Acetate``), and the synonym lists differ far more, so every label
is mapped here to a canonical agent name, a pharmacological class and the
role it plays in line-of-therapy derivation.

Classes and roles were assigned by the package author from the standard
references (the NCI Drug Dictionary and FDA labelling) and are recorded in
this file so that they can be reviewed and corrected in one place. A label
that appears in a release but not in this table makes the build fail, and
``geniekit.lot`` reports such labels as ``unclassified`` at run time rather
than guessing.

Roles (``lot_role``):

- ``systemic``: a cancer-directed systemic agent; starts, joins and ends
  lines of therapy.
- ``backbone``: androgen deprivation (LHRH agonists and antagonists) that is
  continued across lines; recorded as a line component but never starts or
  ends a line.
- ``modulator``: not an antineoplastic in its own right (leucovorin); its
  presence or absence does not distinguish regimens.
- ``local``: a locally administered therapy (intravesical BCG, valrubicin);
  regimens made only of these are not lines of systemic therapy.
- ``investigational``: an investigational drug whose identity and end date
  are masked; counted as systemic, but never matched across regimens.
- ``other``: not cancer-directed in the sense of the regimen tables
  (vedolizumab, methoxsalen, dutasteride) or of unknown identity; ignored by
  default.

Run from the repository root. With ``GENIEKIT_DATA_DIR`` set to a folder of
downloaded releases, the script also checks that every label in the regimen
tables is covered::

    python data-raw/build_drug_vocabulary.py
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "geniekit" / "resources" / "drug_vocabulary.csv"

C, E, IMM, T, A, R, OTH, INV = (
    "chemotherapy",
    "endocrine",
    "immunotherapy",
    "targeted",
    "antibody-drug conjugate",
    "radiopharmaceutical",
    "other",
    "investigational",
)

# label, canonical agent(s) (";"-separated when the product combines agents), class, subclass, role, note
CURATED: list[tuple[str, str, str, str, str, str]] = [
    ("ADT LHRH agonist not specified", "LHRH agonist (unspecified)", E, "LHRH agonist", "backbone", ""),
    ("Abemaciclib", "Abemaciclib", T, "CDK4/6 inhibitor", "systemic", ""),
    (
        "Abiraterone Acetate",
        "Abiraterone",
        E,
        "androgen receptor pathway inhibitor",
        "systemic",
        "CYP17 inhibitor",
    ),
    ("Acalabrutinib", "Acalabrutinib", T, "BTK inhibitor", "systemic", ""),
    ("Afatinib Dimaleate", "Afatinib", T, "EGFR tyrosine kinase inhibitor", "systemic", ""),
    ("Aldesleukin", "Aldesleukin", IMM, "interleukin-2", "systemic", ""),
    ("Alectinib", "Alectinib", T, "ALK tyrosine kinase inhibitor", "systemic", ""),
    ("Alpelisib", "Alpelisib", T, "PI3K inhibitor", "systemic", ""),
    ("Anastrozole", "Anastrozole", E, "aromatase inhibitor", "systemic", ""),
    ("Apalutamide", "Apalutamide", E, "androgen receptor pathway inhibitor", "systemic", ""),
    ("Apatinib", "Apatinib", T, "VEGFR tyrosine kinase inhibitor", "systemic", ""),
    ("Arsenic Trioxide", "Arsenic trioxide", C, "other", "systemic", ""),
    ("Atezolizumab", "Atezolizumab", IMM, "PD-L1 inhibitor", "systemic", ""),
    (
        "Autologous Melanoma Lysate Pulsed Dendritic Cell Vaccine",
        "Autologous dendritic cell vaccine",
        IMM,
        "cancer vaccine",
        "systemic",
        "",
    ),
    ("Avelumab", "Avelumab", IMM, "PD-L1 inhibitor", "systemic", ""),
    ("Azacitidine", "Azacitidine", C, "antimetabolite", "systemic", "hypomethylating agent"),
    ("BCG Solution", "BCG", IMM, "BCG", "local", "intravesical"),
    ("BCG Vaccine", "BCG", IMM, "BCG", "local", "intravesical"),
    ("Bendamustine", "Bendamustine", C, "alkylating agent", "systemic", ""),
    ("Bendamustine Hydrochloride", "Bendamustine", C, "alkylating agent", "systemic", ""),
    ("Bevacizumab", "Bevacizumab", T, "VEGF antibody", "systemic", ""),
    ("Bicalutamide", "Bicalutamide", E, "first-generation antiandrogen", "systemic", ""),
    ("Binimetinib", "Binimetinib", T, "MEK inhibitor", "systemic", ""),
    ("Bleomycin", "Bleomycin", C, "antitumor antibiotic", "systemic", ""),
    ("Bortezomib", "Bortezomib", T, "proteasome inhibitor", "systemic", ""),
    ("Bosutinib", "Bosutinib", T, "BCR-ABL tyrosine kinase inhibitor", "systemic", ""),
    ("Brentuximab Vedotin", "Brentuximab vedotin", A, "CD30 antibody-drug conjugate", "systemic", ""),
    ("Brigatinib", "Brigatinib", T, "ALK tyrosine kinase inhibitor", "systemic", ""),
    ("Busulfan", "Busulfan", C, "alkylating agent", "systemic", ""),
    ("Cabazitaxel", "Cabazitaxel", C, "taxane", "systemic", ""),
    ("Cabozantinib Smalate", "Cabozantinib", T, "multikinase inhibitor", "systemic", "MET, VEGFR, RET"),
    ("Capecitabine", "Capecitabine", C, "antimetabolite", "systemic", "fluoropyrimidine"),
    ("Carboplatin", "Carboplatin", C, "platinum", "systemic", ""),
    ("Carfilzomib", "Carfilzomib", T, "proteasome inhibitor", "systemic", ""),
    ("Carmustine", "Carmustine", C, "alkylating agent", "systemic", "nitrosourea"),
    ("Ceritinib", "Ceritinib", T, "ALK tyrosine kinase inhibitor", "systemic", ""),
    ("Cetuximab", "Cetuximab", T, "EGFR antibody", "systemic", ""),
    ("Chlorambucil", "Chlorambucil", C, "alkylating agent", "systemic", ""),
    ("Cisplatin", "Cisplatin", C, "platinum", "systemic", ""),
    ("Cladribine", "Cladribine", C, "antimetabolite", "systemic", "purine analogue"),
    ("Cobimetinib", "Cobimetinib", T, "MEK inhibitor", "systemic", ""),
    ("Crizotinib", "Crizotinib", T, "ALK tyrosine kinase inhibitor", "systemic", "ALK, ROS1, MET"),
    ("Cyclophosphamide", "Cyclophosphamide", C, "alkylating agent", "systemic", ""),
    ("Cytarabine", "Cytarabine", C, "antimetabolite", "systemic", ""),
    (
        "Cytarabine liposomal",
        "Cytarabine liposomal",
        C,
        "antimetabolite",
        "systemic",
        "liposomal formulation",
    ),
    ("Dabrafenib", "Dabrafenib", T, "BRAF inhibitor", "systemic", ""),
    ("Dacarbazine", "Dacarbazine", C, "alkylating agent", "systemic", ""),
    ("Dactinomycin", "Dactinomycin", C, "antitumor antibiotic", "systemic", ""),
    ("Daratumumab", "Daratumumab", T, "CD38 antibody", "systemic", ""),
    ("Darolutamide", "Darolutamide", E, "androgen receptor pathway inhibitor", "systemic", ""),
    ("Dasatinib", "Dasatinib", T, "BCR-ABL tyrosine kinase inhibitor", "systemic", ""),
    ("Daunorubicin Hydrochloride", "Daunorubicin", C, "anthracycline", "systemic", ""),
    ("Decitabine", "Decitabine", C, "antimetabolite", "systemic", "hypomethylating agent"),
    ("Degarelix", "Degarelix", E, "LHRH antagonist", "backbone", ""),
    ("Docetaxel", "Docetaxel", C, "taxane", "systemic", ""),
    ("Doxorubicin HCL", "Doxorubicin", C, "anthracycline", "systemic", ""),
    ("Doxorubicin Hydrochloride", "Doxorubicin", C, "anthracycline", "systemic", ""),
    ("Durvalumab", "Durvalumab", IMM, "PD-L1 inhibitor", "systemic", ""),
    ("Dutasteride", "Dutasteride", OTH, "5-alpha-reductase inhibitor", "other", "not an antineoplastic"),
    ("Elotuzumab", "Elotuzumab", T, "SLAMF7 antibody", "systemic", ""),
    ("Encorafenib", "Encorafenib", T, "BRAF inhibitor", "systemic", ""),
    ("Enfortumab Vedotin", "Enfortumab vedotin", A, "Nectin-4 antibody-drug conjugate", "systemic", ""),
    ("Ensartinib", "Ensartinib", T, "ALK tyrosine kinase inhibitor", "systemic", ""),
    ("Entrectinib", "Entrectinib", T, "NTRK/ROS1 tyrosine kinase inhibitor", "systemic", ""),
    ("Enzalutamide", "Enzalutamide", E, "androgen receptor pathway inhibitor", "systemic", ""),
    ("Epirubicin HCL", "Epirubicin", C, "anthracycline", "systemic", ""),
    ("Epirubicin Hydrochloride", "Epirubicin", C, "anthracycline", "systemic", ""),
    ("Erdafitinib", "Erdafitinib", T, "FGFR inhibitor", "systemic", ""),
    ("Eribulin Mesylate", "Eribulin", C, "microtubule inhibitor", "systemic", "halichondrin analogue"),
    ("Erlotinib HCL", "Erlotinib", T, "EGFR tyrosine kinase inhibitor", "systemic", ""),
    ("Erlotinib Hydrochloride", "Erlotinib", T, "EGFR tyrosine kinase inhibitor", "systemic", ""),
    ("Estramustine", "Estramustine", C, "other", "systemic", "estradiol-nitrogen mustard conjugate"),
    ("Etoposide", "Etoposide", C, "topoisomerase II inhibitor", "systemic", ""),
    ("Etoposide Phosphate", "Etoposide", C, "topoisomerase II inhibitor", "systemic", "prodrug of etoposide"),
    ("Everolimus", "Everolimus", T, "mTOR inhibitor", "systemic", ""),
    ("Exemestane", "Exemestane", E, "aromatase inhibitor", "systemic", ""),
    (
        "Floxuridine",
        "Floxuridine",
        C,
        "antimetabolite",
        "systemic",
        "fluoropyrimidine; often hepatic arterial",
    ),
    ("Fludarabine Phosphate", "Fludarabine", C, "antimetabolite", "systemic", "purine analogue"),
    ("Fluorouracil", "Fluorouracil", C, "antimetabolite", "systemic", "fluoropyrimidine"),
    ("Flutamide", "Flutamide", E, "first-generation antiandrogen", "systemic", ""),
    ("Fulvestrant", "Fulvestrant", E, "selective estrogen receptor degrader", "systemic", ""),
    ("Gefitinib", "Gefitinib", T, "EGFR tyrosine kinase inhibitor", "systemic", ""),
    ("Gemcitabine HCL", "Gemcitabine", C, "antimetabolite", "systemic", ""),
    ("Gemcitabine Hydrochloride", "Gemcitabine", C, "antimetabolite", "systemic", ""),
    ("Goserlin Acetate", "Goserelin", E, "LHRH agonist", "backbone", "released label is misspelled"),
    ("Histrelin Acetate", "Histrelin", E, "LHRH agonist", "backbone", ""),
    ("Hydroxyurea", "Hydroxyurea", C, "antimetabolite", "systemic", ""),
    ("Ibrutinib", "Ibrutinib", T, "BTK inhibitor", "systemic", ""),
    ("Icotinib", "Icotinib", T, "EGFR tyrosine kinase inhibitor", "systemic", ""),
    ("Ifosfamide", "Ifosfamide", C, "alkylating agent", "systemic", ""),
    ("Imatinib Mesylate", "Imatinib", T, "BCR-ABL tyrosine kinase inhibitor", "systemic", "also KIT, PDGFR"),
    ("Interferon", "Interferon (unspecified)", IMM, "interferon", "systemic", ""),
    ("Interferon Alfacon1", "Interferon alfacon-1", IMM, "interferon", "systemic", ""),
    (
        "Investigational Drug",
        "Investigational Drug",
        INV,
        "investigational",
        "investigational",
        "identity masked",
    ),
    ("Iodine I 131 Tositumomab", "Iodine-131 tositumomab", R, "radiolabelled CD20 antibody", "systemic", ""),
    ("Ipilimumab", "Ipilimumab", IMM, "CTLA-4 inhibitor", "systemic", ""),
    ("Irinotecan HCL", "Irinotecan", C, "topoisomerase I inhibitor", "systemic", ""),
    ("Irinotecan Hydrochloride", "Irinotecan", C, "topoisomerase I inhibitor", "systemic", ""),
    (
        "Irinotecan liposome",
        "Irinotecan liposomal",
        C,
        "topoisomerase I inhibitor",
        "systemic",
        "liposomal formulation",
    ),
    ("Ixabepilone", "Ixabepilone", C, "epothilone", "systemic", ""),
    ("Ixazomib Citrate", "Ixazomib", T, "proteasome inhibitor", "systemic", ""),
    (
        "Ketoconazole",
        "Ketoconazole",
        E,
        "androgen synthesis inhibitor",
        "systemic",
        "historical CRPC therapy",
    ),
    ("Lanreotide Acetate", "Lanreotide", T, "somatostatin analogue", "systemic", ""),
    ("Lapatinib Ditosylate", "Lapatinib", T, "HER2 tyrosine kinase inhibitor", "systemic", ""),
    ("Larotrectinib", "Larotrectinib", T, "NTRK tyrosine kinase inhibitor", "systemic", ""),
    ("Lenalidomide", "Lenalidomide", T, "immunomodulatory imide", "systemic", ""),
    ("Lenvatinib Mesylate", "Lenvatinib", T, "multikinase inhibitor", "systemic", "VEGFR, FGFR, RET"),
    ("Letrozole", "Letrozole", E, "aromatase inhibitor", "systemic", ""),
    ("Leucovorin", "Leucovorin", OTH, "folate biomodulator", "modulator", "not an antineoplastic"),
    ("Leucovorin Calcium", "Leucovorin", OTH, "folate biomodulator", "modulator", "not an antineoplastic"),
    ("Leuprolide", "Leuprolide", E, "LHRH agonist", "backbone", ""),
    ("Leuprolide Acetate", "Leuprolide", E, "LHRH agonist", "backbone", ""),
    ("Lomustine", "Lomustine", C, "alkylating agent", "systemic", "nitrosourea"),
    ("Lorlatinib", "Lorlatinib", T, "ALK tyrosine kinase inhibitor", "systemic", ""),
    ("Lurbinectedin", "Lurbinectedin", C, "other", "systemic", "DNA minor-groove binder"),
    (
        "Lutetium Lu 177 Dotatate",
        "Lutetium-177 dotatate",
        R,
        "radiolabelled somatostatin analogue",
        "systemic",
        "",
    ),
    ("Mechlorethamine Hydrochloride", "Mechlorethamine", C, "alkylating agent", "systemic", ""),
    ("Megestrol Acetate", "Megestrol", E, "progestin", "systemic", ""),
    ("Melphalan", "Melphalan", C, "alkylating agent", "systemic", ""),
    ("Methotrexate", "Methotrexate", C, "antimetabolite", "systemic", "antifolate"),
    ("Methoxsalen", "Methoxsalen", OTH, "psoralen", "other", "photopheresis agent; not an antineoplastic"),
    (
        "Mitomycin",
        "Mitomycin",
        C,
        "antitumor antibiotic",
        "systemic",
        "intravesical in bladder cancer: see route",
    ),
    ("Mitotane", "Mitotane", OTH, "adrenolytic", "systemic", ""),
    ("Mitoxantrone HCL", "Mitoxantrone", C, "anthracenedione", "systemic", ""),
    ("Nabpaclitaxel", "Nab-paclitaxel", C, "taxane", "systemic", "albumin-bound formulation"),
    ("Navitoclax", "Navitoclax", T, "BCL-2 family inhibitor", "systemic", "investigational agent"),
    ("Necitumumab", "Necitumumab", T, "EGFR antibody", "systemic", ""),
    ("Neratinib", "Neratinib", T, "HER2 tyrosine kinase inhibitor", "systemic", ""),
    ("Nilotinib", "Nilotinib", T, "BCR-ABL tyrosine kinase inhibitor", "systemic", ""),
    ("Nilutamide", "Nilutamide", E, "first-generation antiandrogen", "systemic", ""),
    ("Nivolumab", "Nivolumab", IMM, "PD-1 inhibitor", "systemic", ""),
    ("Obinutuzumab", "Obinutuzumab", T, "CD20 antibody", "systemic", ""),
    ("Octreotide", "Octreotide", T, "somatostatin analogue", "systemic", ""),
    ("Olaparib", "Olaparib", T, "PARP inhibitor", "systemic", ""),
    ("Olaratumab", "Olaratumab", T, "PDGFR-alpha antibody", "systemic", ""),
    ("Osimertinib", "Osimertinib", T, "EGFR tyrosine kinase inhibitor", "systemic", ""),
    ("Other NOS", "Other (not otherwise specified)", OTH, "unspecified", "systemic", "identity not recorded"),
    ("Other antineoplastic", "Other antineoplastic", OTH, "unspecified", "systemic", "identity not recorded"),
    ("Other hormone", "Other hormone", E, "unspecified", "systemic", "identity not recorded"),
    ("Oxaliplatin", "Oxaliplatin", C, "platinum", "systemic", ""),
    ("Paclitaxel", "Paclitaxel", C, "taxane", "systemic", ""),
    (
        "Paclitaxel Loaded Polymeric Micelle",
        "Paclitaxel micellar",
        C,
        "taxane",
        "systemic",
        "micellar formulation",
    ),
    ("Paclitaxel Poliglumex", "Paclitaxel poliglumex", C, "taxane", "systemic", "polymer conjugate"),
    ("Paclitaxel Trevatide", "Paclitaxel trevatide", C, "taxane", "systemic", "peptide conjugate"),
    ("Palbociclib", "Palbociclib", T, "CDK4/6 inhibitor", "systemic", ""),
    ("Panitumumab", "Panitumumab", T, "EGFR antibody", "systemic", ""),
    ("Pazopanib HCL", "Pazopanib", T, "multikinase inhibitor", "systemic", "VEGFR, PDGFR, KIT"),
    ("Peginterferon Alfa2b", "Peginterferon alfa-2b", IMM, "interferon", "systemic", ""),
    (
        "Pegylated Liposomal Doxorubicin",
        "Pegylated liposomal doxorubicin",
        C,
        "anthracycline",
        "systemic",
        "",
    ),
    (
        "Pegylated Liposomal Doxorubicin Hydrochloride",
        "Pegylated liposomal doxorubicin",
        C,
        "anthracycline",
        "systemic",
        "",
    ),
    ("Pembrolizumab", "Pembrolizumab", IMM, "PD-1 inhibitor", "systemic", ""),
    ("Pemetrexed Disodium", "Pemetrexed", C, "antimetabolite", "systemic", "antifolate"),
    ("Pertuzumab", "Pertuzumab", T, "HER2 antibody", "systemic", ""),
    (
        "Pertuzumab-Trastuzumab-Hyaluronidase-ZZXF",
        "Pertuzumab;Trastuzumab",
        T,
        "HER2 antibody",
        "systemic",
        "fixed-dose subcutaneous combination of two agents",
    ),
    ("Pomalidomide", "Pomalidomide", T, "immunomodulatory imide", "systemic", ""),
    ("Ponatinib HCL", "Ponatinib", T, "BCR-ABL tyrosine kinase inhibitor", "systemic", ""),
    ("Procarbazine HCL", "Procarbazine", C, "alkylating agent", "systemic", ""),
    ("Procarbazine Hydrochloride", "Procarbazine", C, "alkylating agent", "systemic", ""),
    ("Radium RA 223 Dichloride", "Radium-223", R, "alpha emitter", "systemic", ""),
    ("Ramucirumab", "Ramucirumab", T, "VEGFR2 antibody", "systemic", ""),
    ("Recombinant Interferon Alfa", "Interferon alfa", IMM, "interferon", "systemic", ""),
    ("Recombinant Interferon Alfa2a", "Interferon alfa-2a", IMM, "interferon", "systemic", ""),
    ("Regorafenib", "Regorafenib", T, "multikinase inhibitor", "systemic", "VEGFR, TIE2, KIT, RET, RAF"),
    ("Ribociclib", "Ribociclib", T, "CDK4/6 inhibitor", "systemic", ""),
    ("Rituximab", "Rituximab", T, "CD20 antibody", "systemic", ""),
    (
        "Rituximab and Hyaluronidase Human",
        "Rituximab",
        T,
        "CD20 antibody",
        "systemic",
        "subcutaneous formulation",
    ),
    ("Rociletinib", "Rociletinib", T, "EGFR tyrosine kinase inhibitor", "systemic", "investigational agent"),
    ("Rucaparib Camsylate", "Rucaparib", T, "PARP inhibitor", "systemic", ""),
    ("Ruxolitinib Phosphate", "Ruxolitinib", T, "JAK inhibitor", "systemic", ""),
    ("Sacituzumab Govitecan", "Sacituzumab govitecan", A, "Trop-2 antibody-drug conjugate", "systemic", ""),
    ("Sipuleucel T", "Sipuleucel-T", IMM, "cancer vaccine", "systemic", "autologous cellular immunotherapy"),
    ("Sorafenib Tosylate", "Sorafenib", T, "multikinase inhibitor", "systemic", "VEGFR, PDGFR, RAF"),
    ("Sunitinib Malate", "Sunitinib", T, "multikinase inhibitor", "systemic", "VEGFR, PDGFR, KIT"),
    ("Talazoparib", "Talazoparib", T, "PARP inhibitor", "systemic", ""),
    (
        "Talimogene Laherparepvec",
        "Talimogene laherparepvec",
        IMM,
        "oncolytic virus",
        "local",
        "intralesional",
    ),
    ("Tamoxifen", "Tamoxifen", E, "selective estrogen receptor modulator", "systemic", ""),
    ("Tamoxifen Citrate", "Tamoxifen", E, "selective estrogen receptor modulator", "systemic", ""),
    ("Taselisib", "Taselisib", T, "PI3K inhibitor", "systemic", "investigational agent"),
    (
        "Tegafurgimeraciloteracil Potassium",
        "Tegafur-gimeracil-oteracil (S-1)",
        C,
        "antimetabolite",
        "systemic",
        "fluoropyrimidine",
    ),
    ("Temozolomide", "Temozolomide", C, "alkylating agent", "systemic", ""),
    ("Temsirolimus", "Temsirolimus", T, "mTOR inhibitor", "systemic", ""),
    ("Thalidomide", "Thalidomide", T, "immunomodulatory imide", "systemic", ""),
    ("Thiotepa", "Thiotepa", C, "alkylating agent", "systemic", ""),
    ("Topotecan HCL", "Topotecan", C, "topoisomerase I inhibitor", "systemic", ""),
    ("Toremifene", "Toremifene", E, "selective estrogen receptor modulator", "systemic", ""),
    ("Trametinib", "Trametinib", T, "MEK inhibitor", "systemic", ""),
    ("Trastuzumab", "Trastuzumab", T, "HER2 antibody", "systemic", ""),
    ("Trastuzumab Deruxtecan", "Trastuzumab deruxtecan", A, "HER2 antibody-drug conjugate", "systemic", ""),
    ("Trastuzumab Emtansine", "Trastuzumab emtansine", A, "HER2 antibody-drug conjugate", "systemic", ""),
    (
        "Trastuzumab/Hyaluronidase-oysk",
        "Trastuzumab",
        T,
        "HER2 antibody",
        "systemic",
        "subcutaneous formulation",
    ),
    ("Tremelimumab", "Tremelimumab", IMM, "CTLA-4 inhibitor", "systemic", ""),
    ("Tretinoin", "Tretinoin", OTH, "differentiating agent", "systemic", ""),
    (
        "Trifluridine and Tipiracil Hydrochloride",
        "Trifluridine/tipiracil",
        C,
        "antimetabolite",
        "systemic",
        "fluoropyrimidine",
    ),
    ("Triptorelin", "Triptorelin", E, "LHRH agonist", "backbone", ""),
    ("Tucatinib", "Tucatinib", T, "HER2 tyrosine kinase inhibitor", "systemic", ""),
    ("Valrubicin", "Valrubicin", C, "anthracycline", "local", "intravesical"),
    ("Vedolizumab", "Vedolizumab", OTH, "integrin antibody", "other", "not an antineoplastic"),
    ("Vemurafenib", "Vemurafenib", T, "BRAF inhibitor", "systemic", ""),
    ("Venetoclax", "Venetoclax", T, "BCL-2 inhibitor", "systemic", ""),
    ("Vinblastine Sulfate", "Vinblastine", C, "vinca alkaloid", "systemic", ""),
    ("Vincristine Sulfate", "Vincristine", C, "vinca alkaloid", "systemic", ""),
    (
        "Vincristine Sulfate Liposome",
        "Vincristine liposomal",
        C,
        "vinca alkaloid",
        "systemic",
        "liposomal formulation",
    ),
    ("Vinorelbine Tartrate", "Vinorelbine", C, "vinca alkaloid", "systemic", ""),
    (
        "Yttrium Y90 Ibritumomab Tiuxetan",
        "Yttrium-90 ibritumomab tiuxetan",
        R,
        "radiolabelled CD20 antibody",
        "systemic",
        "",
    ),
    ("Ziv Aflibercept", "Ziv-aflibercept", T, "VEGF trap", "systemic", ""),
]

ROLES = {"systemic", "backbone", "modulator", "local", "investigational", "other"}


def base_label(value: str) -> str:
    """The label before the parenthesised synonym list, whitespace-normalised."""
    return re.sub(r"\s+", " ", str(value).split("(")[0]).strip()


def released_labels(data_dir: Path) -> dict[str, set[str]]:
    """Distinct base labels per cohort from downloaded regimen tables (optional check)."""
    sys.path.insert(0, str(ROOT / "src"))
    import warnings

    from geniekit import io

    found: dict[str, set[str]] = {}
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = io.load_release(folder, tables=["regimen"], genomic=False)
        except Exception as exc:  # report and continue
            print(f"skipping {folder.name}: {exc}")
            continue
        reg = data["regimen"]
        labels = set()
        for k in range(1, 6):
            col = f"drugs_drug_{k}"
            if col in reg.columns:
                labels.update(base_label(v) for v in reg[col].dropna())
        found[data.cohort] = labels
    return found


def main() -> None:
    labels = [row[0] for row in CURATED]
    assert len(labels) == len(set(labels)), "duplicate labels in CURATED"
    for row in CURATED:
        assert row[4] in ROLES, row
    observed: dict[str, set[str]] = {}
    data_dir = os.environ.get("GENIEKIT_DATA_DIR")
    if data_dir and Path(data_dir).is_dir():
        found = released_labels(Path(data_dir))
        for cohort, labs in found.items():
            missing = sorted(labs - set(labels))
            if missing:
                sys.exit(f"{cohort}: labels in the release but not in CURATED: {missing}")
            for lab in labs:
                observed.setdefault(lab, set()).add(cohort)
        print(f"checked {sum(len(v) for v in found.values())} label occurrences in {len(found)} releases")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["label", "agents", "drug_class", "drug_subclass", "lot_role", "cohorts_observed", "note"]
        )
        for label, agents, cls, sub, role, note in sorted(CURATED, key=lambda r: r[0].lower()):
            writer.writerow([label, agents, cls, sub, role, ",".join(sorted(observed.get(label, ()))), note])
    print(f"wrote {OUT} ({len(CURATED)} labels, {len({r[1] for r in CURATED})} canonical agent strings)")


if __name__ == "__main__":
    main()
