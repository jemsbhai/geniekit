"""Lines of therapy from the GENIE BPC regimen tables, counted from an index event the user chooses.

Quick start::

    from geniekit import io, lot

    crc = io.load_release("data/genie_bpc/CRC_v2.0-public")
    result = lot.derive(crc, index="advanced")  # from stage IV diagnosis or first distant metastasis
    result.lines      # one row per line: start, end, agents, regimens, why the line ended
    result.regimens   # every regimen with its line number and the reason for the assignment
    result.cancers    # one row per cancer: index event, number of lines, first-line agents

The index event can also be ``"diagnosis"``, ``"first_progression"``,
``"first_systemic"`` or a table of ``record_id``, ``ca_seq``, ``index_days``.
The grouping rules live in :class:`LineRules` (presets ``"default"``,
``"prostate"``, ``"progression"``) and every setting is written into the
output. Drug identities come from the packaged vocabulary
(:func:`load_vocabulary`), which maps the released labels of both schema
generations to canonical agents, classes and roles.
"""

from .engine import LinesOfTherapy, derive, derive_all
from .index import INDEX_EVENTS, index_events
from .regimens import LOCAL_ROUTES, annotate_regimens, progression_events, regimen_drugs
from .rules import PRESETS, LineRules
from .vocabulary import DrugEntry, DrugVocabulary, base_label, load_vocabulary, synonyms

__all__ = [
    "INDEX_EVENTS",
    "LOCAL_ROUTES",
    "PRESETS",
    "DrugEntry",
    "DrugVocabulary",
    "LineRules",
    "LinesOfTherapy",
    "annotate_regimens",
    "base_label",
    "derive",
    "derive_all",
    "index_events",
    "load_vocabulary",
    "progression_events",
    "regimen_drugs",
    "synonyms",
]
