"""Loading GENIE BPC releases: from Synapse (with a personal access token) or
from files already on disk. Both paths produce a :class:`CohortData`.

>>> from geniekit import io
>>> data = io.load_release("data/genie_bpc/CRC_v2.0-public")   # doctest: +SKIP
>>> data["patient"].shape                                        # doctest: +SKIP
"""

from .cohort import CLINICAL_TABLES, GENOMIC_TABLES, CohortData
from .local import FILENAME_TO_TABLE, find_release_files, load_release, read_table
from .synapse import (
    GENIE_ACCESS_REQUIREMENT,
    AccessDeniedError,
    AccessStatus,
    SynapseError,
    SynapseSession,
    download_release,
    resolve_token,
)

__all__ = [
    "CLINICAL_TABLES",
    "FILENAME_TO_TABLE",
    "GENIE_ACCESS_REQUIREMENT",
    "GENOMIC_TABLES",
    "AccessDeniedError",
    "AccessStatus",
    "CohortData",
    "SynapseError",
    "SynapseSession",
    "download_release",
    "find_release_files",
    "load_release",
    "read_table",
    "resolve_token",
]
