"""Drug vocabulary: released drug labels to canonical agents, classes and roles.

The regimen tables name each drug as ``"<label>(<synonyms>)"``. The label
differs between the two schema generations in salt-form spelling and in one
misspelling, so geniekit maps every released label to a canonical agent name
through the packaged ``resources/drug_vocabulary.csv`` (built and documented
by ``data-raw/build_drug_vocabulary.py``). A label that is not in the
vocabulary is passed through unchanged with ``drug_class="unclassified"`` and
the default role, and is reported by the line-of-therapy derivation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources as importlib_resources

import pandas as pd

__all__ = ["ROLES", "DrugEntry", "DrugVocabulary", "base_label", "load_vocabulary", "synonyms"]

ROLES: tuple[str, ...] = ("systemic", "backbone", "modulator", "local", "investigational", "other")
INVESTIGATIONAL = "Investigational Drug"


def base_label(value: object) -> str:
    """The label before the parenthesised synonym list, whitespace-normalised."""
    return re.sub(r"\s+", " ", str(value).split("(")[0]).strip()


def synonyms(value: object) -> list[str]:
    """The synonyms listed in parentheses after the label (may be empty)."""
    text = str(value)
    if "(" not in text:
        return []
    inner = text[text.index("(") + 1 :]
    inner = inner[: inner.rindex(")")] if ")" in inner else inner
    return [s.strip() for s in inner.split(",") if s.strip()]


@dataclass(frozen=True)
class DrugEntry:
    """One vocabulary row."""

    label: str
    agents: tuple[str, ...]
    drug_class: str
    drug_subclass: str
    lot_role: str
    note: str = ""
    known: bool = True

    @property
    def agent(self) -> str:
        """The agents joined with ``;`` (one name for almost every label)."""
        return ";".join(self.agents)


class DrugVocabulary:
    """Lookup from released drug label (with or without synonyms) to :class:`DrugEntry`."""

    def __init__(self, frame: pd.DataFrame) -> None:
        required = {"label", "agents", "drug_class", "drug_subclass", "lot_role"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"vocabulary lacks columns: {sorted(missing)}")
        self._frame = frame.reset_index(drop=True)
        self._entries: dict[str, DrugEntry] = {}
        for row in self._frame.itertuples(index=False):
            entry = DrugEntry(
                label=row.label,
                agents=tuple(a.strip() for a in str(row.agents).split(";") if a.strip()),
                drug_class=row.drug_class,
                drug_subclass=row.drug_subclass,
                lot_role=row.lot_role,
                note=str(getattr(row, "note", "") or ""),
            )
            self._entries[_key(row.label)] = entry

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, value: object) -> bool:
        return _key(base_label(value)) in self._entries

    def lookup(self, value: object) -> DrugEntry | None:
        """The entry for a released drug value, or None when it is not in the vocabulary."""
        return self._entries.get(_key(base_label(value)))

    def classify(self, value: object, default_role: str = "systemic") -> DrugEntry:
        """Like :meth:`lookup`, but an unknown label is passed through as an unclassified agent."""
        entry = self.lookup(value)
        if entry is not None:
            return entry
        label = base_label(value)
        return DrugEntry(
            label, (label,), "unclassified", "", default_role, "not in the packaged vocabulary", False
        )

    def agents(self, role: str | None = None, drug_class: str | None = None) -> list[str]:
        """Distinct canonical agents, optionally filtered by role or class."""
        out: set[str] = set()
        for e in self._entries.values():
            if (role is None or e.lot_role == role) and (drug_class is None or e.drug_class == drug_class):
                out.update(e.agents)
        return sorted(out)

    def __repr__(self) -> str:
        return f"DrugVocabulary({len(self)} labels, {len(self.agents())} agents)"


def _key(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().lower()


@lru_cache(maxsize=1)
def load_vocabulary() -> DrugVocabulary:
    """The packaged vocabulary (cached)."""
    path = importlib_resources.files("geniekit.resources").joinpath("drug_vocabulary.csv")
    with path.open("r", encoding="utf-8") as fh:
        frame = pd.read_csv(fh, dtype=str, keep_default_na=False)
    return DrugVocabulary(frame)
