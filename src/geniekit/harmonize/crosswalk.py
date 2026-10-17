"""Load and validate the harmonisation crosswalk (``resources/crosswalk.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import yaml

from .. import dictionary
from ..releases import COHORTS, normalize_cohort

__all__ = ["RULE_KINDS", "Crosswalk", "Rule", "load_crosswalk"]

RULE_KINDS: tuple[str, ...] = ("rename", "recode", "collapse", "coerce", "derive", "note")

#: crosswalk table name -> logical tables it applies to
TABLE_TARGETS: dict[str, tuple[str, ...]] = {
    "patient": ("patient",),
    "cancer": ("cancer_index", "cancer_non_index"),
    "regimen": ("regimen",),
    "imaging": ("imaging",),
    "pathology": ("pathology",),
    "med_onc": ("med_onc",),
    "tumor_marker": ("tumor_marker",),
    "radiation": ("radiation",),
    "cpt": ("cpt",),
}


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    table: str
    cohorts: tuple[str, ...]
    basis: str = ""
    caveat: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def variables(self) -> tuple[str, ...]:
        """Source variables the rule touches (for validation and reporting)."""
        p = self.params
        if self.kind == "rename":
            return (p["from"],)
        if self.kind in ("recode", "collapse", "coerce"):
            if "variables" in p:
                return tuple(p["variables"])
            return (p["variable"],)
        if self.kind == "note":
            return tuple(p.get("variables", []))
        return ()

    def applies_to(self, cohort: str, table: str) -> bool:
        if normalize_cohort(cohort) not in self.cohorts:
            return False
        return self.table == "*" or table in TABLE_TARGETS.get(self.table, (self.table,))


@dataclass(frozen=True)
class Crosswalk:
    version: int
    generated: str
    generations: dict[int, tuple[str, ...]]
    rules: tuple[Rule, ...]
    source: str = "geniekit.resources/crosswalk.yaml"

    def rules_for(
        self, cohort: str, table: str | None = None, kinds: tuple[str, ...] | None = None
    ) -> list[Rule]:
        c = normalize_cohort(cohort)
        out = []
        for r in self.rules:
            if c not in r.cohorts:
                continue
            if table is not None and not r.applies_to(c, table):
                continue
            if kinds is not None and r.kind not in kinds:
                continue
            out.append(r)
        return out

    def generation(self, cohort: str) -> int:
        c = normalize_cohort(cohort)
        for g, cohorts in self.generations.items():
            if c in cohorts:
                return g
        raise KeyError(f"{c} is not assigned to a generation in the crosswalk")

    def validate(self, dd: dictionary.DataDictionary | None = None) -> list[str]:
        """Check every rule against the packaged dictionary; return problem strings (empty when clean)."""
        dd = dd or dictionary.load()
        problems: list[str] = []
        ids = [r.id for r in self.rules]
        for dup in {i for i in ids if ids.count(i) > 1}:
            problems.append(f"duplicate rule id {dup!r}")
        # names introduced by earlier rules, per (cohort, table): new name -> original name
        renamed: dict[tuple[str, str], dict[str, str]] = {}
        for r in self.rules:
            if r.kind not in RULE_KINDS:
                problems.append(f"{r.id}: unknown kind {r.kind!r}")
                continue
            if r.table != "*" and r.table not in TABLE_TARGETS:
                problems.append(f"{r.id}: unknown table {r.table!r}")
                continue
            if r.kind == "note" or r.table == "*":
                continue
            dict_table = r.table
            for cohort in r.cohorts:
                declared = set(dd.variables(cohort, dict_table)["variable"])
                aliases = renamed.setdefault((cohort, dict_table), {})
                for var in r.variables:
                    if var not in declared and var not in aliases:
                        problems.append(f"{r.id}: {cohort} does not declare {dict_table}.{var}")
                if r.kind in ("recode", "collapse"):
                    for var in r.variables:
                        original = aliases.get(var, var)
                        levels = {v for v in dd.values(original, dict_table).get(cohort, [])}
                        if not levels:
                            continue
                        unknown = [k for k in r.params["map"] if k not in levels]
                        if unknown and r.kind == "recode":
                            problems.append(f"{r.id}: {cohort} {var} has no level(s) {unknown[:3]}")
                if r.kind == "rename":
                    target = r.params["to"]
                    aliases[target] = r.params["from"]
                    for sib in r.params.get("siblings", []) or []:
                        for suffix in ("_days", "_mos", "_yrs"):
                            aliases[f"{sib['to']}{suffix}"] = f"{sib['from']}{suffix}"
                    canonical = {c for c in COHORTS if c not in r.cohorts}
                    if not any(target in set(dd.variables(c, dict_table)["variable"]) for c in canonical):
                        problems.append(
                            f"{r.id}: target {dict_table}.{target} is not declared by any other cohort"
                        )
        return problems

    def __repr__(self) -> str:
        kinds = {}
        for r in self.rules:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        head = f"Crosswalk(version={self.version}, generated={self.generated!r}, "
        return head + f"rules={len(self.rules)} {kinds})"

    def to_frame(self):
        import pandas as pd

        rows = [
            {
                "id": r.id,
                "kind": r.kind,
                "table": r.table,
                "cohorts": ",".join(r.cohorts),
                "variables": ",".join(r.variables),
                "target": r.params.get("to") or ",".join(r.params.get("outputs", [])),
                "caveat": r.caveat or "",
                "basis": " ".join(r.basis.split()),
            }
            for r in self.rules
        ]
        return pd.DataFrame(rows)


def _rule_from_raw(raw: dict[str, Any]) -> Rule:
    reserved = {"id", "kind", "table", "cohorts", "basis", "caveat"}
    cohorts = tuple(normalize_cohort(c) for c in raw["cohorts"])
    params = {k: v for k, v in raw.items() if k not in reserved}
    return Rule(
        id=str(raw["id"]),
        kind=str(raw["kind"]),
        table=str(raw["table"]),
        cohorts=cohorts,
        basis=str(raw.get("basis", "")).strip(),
        caveat=raw.get("caveat"),
        params=params,
    )


def load_crosswalk(path: str | Path | None = None) -> Crosswalk:
    """Load the packaged crosswalk, or a custom YAML file with the same layout."""
    if path is None:
        return _packaged()
    text = Path(path).read_text(encoding="utf-8")
    return _parse(text, source=str(path))


def _parse(text: str, source: str) -> Crosswalk:
    raw = yaml.safe_load(text)
    generations = {int(k): tuple(normalize_cohort(c) for c in v) for k, v in raw["generations"].items()}
    rules = tuple(_rule_from_raw(r) for r in raw["rules"])
    return Crosswalk(
        version=int(raw["version"]),
        generated=str(raw.get("generated", "")),
        generations=generations,
        rules=rules,
        source=source,
    )


@lru_cache(maxsize=1)
def _packaged() -> Crosswalk:
    text = (
        importlib_resources.files("geniekit.resources").joinpath("crosswalk.yaml").read_text(encoding="utf-8")
    )
    return _parse(text, source="geniekit.resources/crosswalk.yaml")
