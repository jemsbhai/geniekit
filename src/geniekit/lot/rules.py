"""The configurable rules that turn a sequence of regimens into lines of therapy.

Every rule is a field of :class:`LineRules` with a documented default; the
settings used are written into the output so that a derivation can be
reproduced. :data:`PRESETS` holds named starting points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .regimens import LOCAL_ROUTES

__all__ = ["PRESETS", "LineRules", "resolve_rules"]


@dataclass(frozen=True)
class LineRules:
    """Rules for grouping regimens into lines.

    Parameters
    ----------
    combination_window_days:
        A regimen that starts within this many days of the current line's
        start joins it as part of the initial combination (drugs are often
        recorded as separate regimens when they start a few days apart).
    maintenance_same_line:
        A later regimen whose line-defining agents are all already in the
        current line continues the line (maintenance, de-escalation or
        resumption of the same drugs) instead of starting a new one.
    restart_gap_days:
        With ``maintenance_same_line``, a resumption of agents already in the
        line after a gap longer than this (days from the line's last known
        administration) starts a new line anyway. ``None`` disables the gap
        test, so a treatment holiday of any length is one line.
    progression_aware:
        When True, a regimen with new agents starts a new line only if a
        progression assessment (imaging or oncologist) falls after the
        current line started and on or before the new regimen's start; a
        switch without a documented progression (for example for toxicity)
        continues the line with the new agents added. Resumption of the same
        agents after a documented progression starts a new line.
    pre_index:
        What to do with regimens that start before the index event:
        ``"exclude"`` (they are listed with status ``pre_index`` and no line)
        or ``"line_zero"`` (they form line 0, for neoadjuvant and adjuvant
        therapy counted separately).
    pre_index_tolerance_days:
        A regimen starting up to this many days before the index is treated
        as starting at the index (dates are curated to the day, but a regimen
        and the metastasis it treats are sometimes recorded a few days apart).
    backbone_roles:
        Vocabulary roles whose agents never start or end a line; they are
        recorded as line components. The default covers androgen deprivation
        (LHRH agonists and antagonists) and leucovorin.
    excluded_roles:
        Vocabulary roles whose agents are ignored entirely; a regimen made only
        of such agents (intravesical BCG, say) is not a line.
    local_routes:
        ``drugs_admin`` values (CRC and BLADDER release the route) that make
        every agent of the regimen ``local``, hence excluded.
    investigational:
        ``"systemic"`` counts an investigational drug as a line-defining agent
        that never matches another regimen; ``"exclude"`` ignores it.
    backbone_agents, excluded_agents:
        Canonical agent names added to the backbone and excluded sets on top
        of the role-based defaults (for example the first-generation
        antiandrogens in the prostate preset).
    """

    combination_window_days: int = 28
    maintenance_same_line: bool = True
    restart_gap_days: int | None = None
    progression_aware: bool = False
    pre_index: str = "exclude"
    pre_index_tolerance_days: int = 0
    backbone_roles: tuple[str, ...] = ("backbone", "modulator")
    excluded_roles: tuple[str, ...] = ("local", "other")
    local_routes: tuple[str, ...] = LOCAL_ROUTES
    investigational: str = "systemic"
    backbone_agents: tuple[str, ...] = ()
    excluded_agents: tuple[str, ...] = ()
    name: str = field(default="default", compare=False)

    def __post_init__(self) -> None:
        if self.pre_index not in ("exclude", "line_zero"):
            raise ValueError("pre_index must be 'exclude' or 'line_zero'")
        if self.investigational not in ("systemic", "exclude"):
            raise ValueError("investigational must be 'systemic' or 'exclude'")
        if self.combination_window_days < 0 or self.pre_index_tolerance_days < 0:
            raise ValueError("windows must be non-negative")
        if self.restart_gap_days is not None and self.restart_gap_days < 0:
            raise ValueError("restart_gap_days must be non-negative or None")

    def with_(self, **changes: Any) -> LineRules:
        """A copy with some fields changed."""
        return replace(self, **changes)

    def describe(self) -> dict[str, Any]:
        """The settings as a plain dictionary (written into the derivation output)."""
        out = asdict(self)
        out["name"] = self.name
        return out


PRESETS: dict[str, LineRules] = {
    "default": LineRules(),
    # androgen deprivation (LHRH agonists and antagonists, by role) and the
    # first-generation antiandrogens are treated as a backbone that does not
    # define lines; lines are the androgen receptor pathway inhibitors,
    # chemotherapy, radiopharmaceuticals and other agents layered on it
    "prostate": LineRules(backbone_agents=("Bicalutamide", "Flutamide", "Nilutamide"), name="prostate"),
    # a documented progression is required for a switch to count as a new line
    "progression": LineRules(progression_aware=True, name="progression"),
}


def resolve_rules(rules: LineRules | str | None) -> LineRules:
    """A :class:`LineRules` from an instance, a preset name or None (default)."""
    if rules is None:
        return PRESETS["default"]
    if isinstance(rules, LineRules):
        return rules
    try:
        return PRESETS[rules]
    except KeyError:
        raise ValueError(f"unknown rules preset {rules!r}; expected one of {sorted(PRESETS)}") from None
