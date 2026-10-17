"""Line-of-therapy derivation.

:func:`derive` takes a cohort, an index event and a :class:`LineRules`, and
returns a :class:`LinesOfTherapy` with three tables (regimens with their
line assignment and the reason for it, lines, cancers) plus the settings
that produced them.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geniekit.io.cohort import CohortData

from .index import IndexSpec, index_events
from .regimens import KEYS, annotate_regimens, cancer_table, progression_events
from .rules import LineRules, resolve_rules
from .vocabulary import INVESTIGATIONAL, DrugVocabulary, load_vocabulary

__all__ = ["LinesOfTherapy", "derive", "derive_all"]

STATUS_LINE = "line"
STATUS_PRE_INDEX = "pre_index"
STATUS_BACKBONE = "backbone_only"
STATUS_EXCLUDED = "excluded"
STATUS_NO_INDEX = "no_index"
STATUS_NO_START = "no_start_date"


def _split(value: object) -> list[str]:
    return [v for v in str(value).split(";") if v] if isinstance(value, str) and value else []


_TOKEN = re.compile(r" \[regimen \d+\]$")


def _join(values: Iterable[str]) -> str:
    return ";".join(sorted(set(values)))


def _display(values: Iterable[str]) -> str:
    """Agents for output: the per-regimen investigational tokens become the plain label."""
    return ";".join(sorted({_TOKEN.sub("", v) for v in values}))


def _nan_max(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return max(vals) if vals else math.nan


@dataclass
class _Line:
    number: int
    start: float
    defining: set[str] = field(default_factory=set)
    backbone: set[str] = field(default_factory=set)
    regimen_numbers: list[int] = field(default_factory=list)
    ends: list[float] = field(default_factory=list)
    last_admins: list[float] = field(default_factory=list)
    n_investigational: int = 0

    def add(
        self,
        number: int,
        defining: set[str],
        backbone: set[str],
        end: float,
        last_admin: float,
        n_inv: int = 0,
    ) -> None:
        self.regimen_numbers.append(number)
        self.defining |= defining
        self.backbone |= backbone
        self.ends.append(end)
        self.last_admins.append(last_admin)
        self.n_investigational += n_inv

    @property
    def end(self) -> float:
        """Max end date when every member regimen has one, else NaN."""
        if any(isinstance(e, float) and math.isnan(e) for e in self.ends):
            return math.nan
        return max(self.ends) if self.ends else math.nan

    @property
    def last_known(self) -> float:
        return _nan_max([*self.ends, *self.last_admins])


def _classify_agents(regs: pd.DataFrame, rules: LineRules) -> pd.DataFrame:
    """Split each regimen's agents into line-defining, backbone and excluded sets."""
    defining_col, backbone_col, excluded_col = [], [], []
    backbone_agents = set(rules.backbone_agents)
    excluded_agents = set(rules.excluded_agents)
    for agents, roles in zip(regs["agents"], regs["roles"], strict=True):
        d, b, x = [], [], []
        for agent, role in zip(_split(agents), _split(roles), strict=True):
            if agent in excluded_agents or role in rules.excluded_roles:
                x.append(agent)
            elif agent in backbone_agents or role in rules.backbone_roles:
                b.append(agent)
            elif role == "investigational" and rules.investigational == "exclude":
                x.append(agent)
            else:
                d.append(agent)
        defining_col.append(_join(d))
        backbone_col.append(_join(b))
        excluded_col.append(_join(x))
    out = regs.copy()
    out["defining_agents"] = defining_col
    out["backbone_agents"] = backbone_col
    out["excluded_agents"] = excluded_col
    return out


def _progressed(events: np.ndarray, after: float, upto: float) -> float:
    """First progression day in (after, upto], or NaN."""
    if events.size == 0:
        return math.nan
    hit = events[(events > after) & (events <= upto)]
    return float(hit.min()) if hit.size else math.nan


def _derive_cancer(
    regs: pd.DataFrame, index_days: float, events: np.ndarray, rules: LineRules
) -> tuple[list[dict[str, Any]], list[_Line]]:
    """Assign the regimens of one cancer to lines; returns per-regimen records and the lines."""
    records: list[dict[str, Any]] = []
    lines: list[_Line] = []
    cur: _Line | None = None
    pre: _Line | None = None
    window = rules.combination_window_days
    for reg in regs.itertuples(index=False):
        rec: dict[str, Any] = {"status": "", "line_number": pd.NA, "line_role": "", "reason": ""}
        records.append(rec)
        start = float(reg.start_days) if not pd.isna(reg.start_days) else math.nan
        defining = set(_split(reg.defining_agents))
        backbone = set(_split(reg.backbone_agents))
        excluded = set(_split(reg.excluded_agents))
        end = float(reg.end_days) if not pd.isna(reg.end_days) else math.nan
        last_admin = float(reg.last_admin_days) if not pd.isna(reg.last_admin_days) else math.nan
        number = int(reg.regimen_number)
        n_inv = int(reg.n_investigational)
        if math.isnan(index_days):
            rec.update(status=STATUS_NO_INDEX, reason="the cancer has no index event")
            continue
        if math.isnan(start):
            rec.update(status=STATUS_NO_START, reason="regimen start date missing")
            continue
        if start < index_days - rules.pre_index_tolerance_days:
            if rules.pre_index == "exclude":
                rec.update(
                    status=STATUS_PRE_INDEX, reason=f"started {index_days - start:.0f} days before the index"
                )
                continue
            if pre is None:
                pre = _Line(0, start)
                lines.insert(0, pre)
            pre.add(number, defining, backbone, end, last_admin, n_inv)
            rec.update(
                status=STATUS_LINE, line_number=0, line_role="pre_index", reason="started before the index"
            )
            continue
        if not defining:
            if backbone:
                names = ", ".join(sorted(backbone))
                if cur is not None:
                    cur.backbone |= backbone
                    cur.last_admins.append(last_admin)
                    rec.update(
                        status=STATUS_BACKBONE,
                        line_number=cur.number,
                        line_role="backbone",
                        reason=f"backbone agents only ({names}); kept with line {cur.number}",
                    )
                else:
                    rec.update(
                        status=STATUS_BACKBONE, reason=f"backbone agents only ({names}) before any line"
                    )
            else:
                rec.update(
                    status=STATUS_EXCLUDED, reason=f"no line-defining agent ({', '.join(sorted(excluded))})"
                )
            continue
        if cur is None:
            cur = _Line(len([ln for ln in lines if ln.number > 0]) + 1, start)
            lines.append(cur)
            cur.add(number, defining, backbone, end, last_admin, n_inv)
            rec.update(
                status=STATUS_LINE,
                line_number=cur.number,
                line_role="initial",
                reason="first line-defining regimen on or after the index",
            )
            continue
        if start <= cur.start + window:
            cur.add(number, defining, backbone, end, last_admin, n_inv)
            rec.update(
                status=STATUS_LINE,
                line_number=cur.number,
                line_role="combination",
                reason=f"started within {window} days of line {cur.number} start",
            )
            continue
        prog_day = _progressed(events, cur.start, start) if rules.progression_aware else math.nan
        new_agents = defining - cur.defining
        if not new_agents:
            gap = start - cur.last_known if not math.isnan(cur.last_known) else math.nan
            if rules.progression_aware and not math.isnan(prog_day):
                reason = f"same agents resumed after progression on day {prog_day:.0f}"
            elif rules.restart_gap_days is not None and not math.isnan(gap) and gap > rules.restart_gap_days:
                reason = f"same agents resumed after a gap of {gap:.0f} days (limit {rules.restart_gap_days})"
            elif not rules.maintenance_same_line:
                reason = "new regimen (maintenance_same_line is off)"
            else:
                cur.add(number, defining, backbone, end, last_admin, n_inv)
                role = "maintenance" if defining < cur.defining else "continuation"
                rec.update(
                    status=STATUS_LINE,
                    line_number=cur.number,
                    line_role=role,
                    reason=f"agents already in line {cur.number} ({', '.join(sorted(defining))})",
                )
                continue
        else:
            named = ", ".join(sorted(new_agents))
            if rules.progression_aware and math.isnan(prog_day):
                cur.add(number, defining, backbone, end, last_admin, n_inv)
                rec.update(
                    status=STATUS_LINE,
                    line_number=cur.number,
                    line_role="switch_without_progression",
                    reason=f"new agent(s) {named}, no documented progression since line {cur.number} started",
                )
                continue
            reason = f"new agent(s): {named}"
            if rules.progression_aware:
                reason += f" after progression on day {prog_day:.0f}"
        cur = _Line(cur.number + 1, start)
        lines.append(cur)
        cur.add(number, defining, backbone, end, last_admin, n_inv)
        rec.update(status=STATUS_LINE, line_number=cur.number, line_role="initial", reason=reason)
    return records, lines


@dataclass
class LinesOfTherapy:
    """The result of :func:`derive`.

    ``regimens`` is the annotated regimen table of the selected cancers with
    ``status``, ``line_number``, ``line_role`` and ``reason``; ``lines`` has
    one row per line; ``cancers`` one row per selected cancer with its index
    event and line counts; ``events`` the progression assessments used.
    """

    cohort: str
    index_event: str
    rules: LineRules
    regimens: pd.DataFrame
    lines: pd.DataFrame
    cancers: pd.DataFrame
    events: pd.DataFrame
    unknown_labels: list[str] = field(default_factory=list)

    def settings(self) -> dict[str, Any]:
        return {"cohort": self.cohort, "index_event": self.index_event, "rules": self.rules.describe()}

    def summary(self) -> pd.DataFrame:
        """One row per cohort: cancers, cancers with an index, lines per cancer, first-line regimens."""
        rows = []
        for cohort, c in self.cancers.groupby("cohort"):
            with_index = c[c["index_days"].notna()]
            treated = with_index[with_index["n_lines"] > 0]
            first = self.lines[(self.lines["cohort"] == cohort) & (self.lines["line_number"] == 1)]
            top = first["agents"].value_counts().head(3)
            rows.append(
                {
                    "cohort": cohort,
                    "index_event": self.index_event,
                    "cancers": len(c),
                    "with_index": len(with_index),
                    "with_lines": len(treated),
                    "lines_per_treated_cancer": round(float(treated["n_lines"].mean()), 2)
                    if len(treated)
                    else np.nan,
                    "max_lines": int(with_index["n_lines"].max()) if len(with_index) else 0,
                    "top_first_line": "; ".join(f"{k} ({v})" for k, v in top.items()),
                }
            )
        return pd.DataFrame(rows)

    def to_dir(self, path: str | Path, compress: bool = False) -> Path:
        """Write the three tables and the settings into a folder."""
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        suffix = ".csv.gz" if compress else ".csv"
        for name in ("regimens", "lines", "cancers", "events"):
            getattr(self, name).to_csv(folder / f"lot_{name}{suffix}", index=False)
        (folder / "lot_settings.json").write_text(
            json.dumps(self.settings(), indent=2) + "\n", encoding="utf-8"
        )
        return folder

    def __repr__(self) -> str:
        n_c = int(self.cancers["index_days"].notna().sum()) if len(self.cancers) else 0
        return (
            f"LinesOfTherapy({self.cohort}, index={self.index_event!r}, rules={self.rules.name!r}, "
            f"{n_c} cancers with an index, {len(self.lines)} lines, {len(self.regimens)} regimens)"
        )


def derive(
    data: CohortData,
    index: IndexSpec = "advanced",
    rules: LineRules | str | None = None,
    cancers: str = "index",
    vocabulary: DrugVocabulary | None = None,
    progression_sources: tuple[str, ...] = ("imaging", "med_onc"),
    after_days: float = 0.0,
) -> LinesOfTherapy:
    """Derive lines of therapy for the selected cancers of one cohort.

    Parameters
    ----------
    data:
        A loaded (or harmonised) cohort with at least the ``regimen`` table and
        the cancer table(s); imaging and oncologist-note tables are used for
        progression when the index or the rules need them.
    index:
        The index event: ``"diagnosis"``, ``"advanced"`` (default),
        ``"first_progression"``, ``"first_systemic"``, a custom table or a
        callable (see :mod:`geniekit.lot.index`).
    rules:
        A :class:`LineRules`, a preset name (``"default"``, ``"prostate"``,
        ``"progression"``) or None.
    cancers:
        ``"index"`` (default), ``"non_index"`` or ``"all"``.
    """
    rules = resolve_rules(rules)
    vocab = vocabulary or load_vocabulary()
    cancer_rows = cancer_table(data, cancers)
    regs = _classify_agents(annotate_regimens(data, vocab, rules.local_routes), rules)
    defining_mask = regs["defining_agents"] != ""
    idx = index_events(
        data,
        index,
        cancer_rows,
        regimens=regs,
        defining=defining_mask,
        after_days=after_days,
        progression_sources=progression_sources,
    )
    index_name = idx["index_event"].iloc[0] if len(idx) else (index if isinstance(index, str) else "custom")
    need_events = rules.progression_aware or index_name == "first_progression"
    events = (
        progression_events(data, cancer_rows, sources=progression_sources)
        if need_events
        else pd.DataFrame(columns=[*KEYS, "days", "source", "level"])
    )
    keyed = idx[[*KEYS, "index_days", "index_source"]]
    regs = regs.merge(keyed, on=KEYS, how="inner")
    regs = regs.sort_values([*KEYS, "start_days", "regimen_number"], kind="stable").reset_index(drop=True)
    event_days: dict[tuple[str, int], np.ndarray] = {}
    if len(events):
        for key, grp in events.groupby(KEYS):
            event_days[(str(key[0]), int(key[1]))] = grp["days"].to_numpy(dtype=float)
    # per-cancer derivation
    rec_frames: list[pd.DataFrame] = []
    line_rows: list[dict[str, Any]] = []
    cancer_stats: dict[tuple[str, int], dict[str, Any]] = {}
    for key, grp in regs.groupby(KEYS, sort=False):
        k = (str(key[0]), int(key[1]))
        index_days = float(grp["index_days"].iloc[0]) if not pd.isna(grp["index_days"].iloc[0]) else math.nan
        records, lines = _derive_cancer(grp, index_days, event_days.get(k, np.empty(0)), rules)
        rec = pd.DataFrame(records, index=grp.index)
        rec_frames.append(rec)
        numbered = [ln for ln in lines if ln.number > 0]
        ev = event_days.get(k, np.empty(0))
        for i, ln in enumerate(lines):
            nxt = next((m for m in lines[i + 1 :] if m.number > ln.number), None)
            next_start = nxt.start if nxt is not None else math.nan
            end = ln.end
            if nxt is not None:
                end_reason = "next line started"
            elif not math.isnan(end):
                end_reason = "discontinued, no further line"
            else:
                end_reason = "ongoing or end unknown"
            last_known = ln.last_known
            prog = (
                _progressed(ev, ln.start, next_start)
                if (len(ev) and not math.isnan(next_start))
                else math.nan
            )
            line_rows.append(
                {
                    "cohort": data.cohort,
                    "record_id": k[0],
                    "ca_seq": k[1],
                    "line_number": ln.number,
                    "start_days": ln.start,
                    "days_from_index": ln.start - index_days,
                    "end_days": end,
                    "last_known_days": last_known,
                    "duration_days": (last_known - ln.start) if not math.isnan(last_known) else math.nan,
                    "next_line_start_days": next_start,
                    "time_to_next_line_days": (next_start - ln.start)
                    if not math.isnan(next_start)
                    else math.nan,
                    "end_reason": end_reason,
                    "agents": _display(ln.defining),
                    "backbone_agents": _display(ln.backbone),
                    "n_agents": len(ln.defining),
                    "n_investigational": ln.n_investigational,
                    "n_regimens": len(ln.regimen_numbers),
                    "regimen_numbers": ",".join(str(n) for n in ln.regimen_numbers),
                    "investigational": any(a.startswith(INVESTIGATIONAL) for a in ln.defining),
                    "progression_before_next_days": prog,
                }
            )
        first = numbered[0] if numbered else None
        cancer_stats[k] = {
            "n_regimens": len(grp),
            "n_pre_index": int(
                (rec["status"] == STATUS_PRE_INDEX).sum() + (rec["line_role"] == "pre_index").sum()
            ),
            "n_lines": len(numbered),
            "first_line_start_days": first.start if first else math.nan,
            "first_line_days_from_index": (first.start - index_days) if first else math.nan,
            "first_line_agents": _display(first.defining) if first else "",
            "unknown_labels": _join(a for v in grp["unknown_labels"] for a in _split(v)),
        }
    regs_out = regs.copy()
    if rec_frames:
        recs = pd.concat(rec_frames).reindex(regs.index)
        for col in ("status", "line_number", "line_role", "reason"):
            regs_out[col] = recs[col].to_numpy()
    else:
        for col in ("status", "line_number", "line_role", "reason"):
            regs_out[col] = pd.Series(dtype=object)
    regs_out["line_number"] = pd.array(
        regs_out["line_number"].to_numpy() if len(regs_out) else [], dtype="Int64"
    )
    for col in ("agents", "defining_agents", "backbone_agents", "excluded_agents"):
        regs_out[col] = [_display(_split(v)) for v in regs_out[col]]
    regs_out["days_from_index"] = regs_out["start_days"] - regs_out["index_days"]
    line_cols = [
        "cohort",
        "record_id",
        "ca_seq",
        "line_number",
        "start_days",
        "days_from_index",
        "end_days",
        "last_known_days",
        "duration_days",
        "next_line_start_days",
        "time_to_next_line_days",
        "end_reason",
        "agents",
        "backbone_agents",
        "n_agents",
        "n_investigational",
        "n_regimens",
        "regimen_numbers",
        "investigational",
        "progression_before_next_days",
    ]
    lines_out = pd.DataFrame(line_rows, columns=line_cols)
    if len(lines_out):
        lines_out = lines_out.sort_values([*KEYS, "line_number"]).reset_index(drop=True)
        lines_out["ca_seq"] = lines_out["ca_seq"].astype("Int64")
    # cancers
    cancers_out = idx.copy()
    cancers_out.insert(0, "cohort", data.cohort)
    stat_cols = [
        "n_regimens",
        "n_pre_index",
        "n_lines",
        "first_line_start_days",
        "first_line_days_from_index",
        "first_line_agents",
        "unknown_labels",
    ]
    stats = pd.DataFrame(
        [{"record_id": k[0], "ca_seq": k[1], **v} for k, v in cancer_stats.items()],
        columns=[*KEYS, *stat_cols],
    )
    if len(stats):
        stats["ca_seq"] = stats["ca_seq"].astype("Int64")
    cancers_out = cancers_out.merge(stats, on=KEYS, how="left")
    cancers_out["n_regimens"] = cancers_out["n_regimens"].fillna(0).astype(int)
    cancers_out["n_pre_index"] = cancers_out["n_pre_index"].fillna(0).astype(int)
    cancers_out["n_lines"] = cancers_out["n_lines"].fillna(0).astype(int)
    cancers_out["first_line_agents"] = cancers_out["first_line_agents"].fillna("")
    cancers_out["unknown_labels"] = cancers_out["unknown_labels"].fillna("")
    for col in ("tt_os_dx_days", "os_dx_status"):
        if col in cancer_rows.columns:
            extra = cancer_rows[[*KEYS, col]].copy()
            extra["record_id"] = extra["record_id"].astype(str)
            cancers_out = cancers_out.merge(extra, on=KEYS, how="left")
    if "tt_os_dx_days" in cancers_out.columns:
        cancers_out = cancers_out.rename(columns={"tt_os_dx_days": "follow_up_days"})
        cancers_out["follow_up_days"] = pd.to_numeric(cancers_out["follow_up_days"], errors="coerce")
    if "os_dx_status" in cancers_out.columns:
        cancers_out["died"] = pd.to_numeric(cancers_out["os_dx_status"], errors="coerce").eq(1)
        cancers_out = cancers_out.drop(columns=["os_dx_status"])
    unknown = sorted({a for v in regs_out["unknown_labels"] for a in _split(v)})
    return LinesOfTherapy(
        cohort=data.cohort,
        index_event=str(index_name),
        rules=rules,
        regimens=regs_out,
        lines=lines_out,
        cancers=cancers_out,
        events=events,
        unknown_labels=unknown,
    )


def derive_all(items: Iterable[CohortData], **kwargs: Any) -> LinesOfTherapy:
    """:func:`derive` for several cohorts, with the tables concatenated."""
    results = [derive(item, **kwargs) for item in items]
    if not results:
        raise ValueError("no cohorts given")
    first = results[0]
    cat = lambda name: pd.concat([getattr(r, name) for r in results], ignore_index=True)  # noqa: E731
    return LinesOfTherapy(
        cohort=",".join(r.cohort for r in results),
        index_event=first.index_event,
        rules=first.rules,
        regimens=cat("regimens"),
        lines=cat("lines"),
        cancers=cat("cancers"),
        events=cat("events"),
        unknown_labels=sorted({a for r in results for a in r.unknown_labels}),
    )
