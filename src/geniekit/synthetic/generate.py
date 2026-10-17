"""Dictionary-driven synthetic GENIE BPC cohorts.

:func:`generate` builds a :class:`~geniekit.io.CohortData` for any of the six
public cohorts without touching real data. Two layers are combined:

1. **Schema layer.** Every clinical table carries exactly the columns the
   cohort's Variable Synopsis declares (see :mod:`geniekit.dictionary`), typed
   and filled from the declared permitted values. Cohort-specific naming is
   therefore reproduced faithfully: NSCLC gets ``ca_lung_cigarette`` and
   ``dmets_liver``, BLADDER gets ``ca_cigarette`` and ``dist_mets_liver``,
   tumor-marker and radiation tables exist only where the real release has
   them, and so on.
2. **Coherence layer.** The columns that analyses depend on (identifiers,
   diagnosis age and stage, distant metastasis, survival and progression
   fields, regimens with start and end intervals, imaging and oncologist
   assessments, sequencing samples, genomic calls) are overwritten with a
   simulated history per patient so that joins, time intervals and outcomes
   are mutually consistent.

The simulation is arbitrary by design. Event rates, drug menus and gene
frequencies are round numbers chosen to exercise code paths; they are not
estimates of the real cohorts and must never be reported as such. Every frame
is marked ``synthetic=True`` on the returned :class:`CohortData`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import dictionary
from ..io.cohort import CLINICAL_TABLES, CohortData
from ..releases import COHORTS, get_release, normalize_cohort
from .spec import COHORT_SPECS, CohortSpec

__all__ = ["generate", "generate_all"]

MONTH_DAYS = 30.4
YEAR_DAYS = 365.25

_CODE_RE = re.compile(r"^\s*([^\s=]{1,12})\s*=\s*\S")
_RANGE_RE = re.compile(r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*$")

# canonical metastasis site -> (generation 1 suffix, generation 2 suffix)
_MET_SITES: dict[str, tuple[str, str]] = {
    "liver": ("liver", "liver"),
    "bone": ("bone", "bone"),
    "brain": ("brain", "brain_cns"),
    "lung": ("thorax", "pulmonary"),
    "peritoneum": ("abdomen", "peritoneum_and_malignant_peritoneal_effusion"),
    "adrenal": ("abdomen", "adrenal"),
    "lymph_nodes": ("pelvis", "lymph_nodes"),
}

# per cohort: probability of each site given distant metastasis (arbitrary)
_SITE_WEIGHTS: dict[str, dict[str, float]] = {
    "NSCLC": {"bone": 0.4, "brain": 0.35, "liver": 0.25, "lung": 0.3, "adrenal": 0.2, "lymph_nodes": 0.3},
    "CRC": {"liver": 0.6, "lung": 0.35, "peritoneum": 0.25, "bone": 0.08, "brain": 0.03, "lymph_nodes": 0.3},
    "BrCa": {"bone": 0.55, "liver": 0.3, "lung": 0.3, "brain": 0.15, "lymph_nodes": 0.35},
    "PANC": {"liver": 0.65, "peritoneum": 0.35, "lung": 0.25, "bone": 0.05, "lymph_nodes": 0.3},
    "Prostate": {"bone": 0.75, "lymph_nodes": 0.45, "liver": 0.15, "lung": 0.12, "brain": 0.02},
    "BLADDER": {
        "lymph_nodes": 0.5,
        "bone": 0.3,
        "lung": 0.35,
        "liver": 0.3,
        "brain": 0.05,
        "peritoneum": 0.1,
    },
}

_CHROMOSOMES: dict[str, str] = {
    "TP53": "17",
    "KRAS": "12",
    "EGFR": "7",
    "STK11": "19",
    "KEAP1": "19",
    "CDKN2A": "9",
    "RB1": "13",
    "BRAF": "7",
    "MET": "7",
    "APC": "5",
    "PIK3CA": "3",
    "SMAD4": "18",
    "FBXW7": "4",
    "NRAS": "1",
    "SOX9": "17",
    "GATA3": "10",
    "CDH1": "16",
    "MAP3K1": "5",
    "ESR1": "6",
    "AKT1": "14",
    "PTEN": "10",
    "ARID1A": "1",
    "BRCA2": "13",
    "GNAS": "20",
    "AR": "X",
    "SPOP": "17",
    "FOXA1": "14",
    "ATM": "11",
    "CDK12": "17",
    "KDM6A": "X",
    "KMT2D": "12",
    "FGFR3": "4",
    "ERBB2": "17",
    "MYC": "8",
    "CCND1": "11",
    "ALK": "2",
    "ROS1": "6",
    "RET": "10",
    "NTRK1": "1",
    "ERG": "21",
}
_AMINO = "ACDEFGHIKLMNPQRSTVWY"
_BASES = "ACGT"

_FILLER_GENES = (
    "NF1",
    "ARID2",
    "SETD2",
    "KMT2C",
    "NOTCH1",
    "FAT1",
    "ATRX",
    "CREBBP",
    "EP300",
    "BRCA1",
    "PALB2",
    "CTNNB1",
    "FGFR1",
    "FGFR2",
    "IDH1",
    "JAK2",
    "MTOR",
    "NFE2L2",
    "PBRM1",
    "POLE",
    "TSC1",
    "TSC2",
    "VHL",
)


@dataclass
class _Patient:
    record_id: str
    institution: str
    birth_year: int
    sex: str
    age_dx: int
    dob_dx: int  # days from birth to index diagnosis
    stage: str
    advanced_at: float | None  # days from dx when distant disease exists (0 for stage IV)
    met_sites: list[str]
    follow_up: float  # days from dx to last known alive
    death: float | None  # days from dx to death (None if alive at last follow up)
    progression: float | None  # days from dx to first progression after advanced disease
    adjuvant: bool
    index_ca_seq: int
    prior_cancer: bool
    regimens: list[dict] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)

    @property
    def end(self) -> float:
        return self.death if self.death is not None else self.follow_up


class _Schema:
    """Column specification for one cohort, derived from the dictionary."""

    def __init__(self, cohort: str) -> None:
        self.cohort = cohort
        dd = dictionary.load()
        self.generation = dd.schema_generation(cohort)
        self._rows = dd.variables(cohort)

    def columns(self, table: str) -> pd.DataFrame:
        dict_table = "cancer" if table in ("cancer_index", "cancer_non_index") else table
        rows = self._rows[self._rows["table"] == dict_table]
        return rows.sort_values("order")

    def has_table(self, table: str) -> bool:
        return not self.columns(table).empty

    def levels(self, table: str, variable: str) -> list[str]:
        rows = self.columns(table)
        hit = rows[rows["variable"] == variable]
        return list(hit["values"].iloc[0]) if len(hit) else []

    def dtype(self, table: str, variable: str) -> str:
        rows = self.columns(table)
        hit = rows[rows["variable"] == variable]
        return str(hit["dtype"].iloc[0]) if len(hit) else ""


def _code(level: str) -> str:
    m = _CODE_RE.match(level)
    return m.group(1) if m else level.strip()


_PSEUDO_LEVELS = ("free-text", "icd-o-3", "genie-[institution]", "yyyy", "xxxxxx")


def _parse_levels(levels: Iterable[str], numeric: bool) -> list:
    out: list = []
    for raw in levels:
        if any(marker in raw.lower() for marker in _PSEUDO_LEVELS):
            continue
        code = _code(raw)
        if numeric:
            try:
                out.append(float(code) if "." in code else int(code))
            except ValueError:
                continue
        else:
            out.append(code)
    return out


def _pick_level(levels: list[str], *needles: str, default: str | None = None) -> str | None:
    """First declared level whose text contains all ``needles`` (case-insensitive)."""
    for raw in levels:
        text = raw.lower()
        if all(n.lower() in text for n in needles):
            return _code(raw)
    return default


class _Generator:
    def __init__(self, cohort: str, n_patients: int, seed: int) -> None:
        self.cohort = cohort
        self.spec: CohortSpec = COHORT_SPECS[cohort]
        self.n = n_patients
        self.rng = np.random.default_rng(seed)
        self.schema = _Schema(cohort)
        self.release = get_release(cohort)
        self.version = self.release.version
        self.gen = self.schema.generation
        self.patients: list[_Patient] = []
        self._id_counter = 0

    # ------------------------------------------------------------------ utils
    def _days_to_derived(self, frame: pd.DataFrame, stem: str, days: pd.Series) -> None:
        """Fill ``<stem>_days/_mos/_yrs`` (or ``_int``-style) columns that exist."""
        days = pd.Series(np.round(np.asarray(days, dtype=float)))
        for suffix, divisor in (("days", 1.0), ("mos", MONTH_DAYS), ("yrs", YEAR_DAYS)):
            col = f"{stem}_{suffix}"
            if col in frame.columns:
                frame[col] = (days / divisor).to_numpy() if divisor != 1.0 else days.to_numpy()

    def _set(self, frame: pd.DataFrame, column: str, values) -> None:
        if column in frame.columns:
            frame[column] = values

    def _base_frame(self, table: str, n: int) -> pd.DataFrame:
        """Generic fill of every declared column (schema layer)."""
        cols = self.schema.columns(table)
        data: dict[str, object] = {}
        for _, row in cols.iterrows():
            var, dtype, levels = row["variable"], row["dtype"], list(row["values"])
            numeric = dtype.lower().startswith("num")
            data[var] = self._generic_values(var, numeric, levels, n)
        frame = pd.DataFrame(data)
        self._set(frame, "cohort", self.cohort)
        self._set(frame, "release_version", self.version.lstrip("v"))
        return frame

    def _generic_values(self, var: str, numeric: bool, levels: list[str], n: int):
        rng = self.rng
        present = rng.random(n) < 0.7
        if levels:
            if len(levels) == 1 and _RANGE_RE.match(levels[0]):
                lo, hi = (int(x) for x in _RANGE_RE.match(levels[0]).groups())
                vals = rng.integers(lo, hi + 1, size=n).astype(float)
                return np.where(present, vals, np.nan)
            if levels == ["YYYY"]:
                vals = rng.integers(2010, 2020, size=n)
                return vals if numeric else vals.astype(str).astype(object)
            parsed = _parse_levels(levels, numeric)
            if parsed:
                picks = rng.choice(np.array(parsed, dtype=object), size=n)
                if numeric:
                    return np.where(present, picks.astype(float), np.nan)
                out = np.where(present, picks, None)
                return pd.array(out, dtype=object)
        if numeric:
            if var.endswith("_days") or var.endswith("_int") or "_int_" in var:
                vals = rng.integers(0, 2000, size=n).astype(float)
            elif var.endswith("_mos"):
                vals = rng.integers(0, 2000, size=n) / MONTH_DAYS
            elif var.endswith("_yrs"):
                vals = rng.integers(0, 2000, size=n) / YEAR_DAYS
            elif var.startswith("n_") or var.endswith("_num") or var.endswith("_number"):
                vals = rng.integers(0, 5, size=n).astype(float)
            elif var.startswith("age_"):
                vals = rng.integers(30, 90, size=n).astype(float)
            elif var == "birth_year":
                vals = rng.integers(1930, 1990, size=n).astype(float)
            else:
                return np.full(n, np.nan)
            return np.where(present, vals, np.nan)
        return pd.array([None] * n, dtype=object)

    # --------------------------------------------------------------- patients
    def _new_record_id(self, institution: str) -> str:
        self._id_counter += 1
        return f"GENIE-{institution}-{self._id_counter:06d}"

    def _simulate_patients(self) -> None:
        rng, spec = self.rng, self.spec
        institutions = list(self.release.institutions)
        stages = list(spec.stage_weights)
        weights = np.array(list(spec.stage_weights.values()))
        weights = weights / weights.sum()
        for _ in range(self.n):
            inst = str(rng.choice(institutions))
            rid = self._new_record_id(inst)
            sex = "Female" if rng.random() < spec.sex_female_fraction else "Male"
            # the public BrCa cohort is restricted to early-onset disease (age 18 to 56 at sequencing)
            age_range = (25, 56) if self.cohort == "BrCa" else (40, 86)
            age_dx = int(rng.integers(*age_range))
            birth_year = int(rng.integers(1930, 1980))
            dob_dx = round(age_dx * YEAR_DAYS + float(rng.integers(0, 365)))
            stage = str(rng.choice(stages, p=weights))
            prior = rng.random() < 0.15
            index_seq = 1 if prior else 0
            follow_up = float(rng.uniform(300, 2600))
            if stage == "Stage IV":
                advanced_at: float | None = 0.0
                adjuvant = False
            else:
                adjuvant = rng.random() < 0.5
                if rng.random() < 0.35:
                    advanced_at = float(rng.exponential(600) + 60)
                    if advanced_at > follow_up:
                        advanced_at = None
                else:
                    advanced_at = None
            # death: shorter survival once advanced
            if advanced_at is not None:
                t_death = advanced_at + rng.exponential(spec.median_os_days / np.log(2))
            else:
                t_death = rng.exponential(spec.median_os_days * 4 / np.log(2)) + 200
            death = float(t_death) if t_death <= follow_up else None
            if death is not None:
                follow_up = death
            progression = None
            if advanced_at is not None:
                t_prog = advanced_at + rng.exponential(spec.median_pfs_days / np.log(2)) + 20
                if t_prog < follow_up:
                    progression = float(t_prog)
            met_sites: list[str] = []
            if advanced_at is not None:
                for site, p in _SITE_WEIGHTS[self.cohort].items():
                    if rng.random() < p:
                        met_sites.append(site)
                if not met_sites:
                    met_sites.append(max(_SITE_WEIGHTS[self.cohort], key=_SITE_WEIGHTS[self.cohort].get))
            self.patients.append(
                _Patient(
                    record_id=rid,
                    institution=inst,
                    birth_year=birth_year,
                    sex=sex,
                    age_dx=age_dx,
                    dob_dx=dob_dx,
                    stage=stage,
                    advanced_at=advanced_at,
                    met_sites=met_sites,
                    follow_up=follow_up,
                    death=death,
                    progression=progression,
                    adjuvant=adjuvant,
                    index_ca_seq=index_seq,
                    prior_cancer=prior,
                )
            )
        self._simulate_regimens()

    def _simulate_regimens(self) -> None:
        rng, spec = self.rng, self.spec
        for p in self.patients:
            regs: list[dict] = []
            number = 0
            adjuvant_end: float | None = None
            if p.adjuvant and p.stage != "Stage IV":
                start = float(rng.uniform(30, 90))
                end = start + float(rng.uniform(90, 180))
                # recurrence during adjuvant therapy ends the adjuvant regimen
                if p.advanced_at is not None and end > p.advanced_at:
                    end = max(p.advanced_at, start + 1.0)
                if start < p.end and (p.advanced_at is None or start < p.advanced_at):
                    number += 1
                    adjuvant_end = min(end, p.end)
                    regs.append(
                        {
                            "number": number,
                            "drugs": sorted(spec.first_line[int(rng.integers(len(spec.first_line)))]),
                            "start": start,
                            "end": adjuvant_end,
                            "intent": "adjuvant",
                        }
                    )
            if p.advanced_at is not None:
                t = p.advanced_at + float(rng.uniform(10, 60))
                if adjuvant_end is not None:
                    t = max(t, adjuvant_end + float(rng.uniform(7, 30)))
                line = 0
                while t < p.end and line < 4:
                    line += 1
                    menu = spec.first_line if line == 1 else spec.later_line
                    drugs = sorted(menu[int(rng.integers(len(menu)))])
                    duration = float(rng.exponential(200) + 21)
                    end = min(t + duration, p.end)
                    number += 1
                    regs.append(
                        {"number": number, "drugs": drugs, "start": t, "end": end, "intent": f"line{line}"}
                    )
                    if end >= p.end:
                        break
                    t = end + float(rng.uniform(7, 120))
            p.regimens = regs

    # ------------------------------------------------------------------ tables
    def patient_table(self) -> pd.DataFrame:
        f = self._base_frame("patient", self.n)
        ps = self.patients
        self._set(f, "record_id", [p.record_id for p in ps])
        self._set(f, "institution", [p.institution for p in ps])
        self._set(f, "redacted", "No")
        self._set(f, "birth_year", [p.birth_year for p in ps])
        self._set(f, "naaccr_sex_code", [p.sex for p in ps])
        last_alive = np.array([p.dob_dx + p.follow_up for p in ps])
        death = np.array([p.dob_dx + p.death if p.death is not None else np.nan for p in ps])
        self._set(f, "last_alive_int", np.round(last_alive))
        self._set(f, "dob_lastalive_int", np.round(last_alive))
        self._set(f, "dob_lastalive_int_mos", last_alive / MONTH_DAYS)
        self._set(f, "dob_lastalive_int_yrs", last_alive / YEAR_DAYS)
        self._set(f, "age_last_fu_yrs", np.floor(last_alive / YEAR_DAYS))
        self._set(f, "last_oncvisit_int", np.round(last_alive - self.rng.integers(0, 60, size=self.n)))
        self._set(f, "last_anyvisit_int", np.round(last_alive))
        self._set(f, "hybrid_death_int", np.round(death))
        self._set(f, "age_death_yrs", np.floor(death / YEAR_DAYS))
        src_levels = self.schema.levels("patient", "hybrid_death_source")
        sources = _parse_levels(src_levels, False) or ["Curated", "EHR", "NDI"]
        self._set(
            f,
            "hybrid_death_source",
            pd.array(
                [str(self.rng.choice(sources)) if p.death is not None else None for p in ps], dtype=object
            ),
        )
        self._set(f, "n_cancers", [1 + int(p.prior_cancer) for p in ps])
        self._set(f, "n_cancers_index", 1)
        self._set(f, "n_regimens_pt", [len(p.regimens) for p in ps])
        if "race_ethnicity" in f.columns:
            levels = _parse_levels(self.schema.levels("patient", "race_ethnicity"), False)
            if levels:
                f["race_ethnicity"] = self.rng.choice(np.array(levels, dtype=object), size=self.n)
        return f

    def _cancer_rows(self, index: bool) -> pd.DataFrame:
        ps = [p for p in self.patients if index or p.prior_cancer]
        table = "cancer_index" if index else "cancer_non_index"
        f = self._base_frame(table, len(ps))
        if f.empty:
            return f
        if not index:
            # the non-index file carries the subset of columns the real release uses
            keep = [
                c
                for c in f.columns
                if not c.startswith(("dmets", "dist_mets", "dx_to_d", "pfs", "tt_pfs", "os_adv", "tt_os_adv"))
            ]
            f = f[keep]
        rng = self.rng
        self._set(f, "record_id", [p.record_id for p in ps])
        self._set(f, "institution", [p.institution for p in ps])
        self._set(f, "tr_eligible", 1 if index else 0)
        self._set(f, "redcap_ca_index", "Yes" if index else "No")
        self._set(f, "ca_seq", [p.index_ca_seq if index else 0 for p in ps])
        offsets = np.zeros(len(ps)) if index else -rng.integers(400, 4000, size=len(ps)).astype(float)
        dob_dx = np.array([p.dob_dx for p in ps], dtype=float) + offsets
        self._days_to_derived(f, "dob_ca_dx", pd.Series(dob_dx))
        self._set(f, "age_dx", np.floor(dob_dx / YEAR_DAYS))
        self._set(f, "first_index_ca_days", np.array([p.dob_dx for p in ps], dtype=float))
        self._set(f, "ca_dx_how", "Pathology")
        if index:
            self._set(f, "ca_d_site", self.spec.ca_d_site)
            self._set(f, "ca_type", self.spec.ca_type)
            self._set(
                f,
                "naaccr_histology_cd",
                rng.choice(np.array(self.spec.histology_codes), size=len(ps)).astype(float),
            )
            stages = [p.stage for p in ps]
            self._set(f, "stage_dx", stages)
            self._set(f, "stage_dx_iv", ["Stage IV" if s == "Stage IV" else "Stage I-III" for s in stages])
            self._fill_metastases(f, ps)
            self._fill_cancer_outcomes(f, ps)
            self._fill_cohort_specific(f, ps)
        else:
            other = _parse_levels(self.schema.levels("cancer", "ca_type"), False)
            other = [o for o in other if o != self.spec.ca_type] or ["Other"]
            self._set(f, "ca_type", rng.choice(np.array(other, dtype=object), size=len(ps)))
            self._set(f, "ca_d_site", pd.array([None] * len(ps), dtype=object))
            self._set(f, "ca_stage", pd.array([None] * len(ps), dtype=object))
        return f

    def _fill_metastases(self, f: pd.DataFrame, ps: list[_Patient]) -> None:
        rng = self.rng
        n = len(ps)
        advanced = np.array([p.advanced_at for p in ps], dtype=float)  # nan when None
        early = np.array([p.stage != "Stage IV" for p in ps])
        post_dx = np.where(early, np.where(np.isnan(advanced), 0.0, 1.0), np.nan)
        self._set(f, "dmets_stage_i_iii", post_dx)
        self._set(f, "dmets_post_dx", post_dx)
        t_post = np.where(early & ~np.isnan(advanced), advanced, np.nan)
        self._days_to_derived(f, "dx_to_dmets", pd.Series(t_post))
        self._days_to_derived(f, "dx_to_dist_mets", pd.Series(t_post))
        prefix_flag = "dmets_" if self.gen == 1 else "dist_mets_"
        prefix_time = "dx_to_dmets_" if self.gen == 1 else "dx_to_dist_mets_"
        for site, (g1, g2) in _MET_SITES.items():
            suffix = g1 if self.gen == 1 else g2
            flag_col = f"{prefix_flag}{suffix}"
            if flag_col not in f.columns:
                continue
            has = np.array([site in p.met_sites for p in ps])
            f[flag_col] = np.where(np.isnan(advanced), 0.0, has.astype(float))
            days = np.where(has & ~np.isnan(advanced), advanced + rng.integers(0, 45, size=n), np.nan)
            self._days_to_derived(f, f"{prefix_time}{suffix}", pd.Series(days))
        # sites that are not simulated are explicitly absent (0) with no time
        simulated = {f"{prefix_flag}{(g1 if self.gen == 1 else g2)}" for g1, g2 in _MET_SITES.values()}
        for col in f.columns:
            if (
                col.startswith(prefix_flag)
                and col not in simulated
                and col not in ("dmets_stage_i_iii", "dmets_post_dx")
            ):
                f[col] = 0.0
                self._days_to_derived(
                    f, col.replace(prefix_flag, prefix_time, 1), pd.Series(np.full(n, np.nan))
                )

    def _fill_cancer_outcomes(self, f: pd.DataFrame, ps: list[_Patient]) -> None:
        n = len(ps)
        end = np.array([p.end for p in ps], dtype=float)
        dead = np.array([p.death is not None for p in ps], dtype=float)
        self._set(f, "os_dx_status", dead)
        self._days_to_derived(f, "tt_os_dx", pd.Series(end))
        advanced = np.array([p.advanced_at for p in ps], dtype=float)
        has_adv = ~np.isnan(advanced)
        self._set(f, "os_adv_status", np.where(has_adv, dead, np.nan))
        self._days_to_derived(f, "tt_os_adv", pd.Series(np.where(has_adv, end - advanced, np.nan)))
        levels = self.schema.levels("cancer", "pfs_cohort")
        iv = _pick_level(levels, "Stage IV", default="Stage IV")
        post = _pick_level(
            levels, "Post-Diagnosis", default=_pick_level(levels, "I-III", default="Stage I-III")
        )
        pfs_cohort = [
            (iv if p.stage == "Stage IV" else post) if p.advanced_at is not None else None for p in ps
        ]
        self._set(f, "pfs_cohort", pd.array(pfs_cohort, dtype=object))
        prog = np.array([p.progression for p in ps], dtype=float)
        has_prog = ~np.isnan(prog)
        event_time = np.where(has_prog, prog, end)
        status = np.where(has_adv, np.where(has_prog, 1.0, dead), np.nan)
        for kind, jitter in (("i", 0.0), ("m", 15.0), ("i_or_m", 0.0), ("i_and_m", 15.0)):
            t = np.where(
                has_adv, np.minimum(event_time + (jitter if kind != "i" else 0.0), end) - advanced, np.nan
            )
            self._set(f, f"pfs_{kind}_adv_status", status)
            self._days_to_derived(f, f"tt_pfs_{kind}_adv", pd.Series(t))
        _ = n

    def _fill_cohort_specific(self, f: pd.DataFrame, ps: list[_Patient]) -> None:
        rng = self.rng
        n = len(ps)
        smoke_col = "ca_lung_cigarette" if "ca_lung_cigarette" in f.columns else "ca_cigarette"
        if smoke_col in f.columns:
            levels = _parse_levels(self.schema.levels("cancer", smoke_col), False)
            if levels:
                f[smoke_col] = rng.choice(np.array(levels, dtype=object), size=n)
        grade = _parse_levels(self.schema.levels("cancer", "ca_grade"), False)
        if grade and "ca_grade" in f.columns:
            f["ca_grade"] = rng.choice(np.array(grade, dtype=object), size=n)
        if "naaccr_seer_sum_stage" in f.columns:
            if self.gen == 1:
                f["naaccr_seer_sum_stage"] = pd.array(
                    [str(rng.choice(["Localized", "Regional", "Distant", "Unknown"])) for _ in range(n)],
                    dtype=object,
                )
            else:
                f["naaccr_seer_sum_stage"] = rng.choice(np.array([1, 2, 3, 4, 7, 9], dtype=float), size=n)
        if self.cohort == "BrCa":
            for col, pos in (("ca_bca_er", "Positive"), ("ca_bca_pr", "Positive")):
                lv = self.schema.levels("cancer", col)
                yes, no = (
                    _pick_level(lv, pos, default="Positive"),
                    _pick_level(lv, "Negative", default="Negative"),
                )
                self._set(
                    f, col, pd.array([yes if rng.random() < 0.75 else no for _ in range(n)], dtype=object)
                )
            lv = self.schema.levels("cancer", "ca_bca_her_summ")
            yes, no = (
                _pick_level(lv, "Positive", default="Positive"),
                _pick_level(lv, "Negative", default="Negative"),
            )
            self._set(
                f,
                "ca_bca_her_summ",
                pd.array([yes if rng.random() < 0.2 else no for _ in range(n)], dtype=object),
            )
        if self.cohort == "Prostate":
            lv = _parse_levels(self.schema.levels("cancer", "ca_pros_path_gscore"), False)
            if lv and "ca_pros_path_gscore" in f.columns:
                f["ca_pros_path_gscore"] = rng.choice(np.array(lv, dtype=object), size=n)

    def regimen_table(self) -> pd.DataFrame:
        rows: list[dict] = []
        for p in self.patients:
            for i, r in enumerate(p.regimens):
                rows.append({"patient": p, "reg": r, "within": i + 1})
        f = self._base_frame("regimen", len(rows))
        if f.empty:
            return f
        rng = self.rng
        ps = [r["patient"] for r in rows]
        regs = [r["reg"] for r in rows]
        self._set(f, "record_id", [p.record_id for p in ps])
        self._set(f, "institution", [p.institution for p in ps])
        self._set(f, "ca_seq", [p.index_ca_seq for p in ps])
        self._set(f, "regimen_number", [r["number"] for r in regs])
        self._set(f, "regimen_number_within_cancer", [r["within"] for r in rows])
        self._set(f, "redcap_ca_index", "Yes")
        self._set(f, "drugs_num", [len(r["drugs"]) for r in regs])
        self._set(f, "drugs_inst", [p.institution for p in ps])
        self._set(f, "drugs_firstinst", "Yes")
        trial = ["Yes" if "Investigational Drug" in r["drugs"] or rng.random() < 0.08 else "No" for r in regs]
        self._set(f, "drugs_ct_yn", pd.array(trial, dtype=object))
        self._set(f, "regimen_drugs", [", ".join(r["drugs"]) for r in regs])
        start = np.array([r["start"] for r in regs])
        end = np.array([r["end"] for r in regs])
        dob_dx = np.array([p.dob_dx for p in ps], dtype=float)
        for k in range(1, 6):
            has = np.array([len(r["drugs"]) >= k for r in regs])
            self._set(
                f,
                f"drugs_drug_{k}",
                pd.array([r["drugs"][k - 1] if len(r["drugs"]) >= k else None for r in regs], dtype=object),
            )
            self._set(f, f"drugs_startdt_int_{k}", np.where(has, np.round(dob_dx + start), np.nan))
            self._set(f, f"dx_drug_start_int_{k}", np.where(has, np.round(start), np.nan))
            self._set(f, f"dx_drug_start_int_mos_{k}", np.where(has, start / MONTH_DAYS, np.nan))
            drug_end = np.where(has, np.round(end - rng.integers(0, 15, size=len(regs))), np.nan)
            drug_end = np.maximum(drug_end, np.round(start))
            self._set(f, f"dx_drug_end_int_{k}", drug_end)
            self._set(f, f"dx_drug_end_int_mos_{k}", drug_end / MONTH_DAYS)
            self._set(f, f"dx_drug_end_or_lastadm_int_{k}", drug_end)
        start = np.round(start)
        end = np.round(end)
        self._set(f, "dx_reg_start_int", start)
        self._set(f, "dx_reg_start_int_mos", start / MONTH_DAYS)
        self._set(f, "dx_reg_start_int_yrs", start / YEAR_DAYS)
        first_end = (
            f["dx_drug_end_int_1"].to_numpy(dtype=float)
            if "dx_drug_end_int_1" in f.columns
            else np.round(end)
        )
        self._set(f, "dx_reg_end_any_int", np.minimum(first_end, np.round(end)))
        self._set(f, "dx_reg_end_any_int_mos", np.minimum(first_end, np.round(end)) / MONTH_DAYS)
        self._set(f, "dx_reg_end_any_int_yrs", np.minimum(first_end, np.round(end)) / YEAR_DAYS)
        self._set(f, "dx_reg_end_all_int", np.round(end))
        self._set(f, "dx_reg_end_all_int_mos", end / MONTH_DAYS)
        self._set(f, "dx_reg_end_all_int_yrs", end / YEAR_DAYS)
        dead = np.array([p.death is not None for p in ps], dtype=float)
        pend = np.array([p.end for p in ps])
        self._set(f, "os_g_status", dead)
        self._days_to_derived(f, "tt_os_g", pd.Series(pend - start))
        for k in range(1, 6):
            has = np.array([len(r["drugs"]) >= k for r in regs])
            self._set(
                f, f"os_d{k}_status", np.where(has, dead, np.nan)
            ) if f"os_d{k}_status" in f.columns else None
            self._days_to_derived(f, f"tt_os_d{k}", pd.Series(np.where(has, pend - start, np.nan)))
        self._set(f, "os_d_status", dead)
        prog = np.array([p.progression if p.progression is not None else np.nan for p in ps])
        prog_after = np.where(~np.isnan(prog) & (prog > start), prog, np.nan)
        status = np.where(~np.isnan(prog_after), 1.0, dead)
        t = np.where(~np.isnan(prog_after), prog_after, pend) - start
        for kind, jitter in (("i", 0.0), ("m", 15.0), ("i_or_m", 0.0), ("i_and_m", 15.0)):
            self._set(f, f"pfs_{kind}_g_status", status)
            self._days_to_derived(f, f"tt_pfs_{kind}_g", pd.Series(np.minimum(t + jitter, pend - start)))
        nxt = []
        for p, r in zip(ps, regs, strict=True):
            later = [x["start"] for x in p.regimens if x["start"] > r["start"]]
            nxt.append(min(later) if later else np.nan)
        nxt = np.array(nxt)
        has_next = ~np.isnan(nxt)
        self._set(f, "ttnt_any_ca_status", np.where(has_next, 1.0, dead))
        self._days_to_derived(f, "ttnt_any_ca", pd.Series(np.where(has_next, nxt, pend) - start))
        self._set(f, "ttnt_ca_seq_status", np.where(has_next, 1.0, dead))
        self._days_to_derived(f, "ttnt_ca_seq", pd.Series(np.where(has_next, nxt, pend) - start))
        return f

    def imaging_table(self) -> pd.DataFrame:
        rng = self.rng
        levels_ca = self.schema.levels("imaging", "image_ca")
        yes = _pick_level(levels_ca, "there is evidence", default="Yes")
        no = _pick_level(levels_ca, "no evidence", default="No")
        levels_ov = self.schema.levels("imaging", "image_overall")
        progressing = _pick_level(levels_ov, "Progressing", default="Progressing/Worsening/Enlarging")
        stable = _pick_level(levels_ov, "Stable", default="Stable/No change")
        improving = _pick_level(levels_ov, "Improving", default="Improving/Responding")
        scan_types = _parse_levels(self.schema.levels("imaging", "image_scan_type"), False) or ["CT"]
        scan_types = [s for s in scan_types if "Mammogram" not in s or self.cohort == "BrCa"]
        rows = []
        for p in self.patients:
            t = -float(rng.integers(0, 40))
            number = 0
            while t < p.end:
                number += 1
                disease = p.advanced_at is not None and t >= p.advanced_at
                if p.stage != "Stage IV" and p.advanced_at is None:
                    disease = t < 30  # tumour visible before resection
                if number == 1:
                    overall = None
                elif p.progression is not None and t >= p.progression and disease:
                    overall = progressing if rng.random() < 0.8 else stable
                elif disease:
                    overall = stable if rng.random() < 0.6 else improving
                else:
                    overall = None
                rows.append(
                    {
                        "record_id": p.record_id,
                        "institution": p.institution,
                        "scan_number": number,
                        "image_scan_int": p.dob_dx + t,
                        "dx_scan_days": t,
                        "image_scan_type": str(rng.choice(scan_types[: max(1, min(3, len(scan_types)))])),
                        "image_ca": yes if disease else no,
                        "image_overall": overall,
                    }
                )
                t += float(rng.uniform(45, 150))
        f = self._base_frame("imaging", len(rows))
        if f.empty:
            return f
        sim = pd.DataFrame(rows)
        for col in sim.columns:
            if col in f.columns:
                f[col] = (
                    sim[col].to_numpy()
                    if col not in ("image_overall",)
                    else pd.array(sim[col].tolist(), dtype=object)
                )
        self._days_to_derived(f, "dx_scan", f["dx_scan_days"])
        for col in [
            c for c in f.columns if c.startswith("image_scansite___") or c.startswith("image_casite")
        ]:
            f[col] = pd.array([None] * len(f), dtype=object)
        return f

    def med_onc_table(self) -> pd.DataFrame:
        rng = self.rng
        lv_ca = self.schema.levels("med_onc", "md_ca")
        yes = _pick_level(lv_ca, "there is evidence", default="Yes")
        no = _pick_level(lv_ca, "no evidence", default="No")
        lv_st = self.schema.levels("med_onc", "md_ca_status")
        progressing = _pick_level(lv_st, "Progressing", default="Progressing/Worsening/Enlarging")
        stable = _pick_level(lv_st, "Stable", default="Stable/No change")
        improving = _pick_level(lv_st, "Improving", default="Improving/Responding")
        not_stated = _pick_level(lv_st, "Not stated", default="Not stated/Indeterminate")
        type_levels = self.schema.levels("med_onc", "md_type_ca_cur")
        ca_type = _pick_level(type_levels, *self.spec.ca_type.split()[:1], default=self.spec.ca_type)
        rows = []
        for p in self.patients:
            t = float(rng.integers(5, 40))
            number = 0
            while t < p.end:
                number += 1
                disease = (p.advanced_at is not None and t >= p.advanced_at) or (
                    p.stage != "Stage IV" and p.advanced_at is None and t < 30
                )
                if not disease:
                    status = None
                elif p.progression is not None and t >= p.progression + 15:
                    status = progressing if rng.random() < 0.7 else not_stated
                else:
                    status = str(rng.choice([stable, improving, not_stated]))
                rows.append(
                    {
                        "record_id": p.record_id,
                        "institution": p.institution,
                        "md_visit_number": number,
                        "md_onc_visit_int": p.dob_dx + t,
                        "dx_md_visit_days": t,
                        "md_type_ca_cur": ca_type,
                        "md_ca": yes if disease else no,
                        "md_ca_status": status,
                    }
                )
                t += float(rng.uniform(25, 60))
        f = self._base_frame("med_onc", len(rows))
        if f.empty:
            return f
        sim = pd.DataFrame(rows)
        for col in sim.columns:
            if col in f.columns:
                f[col] = (
                    pd.array(sim[col].tolist(), dtype=object)
                    if col == "md_ca_status"
                    else sim[col].to_numpy()
                )
        self._days_to_derived(f, "dx_md_visit", f["dx_md_visit_days"])
        self._set(f, "md_inst", [p for p in f["institution"]])
        return f

    def cpt_table(self) -> pd.DataFrame:
        rng = self.rng
        panels = [p.name for p in self.release.files_in("gene_panels").values()]
        sample_types = _parse_levels(self.schema.levels("cpt", "sample_type"), False) or ["Primary tumor"]
        seq_numeric = self.schema.dtype("cpt", "cpt_seq_date").lower().startswith("num")
        rows = []
        for p in self.patients:
            n_samples = 2 if rng.random() < 0.1 else 1
            for k in range(1, n_samples + 1):
                sid = f"{p.record_id}-T{k:02d}"
                p.sample_ids.append(sid)
                inst_panels = [x for x in panels if x.startswith(p.institution)] or panels
                t = float(rng.integers(5, 120)) if k == 1 else float(rng.integers(120, max(121, int(p.end))))
                t = min(t, p.end)
                year = 2013 + int((p.dob_dx + t) % 7)
                rows.append(
                    {
                        "record_id": p.record_id,
                        "institution": p.institution,
                        "cpt_number": k,
                        "ca_seq": p.index_ca_seq,
                        "cpt_n_ca_seq": 1,
                        "cpt_genie_sample_id": sid,
                        "cpt_oncotree_code": str(rng.choice(list(self.spec.oncotree_codes))),
                        "sample_type": str(sample_types[0] if k == 1 else rng.choice(sample_types)),
                        "cpt_seq_assay_id": str(rng.choice(inst_panels)),
                        "cpt_seq_date": year if seq_numeric else str(year),
                        "dx_cpt_rep_days": t,
                        "dob_cpt_report_days": p.dob_dx + t,
                        "cpt_order_int": p.dob_dx + t - 14,
                        "cpt_report_post_death": 0,
                        "cpt_report_post_last_alive": 0,
                        "path_proc_number": 1,
                        "path_rep_number": 1,
                    }
                )
        f = self._base_frame("cpt", len(rows))
        if f.empty:
            return f
        sim = pd.DataFrame(rows)
        for col in sim.columns:
            if col in f.columns:
                f[col] = sim[col].to_numpy()
        self._days_to_derived(f, "dx_cpt_rep", f["dx_cpt_rep_days"])
        self._days_to_derived(f, "dob_cpt_report", f["dob_cpt_report_days"])
        return f

    def pathology_table(self) -> pd.DataFrame:
        rng = self.rng
        proc_types = _parse_levels(self.schema.levels("pathology", "path_proc_type"), False) or [
            "Surgical pathology"
        ]
        rows = []
        for p in self.patients:
            n_reports = int(rng.integers(1, 4))
            for k in range(1, n_reports + 1):
                t = float(rng.integers(-10, 5)) if k == 1 else float(rng.uniform(30, max(31, p.end)))
                rows.append(
                    {
                        "record_id": p.record_id,
                        "institution": p.institution,
                        "path_proc_number": k,
                        "path_rep_number": k,
                        "path_proc_int": p.dob_dx + t,
                        "dx_path_proc_days": t,
                        "path_proc_type": str(proc_types[0] if k == 1 else rng.choice(proc_types)),
                        "path_num_spec": 1,
                        "path_site1": str(rng.choice(list(self.spec.icdo_sites))),
                    }
                )
        f = self._base_frame("pathology", len(rows))
        if f.empty:
            return f
        sim = pd.DataFrame(rows)
        for col in sim.columns:
            if col in f.columns:
                f[col] = sim[col].to_numpy()
        self._days_to_derived(f, "dx_path_proc", f["dx_path_proc_days"])
        # biomarkers: MSI / MMR where the cohort declares them, PD-L1 everywhere
        first = (f["path_proc_number"] == 1).to_numpy()
        if "msi_testing" in f.columns:
            tested = first & (rng.random(len(f)) < 0.8)
            f["msi_testing"] = np.where(tested, "Yes", "No")
            lv = self.schema.levels("pathology", "msi_result")
            high = _pick_level(lv, "high", default="MSI-H")
            stable = _pick_level(lv, "stable", default="MSS")
            result = [(high if rng.random() < 0.15 else stable) if t else None for t in tested]
            f["msi_result"] = pd.array(result, dtype=object)
            if "msi_high_any" in f.columns:
                f["msi_high_any"] = pd.array(
                    ["Yes" if r == high else ("No" if r else None) for r in result], dtype=object
                )
        if "mmr_testing" in f.columns:
            tested = first & (rng.random(len(f)) < 0.6)
            f["mmr_testing"] = np.where(tested, "Yes", "No")
            lv = self.schema.levels("pathology", "mmr_result")
            deficient = _pick_level(lv, "deficient", default="MMR-D")
            proficient = _pick_level(lv, "proficient", default="MMR-P")
            f["mmr_result"] = pd.array(
                [(deficient if rng.random() < 0.12 else proficient) if t else None for t in tested],
                dtype=object,
            )
        if "pdl1_testing" in f.columns:
            tested = first & (rng.random(len(f)) < (0.7 if self.cohort == "NSCLC" else 0.2))
            f["pdl1_testing"] = np.where(tested, "Yes", "No")
            if "pdl1_perc" in f.columns:
                f["pdl1_perc"] = np.where(
                    tested, rng.choice([0, 1, 5, 25, 50, 80], size=len(f)).astype(float), np.nan
                )
        return f

    def tumor_marker_table(self) -> pd.DataFrame:
        rng = self.rng
        if not self.schema.has_table("tumor_marker") or not self.spec.tumor_markers:
            return pd.DataFrame()
        rows = []
        for p in self.patients:
            number = 0
            for tm, units, uln in self.spec.tumor_markers:
                if rng.random() < 0.6:
                    continue
                t = float(rng.integers(-5, 30))
                while t < p.end and number < 40:
                    number += 1
                    elevated = p.advanced_at is not None and t >= p.advanced_at
                    value = float(rng.lognormal(np.log(uln * (3.0 if elevated else 0.5)), 0.6))
                    rows.append(
                        {
                            "record_id": p.record_id,
                            "institution": p.institution,
                            "tm_number": number,
                            "tm_type": tm,
                            "tm_num_result": round(value, 1),
                            "tm_result_units": units,
                            "tm_normal_range_upper": uln,
                            "tm_normal_range_lower": 0.0,
                            "dx_tm_days": t,
                            "tm_spec_collect_int": p.dob_dx + t,
                        }
                    )
                    t += float(rng.uniform(40, 120))
        f = self._base_frame("tumor_marker", len(rows))
        if f.empty:
            return f
        sim = pd.DataFrame(rows)
        for col in sim.columns:
            if col in f.columns:
                f[col] = sim[col].to_numpy()
        self._days_to_derived(f, "dx_tm", f["dx_tm_days"])
        return f

    def radiation_table(self) -> pd.DataFrame:
        rng = self.rng
        if not self.schema.has_table("radiation"):
            return pd.DataFrame()
        rt_types = _parse_levels(self.schema.levels("radiation", "rt_type"), False) or [
            "External beam radiation therapy"
        ]
        rows = []
        for p in self.patients:
            if rng.random() > 0.3:
                continue
            t = float(rng.uniform(20, max(21, min(p.end, 400))))
            fractions = int(rng.choice([5, 10, 25, 30, 35]))
            rows.append(
                {
                    "record_id": p.record_id,
                    "institution": p.institution,
                    "ca_seq": p.index_ca_seq,
                    "rt_number": 1,
                    "rt_summ": "Yes",
                    "rt_type": str(rt_types[0]),
                    "dx_rt_start_days": t,
                    "rt_fractions": fractions,
                    "rt_total_dose": float(fractions * 200),
                    "rt_inst": p.institution,
                }
            )
        f = self._base_frame("radiation", len(rows))
        if f.empty:
            return f
        sim = pd.DataFrame(rows)
        for col in sim.columns:
            if col in f.columns:
                f[col] = sim[col].to_numpy()
        self._days_to_derived(f, "dx_rt_start", f["dx_rt_start_days"])
        for col in [c for c in f.columns if c.startswith("rt_site___")]:
            f[col] = pd.array([None] * len(f), dtype=object)
        return f

    # ---------------------------------------------------------------- genomic
    def _hgvsp(self, gene: str) -> tuple[str, str]:
        rng = self.rng
        hotspots = self.spec.hotspots.get(gene)
        if hotspots and rng.random() < 0.7:
            change = str(rng.choice(list(hotspots)))
            cls = "In_Frame_Del" if "del" in change else "Missense_Mutation"
            return f"p.{change}", cls
        pos = int(rng.integers(10, 900))
        roll = rng.random()
        if roll < 0.65:
            return f"p.{rng.choice(list(_AMINO))}{pos}{rng.choice(list(_AMINO))}", "Missense_Mutation"
        if roll < 0.85:
            return f"p.{rng.choice(list(_AMINO))}{pos}*", "Nonsense_Mutation"
        return f"p.{rng.choice(list(_AMINO))}{pos}fs", "Frame_Shift_Del"

    def mutations_table(self) -> pd.DataFrame:
        rng = self.rng
        rows = []
        for p in self.patients:
            for sid in p.sample_ids:
                for gene, prob in self.spec.genes.items():
                    if rng.random() >= prob:
                        continue
                    hgvsp, cls = self._hgvsp(gene)
                    ref = str(rng.choice(list(_BASES)))
                    alt = str(rng.choice([b for b in _BASES if b != ref]))
                    start = int(rng.integers(1_000_000, 200_000_000))
                    vtype = "DEL" if cls in ("In_Frame_Del", "Frame_Shift_Del") else "SNP"
                    rows.append(
                        {
                            "Hugo_Symbol": gene,
                            "Entrez_Gene_Id": 0,
                            "Center": p.institution,
                            "NCBI_Build": "GRCh37",
                            "Chromosome": _CHROMOSOMES.get(gene, "1"),
                            "Start_Position": start,
                            "End_Position": start + (int(rng.integers(3, 15)) if vtype == "DEL" else 0),
                            "Strand": "+",
                            "Consequence": "missense_variant"
                            if cls == "Missense_Mutation"
                            else "frameshift_variant",
                            "Variant_Classification": cls,
                            "Variant_Type": vtype,
                            "Reference_Allele": ref if vtype == "SNP" else ref * 4,
                            "Tumor_Seq_Allele1": ref if vtype == "SNP" else ref * 4,
                            "Tumor_Seq_Allele2": alt if vtype == "SNP" else "-",
                            "Tumor_Sample_Barcode": sid,
                            "Matched_Norm_Sample_Barcode": None,
                            "HGVSp_Short": hgvsp,
                            "HGVSc": None,
                            "t_ref_count": int(rng.integers(50, 500)),
                            "t_alt_count": int(rng.integers(5, 200)),
                            "n_ref_count": None,
                            "n_alt_count": None,
                            "Mutation_Status": "SOMATIC",
                            "Validation_Status": None,
                            "Annotation_Status": "SUCCESS",
                        }
                    )
        columns = [
            "Hugo_Symbol",
            "Entrez_Gene_Id",
            "Center",
            "NCBI_Build",
            "Chromosome",
            "Start_Position",
            "End_Position",
            "Strand",
            "Consequence",
            "Variant_Classification",
            "Variant_Type",
            "Reference_Allele",
            "Tumor_Seq_Allele1",
            "Tumor_Seq_Allele2",
            "Tumor_Sample_Barcode",
            "Matched_Norm_Sample_Barcode",
            "HGVSp_Short",
            "HGVSc",
            "t_ref_count",
            "t_alt_count",
            "n_ref_count",
            "n_alt_count",
            "Mutation_Status",
            "Validation_Status",
            "Annotation_Status",
        ]
        return pd.DataFrame(rows, columns=columns)

    def cna_table(self) -> pd.DataFrame:
        rng = self.rng
        samples = [sid for p in self.patients for sid in p.sample_ids]
        genes = sorted(set(self.spec.genes) | set(self.spec.cna_genes) | set(_FILLER_GENES))
        matrix = np.zeros((len(genes), len(samples)), dtype=int)
        for gi, gene in enumerate(genes):
            call = self.spec.cna_genes.get(gene)
            if call is None:
                continue
            hits = rng.random(len(samples)) < 0.08
            matrix[gi, hits] = call
        frame = pd.DataFrame(matrix, columns=samples)
        frame.insert(0, "Hugo_Symbol", genes)
        return frame

    def structural_variant_table(self) -> pd.DataFrame:
        rng = self.rng
        rows = []
        for p in self.patients:
            for sid in p.sample_ids:
                for gene, partner, label in self.spec.fusions:
                    if rng.random() < 0.05:
                        if self.gen == 1:
                            rows.append(
                                {
                                    "Hugo_Symbol": gene,
                                    "Entrez_Gene_Id": 0,
                                    "Center": p.institution,
                                    "Tumor_Sample_Barcode": sid,
                                    "Fusion": label,
                                    "DNA_support": "yes",
                                    "RNA_support": "unknown",
                                    "Method": None,
                                    "Frame": "in-frame",
                                }
                            )
                        else:
                            rows.append(
                                {
                                    "Sample_Id": sid,
                                    "SV_Status": "SOMATIC",
                                    "Site1_Hugo_Symbol": partner,
                                    "Site1_Entrez_Gene_Id": 0,
                                    "Site2_Hugo_Symbol": gene,
                                    "Site2_Entrez_Gene_Id": 0,
                                    "NCBI_Build": "GRCh37",
                                    "Class": "FUSION",
                                    "Event_Info": label,
                                    "Comments": None,
                                }
                            )
        if self.gen == 1:
            cols = [
                "Hugo_Symbol",
                "Entrez_Gene_Id",
                "Center",
                "Tumor_Sample_Barcode",
                "Fusion",
                "DNA_support",
                "RNA_support",
                "Method",
                "Frame",
            ]
        else:
            cols = [
                "Sample_Id",
                "SV_Status",
                "Site1_Hugo_Symbol",
                "Site1_Entrez_Gene_Id",
                "Site2_Hugo_Symbol",
                "Site2_Entrez_Gene_Id",
                "NCBI_Build",
                "Class",
                "Event_Info",
                "Comments",
            ]
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------------- build
    def build(self, tables: Iterable[str] | None, genomic: bool) -> CohortData:
        self._simulate_patients()
        wanted = set(tables) if tables else None
        out: dict[str, pd.DataFrame] = {}
        builders = {
            "patient": self.patient_table,
            "cancer_index": lambda: self._cancer_rows(True),
            "cancer_non_index": lambda: self._cancer_rows(False),
            "regimen": self.regimen_table,
            "imaging": self.imaging_table,
            "pathology": self.pathology_table,
            "med_onc": self.med_onc_table,
            "cpt": self.cpt_table,
            "tumor_marker": self.tumor_marker_table,
            "radiation": self.radiation_table,
        }
        # cpt first so sample ids exist for the genomic tables, patient last for counts
        order = ["cpt"] + [t for t in CLINICAL_TABLES if t not in ("cpt", "patient")] + ["patient"]
        for name in order:
            if wanted is not None and name not in wanted:
                continue
            if not self.schema.has_table(name):
                continue
            frame = builders[name]()
            if frame is not None and not frame.empty:
                out[name] = frame.reset_index(drop=True)
        if "cpt" not in out and any(p.sample_ids == [] for p in self.patients):
            self.cpt_table()  # assign sample ids even when the table is not requested
        if genomic:
            if wanted is None or "mutations" in wanted:
                out["mutations"] = self.mutations_table()
            if wanted is None or "cna" in wanted:
                out["cna"] = self.cna_table()
            sv_name = "fusions" if self.gen == 1 else "sv"
            if wanted is None or sv_name in wanted:
                out[sv_name] = self.structural_variant_table()
        if "patient" in out:
            pt = out["patient"]
            counts = {
                "n_imaging_reports_pt": "imaging",
                "n_path_reports_pt": "pathology",
                "n_md_notes_pt": "med_onc",
                "n_cpt_pt": "cpt",
            }
            for col, table in counts.items():
                if col in pt.columns and table in out:
                    c = out[table].groupby("record_id").size()
                    pt[col] = pt["record_id"].map(c).fillna(0).astype(int).to_numpy()
        ordered = {n: out[n] for n in [*CLINICAL_TABLES, "mutations", "fusions", "sv", "cna"] if n in out}
        return CohortData(
            cohort=self.cohort,
            version=self.version,
            tables=ordered,
            sources={n: "geniekit.synthetic" for n in ordered},
            synthetic=True,
        )


def generate(
    cohort: str,
    n_patients: int = 200,
    seed: int = 0,
    tables: Iterable[str] | None = None,
    genomic: bool = True,
) -> CohortData:
    """Generate a synthetic release for ``cohort`` (see module docstring).

    Parameters
    ----------
    cohort:
        One of the six public cohorts (any accepted alias).
    n_patients:
        Number of simulated patients.
    seed:
        Seed for the random generator; the same seed gives the same cohort.
    tables:
        Restrict the clinical and genomic tables produced.
    genomic:
        Whether to produce the mutations, copy-number and structural-variant
        tables.
    """
    c = normalize_cohort(cohort)
    if c not in COHORT_SPECS:
        raise ValueError(f"No synthetic specification for cohort {c!r}")
    if n_patients < 1:
        raise ValueError("n_patients must be at least 1")
    return _Generator(c, n_patients, seed).build(tables, genomic)


def generate_all(n_patients: int = 100, seed: int = 0, genomic: bool = True) -> dict[str, CohortData]:
    """Synthetic releases for all six cohorts, keyed by cohort label."""
    return {
        c: generate(c, n_patients=n_patients, seed=seed + i, genomic=genomic) for i, c in enumerate(COHORTS)
    }
