"""Parse the "Distant Metastasis Classification" appendix of every Analytic Data Guide.

Each guide ends with a two-column table mapping ICD-O-3 topography codes to a
metastasis classification such as "Distant (Liver)" or "Local/Regional
(Thorax)". The classification is what the ``dmets_*`` (generation 1) and
``dist_mets_*`` (generation 2) flags are derived from, so the tables decide
which cross-generation renames are legitimate (see ``crosswalk.yaml``) and
feed the site rederivation in ``geniekit.harmonize.metastasis``.

The primary strategy reads each page's raw character stream in content
order, where every table cell appears as "<code> <description> <status>
(<region>)" even when the cell text wraps (pdfplumber's word clustering drops
some wrapped cells). Ruled-table extraction is run as a cross-check: where
both strategies label a code they must agree, and the script fails otherwise.

Run from the repository root (downloads the six guides into data-raw/cache)::

    python data-raw/build_metastasis_sites.py

Output: ``data-raw/metastasis_sites.json`` and a printed comparison of the
per-region code sets across cohorts.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    sys.exit("pdfplumber is required: pip install pdfplumber")

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ROOT / "src" / "geniekit" / "resources" / "releases.json"
CACHE = ROOT / "data-raw" / "cache"
OUT = ROOT / "data-raw" / "metastasis_sites.json"

TITLE = re.compile(r"Appendix \d\..*Distant Metastasis Classification")
CODE_START = re.compile(r"^\s*([CF]\d\d(?:\.\d)?)\b\s*(.*)$", re.S)
LABEL_FULL = re.compile(r"^\s*(Distant|Local/Regional|Local|Regional|Not distant)\s*\((.*)\)\s*$", re.S)
# code, description (which may not run across the next code), status, region
ENTRY = re.compile(
    r"([CF]\d\d(?:\.\d)?)\s+((?:(?![CF]\d\d(?:\.\d)?\b).)*?)\s*"
    r"(Distant|Local/Regional|Local|Regional|Not distant)\s*\(([^)]*)\)",
    re.S,
)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "geniekit-build/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())
    return dest


def _appendix_start(pdf) -> int:
    n = len(pdf.pages)
    start = None
    for i in range(n - 1, max(n - 60, 0), -1):
        if TITLE.search(pdf.pages[i].extract_text() or ""):
            start = i
    if start is None:
        raise RuntimeError("appendix not found")
    return start


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_tables(pdf, start: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for page in pdf.pages[start:]:
        for table in page.extract_tables():
            for row in table:
                cells = [_clean((c or "").replace("\n", " ")) for c in row]
                cells = [c for c in cells if c]
                if len(cells) < 2:
                    continue
                m = CODE_START.match(cells[0])
                lab = LABEL_FULL.match(cells[-1])
                if m and lab:
                    out[m.group(1)] = {
                        "description": _clean(m.group(2)),
                        "status": lab.group(1),
                        "region": _clean(lab.group(2)),
                    }
    return out


def parse_stream(pdf, start: int) -> dict[str, dict]:
    """Primary pass: regex over each page's character stream in content order."""
    out: dict[str, dict] = {}
    for page in pdf.pages[start:]:
        stream = re.sub(r"\s+", " ", "".join(c["text"] for c in page.chars))
        for m in ENTRY.finditer(stream):
            code, desc, status, region = m.groups()
            out[code] = {"description": _clean(desc), "status": status, "region": _clean(region)}
    return out


def parse_guide(pdf_path: Path) -> dict[str, dict]:
    with pdfplumber.open(pdf_path) as pdf:
        start = _appendix_start(pdf)
        primary = parse_stream(pdf, start)
        check = parse_tables(pdf, start)
    for code, t in check.items():
        p = primary.get(code)
        if p is None:
            raise RuntimeError(
                f"{pdf_path.name}: {code} found by the table pass but not in the character stream"
            )
        if p["region"].lower() != t["region"].lower() or p["status"] != t["status"]:
            raise RuntimeError(
                f"{pdf_path.name}: {code} is {t['status']} ({t['region']}) by table but "
                f"{p['status']} ({p['region']}) in the character stream"
            )
    return {code: {**v, "verified_by_table": code in check} for code, v in sorted(primary.items())}


def compare(sites: dict[str, dict[str, dict]]) -> None:
    cohorts = list(sites)
    regions = sorted({v["region"].lower() for c in cohorts for v in sites[c].values()})
    print("\nper-region code counts (region as spelled in each guide, case-insensitive):")
    print(f"{'region':40s}" + "".join(f"{c:>10s}" for c in cohorts))
    for region in regions:
        counts = [sum(1 for v in sites[c].values() if v["region"].lower() == region) for c in cohorts]
        print(f"{region:40s}" + "".join(f"{n:10d}" for n in counts))
    by_region: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for c in cohorts:
        for code, v in sites[c].items():
            by_region[v["region"].lower()].setdefault(c, set()).add(code)
    print("\nregions whose code set is identical in every cohort that uses the region:")
    for region, per in sorted(by_region.items()):
        sets = list(per.values())
        if len(per) == len(cohorts) and all(s == sets[0] for s in sets):
            print(f"   {region}: {len(sets[0])} codes")


def main() -> None:
    releases = json.loads(RELEASES.read_text(encoding="utf-8"))["releases"]
    sites: dict[str, dict[str, dict]] = {}
    for rel in releases:
        url = rel["sources"]["data_guide_url"]
        dest = CACHE / f"{rel['cohort']}_{rel['version']}_guide.pdf"
        _download(url, dest)
        parsed = parse_guide(dest)
        n_checked = sum(1 for v in parsed.values() if v["verified_by_table"])
        print(f"{rel['label']:22s} {len(parsed):4d} codes ({n_checked} cross-checked by the table pass)")
        sites[rel["cohort"]] = parsed
    OUT.write_text(
        json.dumps({"generated_from": "Analytic Data Guide appendices", "cohorts": sites}, indent=1) + "\n"
    )
    print(f"wrote {OUT}")
    compare(sites)


if __name__ == "__main__":
    main()
