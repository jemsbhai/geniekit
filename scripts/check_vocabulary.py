"""Fail if documentation or docstrings contain words on the project's banned list.

Usage::

    python scripts/check_vocabulary.py            # scan README, docs, CHANGELOG, NOTICE and src
    python scripts/check_vocabulary.py path ...   # scan specific files or directories

Exit status 1 when any banned word is found. The check is case-insensitive and
matches whole words only. Typographic dashes and curly quotes are reported as
well.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BANNED_WORDS = [
    "robust",
    "robustly",
    "robustness",
    "leverage",
    "leverages",
    "leveraged",
    "leveraging",
    "seamless",
    "seamlessly",
    "delve",
    "delves",
    "delving",
    "harness",
    "harnesses",
    "harnessing",
    "moreover",
    "furthermore",
    "additionally",
    "utilize",
    "utilizes",
    "utilizing",
    "cutting-edge",
    "state-of-the-art",
    "game-changing",
    "holistic",
    "synergy",
    "streamline",
    "streamlined",
    "empower",
    "empowers",
    "unlock",
    "unlocks",
]

TYPOGRAPHY = {
    chr(0x2013): "en dash",
    chr(0x2014): "em dash",
    chr(0x2018): "curly single quote",
    chr(0x2019): "curly single quote",
    chr(0x201C): "curly double quote",
    chr(0x201D): "curly double quote",
}

DEFAULT_PATHS = ["README.md", "CHANGELOG.md", "NOTICE.md", "docs", "src", "examples"]
EXTENSIONS = {".md", ".py", ".rst", ".txt", ".yaml", ".yml", ".ipynb"}

_WORD_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in BANNED_WORDS) + r")\b", re.IGNORECASE)


def iter_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(f for f in path.rglob("*") if f.suffix in EXTENSIONS and f.is_file())
        elif path.is_file():
            out.append(path)
    return sorted(set(out))


def scan(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path}: not UTF-8"]
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _WORD_RE.finditer(line):
            problems.append(f"{path}:{lineno}: banned word {m.group(0)!r}")
        for ch, name in TYPOGRAPHY.items():
            if ch in line:
                problems.append(f"{path}:{lineno}: {name}")
    return problems


def main(argv: list[str]) -> int:
    paths = argv or DEFAULT_PATHS
    problems: list[str] = []
    for f in iter_files(paths):
        problems.extend(scan(f))
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
