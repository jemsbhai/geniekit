"""Generate synthetic versions of all six public GENIE BPC releases.

Run from the repository root::

    python examples/synthetic_cohorts.py [output_dir]

Writes one folder per cohort (``<cohort>_<version>/<table>.csv``) that
``geniekit.io.load_release`` reads back, and prints a table summary. The data
are invented; see ``geniekit.synthetic`` for the caveats.
"""

from __future__ import annotations

import sys
from pathlib import Path

from geniekit import io, synthetic


def main(out: Path) -> None:
    for cohort, data in synthetic.generate_all(n_patients=200, seed=2026).items():
        folder = data.to_dir(out / f"{cohort}_{data.version}")
        back = io.load_release(folder)
        print(f"\n{back.label} written to {folder}")
        print(back.summary()[["table", "rows", "columns", "patients"]].to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data") / "synthetic")
