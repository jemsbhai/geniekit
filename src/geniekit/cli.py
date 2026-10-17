"""Command line interface: ``geniekit <command> [options]``.

Commands: ``releases``, ``dictionary``, ``access``, ``download``, ``inspect``,
``crosswalk``, ``harmonize``. ``lot`` and ``aggregate`` are added with their modules.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import pandas as pd

from . import dictionary, harmonize, io, lot, releases
from ._version import __version__


def _print_frame(frame: pd.DataFrame, max_rows: int | None = None) -> None:
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", None, "display.width", 200):
        print(frame.to_string(index=False) if not frame.empty else "(no rows)")


def cmd_releases(args: argparse.Namespace) -> int:
    rels = releases.list_releases(args.cohort)
    rows = [
        {
            "cohort": r.cohort,
            "version": r.version,
            "patients": r.n_patients,
            "institutions": "/".join(r.institutions),
            "schema_gen": r.schema_generation,
            "release_folder": r.release_folder,
            "clinical_files": len(r.clinical),
            "genomic_files": len(r.genomic),
        }
        for r in rels
    ]
    _print_frame(pd.DataFrame(rows))
    if args.files:
        frame = releases.to_frame()
        if args.cohort:
            frame = frame[frame["cohort"] == releases.normalize_cohort(args.cohort)]
        print()
        _print_frame(frame[frame["group"].isin(["clinical", "genomic", "documentation"])])
    print(f"\nregistry generated on {releases.registry_generated_on()}")
    return 0


def cmd_dictionary(args: argparse.Namespace) -> int:
    dd = dictionary.load()
    if args.dict_command == "lookup":
        rows = dd.lookup(args.variable, args.table)
        if rows.empty:
            print(f"{args.variable!r} is not declared in any cohort", file=sys.stderr)
            return 1
        out = rows[["cohort", "table", "dtype", "n_values", "label"]]
        _print_frame(out)
        if args.values:
            for c, vals in zip(rows["cohort"], rows["values"], strict=True):
                print(f"\n{c}:")
                for v in vals:
                    print(f"  - {v}")
        return 0
    if args.dict_command == "diff":
        kinds = args.kind or None
        diff = dd.differences(args.table, kinds)
        _print_frame(diff, max_rows=None)
        print(f"\n{len(diff)} rows")
        return 0
    if args.dict_command == "presence":
        mat = dd.presence(args.table)
        if args.only_partial:
            mat = mat[~mat.all(axis=1)]
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(mat.replace({True: "x", False: "."}).to_string())
        return 0
    return 2


def cmd_access(args: argparse.Namespace) -> int:
    try:
        session = io.SynapseSession(args.token)
        who = session.whoami()
        status = session.access_status()
    except io.SynapseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Synapse user: {who.get('userName')} (owner id {who.get('ownerId')})")
    print(status.describe())
    return 0 if status.approved else 2


def cmd_download(args: argparse.Namespace) -> int:
    try:
        session = io.SynapseSession(args.token)
        folder = session.download_release(
            args.cohort,
            version=args.version,
            dest=args.dest,
            groups=args.groups,
            names=args.tables or None,
            overwrite=args.overwrite,
            progress=print,
        )
    except (io.SynapseError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"release folder: {folder}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    try:
        data = io.load_release(
            args.path, cohort=args.cohort, version=args.version, genomic=not args.no_genomic
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(data)
    _print_frame(data.summary())
    report = data.validation.get("columns") or {}
    if report:
        print("\ncolumn differences versus the packaged dictionary:")
        for table, r in report.items():
            not_released = r.get("not_released") or []
            print(
                f"  {table}: missing {len(r['missing'])}, unexpected {len(r['extra'])}, "
                f"declared but not released {len(not_released)}"
            )
            if r["extra"]:
                print("    unexpected: " + ", ".join(r["extra"][:15]))
    return 0


def cmd_crosswalk(args: argparse.Namespace) -> int:
    cw = harmonize.load_crosswalk(args.file)
    frame = cw.to_frame()
    if args.kind:
        frame = frame[frame["kind"].isin(args.kind)]
    if not args.basis:
        frame = frame.drop(columns=["basis"])
    _print_frame(frame, max_rows=None)
    print(f"\n{cw}")
    if args.validate:
        problems = cw.validate()
        if problems:
            print("validation problems:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("validation: clean")
    return 0


def cmd_harmonize(args: argparse.Namespace) -> int:
    from pathlib import Path

    loaded = []
    for path in args.paths:
        try:
            loaded.append(io.load_release(path, genomic=not args.no_genomic))
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if args.compress else ".csv"
    if len(loaded) == 1:
        h = harmonize.harmonize(loaded[0], keep_original=args.keep_original)
        print(h)
        for name, frame in h.tables.items():
            frame.to_csv(out_dir / f"{name}{suffix}", index=False)
        report = h.report
    else:
        stacked = harmonize.stack(loaded)
        print(stacked)
        _print_frame(stacked.summary())
        for name, frame in stacked.tables.items():
            frame.to_csv(out_dir / f"{name}{suffix}", index=False)
            stacked.availability(name).to_csv(out_dir / f"availability_{name}.csv")
        report = stacked.report
    report.to_frame().to_csv(out_dir / "harmonization_report.csv", index=False)
    _print_frame(report.summary())
    unmapped = report.unmapped()
    if len(unmapped):
        print(f"\n{len(unmapped)} variable(s) had values the crosswalk does not know:")
        _print_frame(unmapped[["cohort", "table", "source", "detail"]])
    print(f"\nwritten to {out_dir}")
    return 0


def cmd_lot(args: argparse.Namespace) -> int:

    tables = ["patient", "cancer_index", "cancer_non_index", "regimen", "imaging", "med_onc"]
    loaded = []
    for path in args.paths:
        try:
            loaded.append(io.load_release(path, tables=tables, genomic=False))
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    index: object = args.index
    if args.index_table:
        index = pd.read_csv(args.index_table)
    try:
        rules = lot.PRESETS[args.rules]
    except KeyError:
        print(
            f"error: unknown rules preset {args.rules!r}; expected one of {sorted(lot.PRESETS)}",
            file=sys.stderr,
        )
        return 1
    changes = {}
    if args.combination_window is not None:
        changes["combination_window_days"] = args.combination_window
    if args.restart_gap is not None:
        changes["restart_gap_days"] = args.restart_gap
    if args.progression_aware:
        changes["progression_aware"] = True
    if args.pre_index:
        changes["pre_index"] = args.pre_index
    if changes:
        rules = rules.with_(**changes, name=f"{rules.name}+custom")
    try:
        result = lot.derive_all(loaded, index=index, rules=rules, cancers=args.cancers)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result)
    _print_frame(result.summary())
    if result.unknown_labels:
        print(f"\ndrug labels not in the packaged vocabulary: {', '.join(result.unknown_labels)}")
    folder = result.to_dir(args.out, compress=args.compress)
    print(f"\nwritten to {folder}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="geniekit", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"geniekit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("releases", help="list the known public BPC releases")
    pr.add_argument("--cohort", help="restrict to one cohort")
    pr.add_argument(
        "--files", action="store_true", help="also list clinical, genomic and documentation files"
    )
    pr.set_defaults(func=cmd_releases)

    pdict = sub.add_parser("dictionary", help="query the compiled variable synopses")
    dsub = pdict.add_subparsers(dest="dict_command", required=True)
    lk = dsub.add_parser("lookup", help="show how a variable is declared in each cohort")
    lk.add_argument("variable")
    lk.add_argument("--table", choices=dictionary.TABLES)
    lk.add_argument("--values", action="store_true", help="print permitted values per cohort")
    df = dsub.add_parser("diff", help="list variables not declared identically across cohorts")
    df.add_argument("--table", choices=dictionary.TABLES)
    df.add_argument(
        "--kind",
        action="append",
        choices=["missing_in_some", "dtype", "value_set", "value_text", "label"],
        help="restrict to one or more kinds of difference (repeatable)",
    )
    pres = dsub.add_parser("presence", help="presence matrix of variables by cohort")
    pres.add_argument("--table", choices=dictionary.TABLES)
    pres.add_argument("--only-partial", action="store_true", help="hide variables present in every cohort")
    pdict.set_defaults(func=cmd_dictionary)

    pa = sub.add_parser("access", help="check the Synapse token and GENIE access status")
    pa.add_argument("--token", help="Synapse personal access token (default: SYNAPSE_AUTH_TOKEN)")
    pa.set_defaults(func=cmd_access)

    pdl = sub.add_parser("download", help="download a public BPC release from Synapse")
    pdl.add_argument("cohort")
    pdl.add_argument("--version", help="release version, e.g. v2.0-public (default: latest public)")
    pdl.add_argument("--dest", help="destination directory (default: $GENIEKIT_DATA_DIR or ./data/genie_bpc)")
    pdl.add_argument(
        "--groups",
        nargs="+",
        default=["clinical", "genomic"],
        choices=list(releases.FILE_GROUPS),
        help="file groups to download",
    )
    pdl.add_argument("--tables", nargs="+", help="restrict to these logical table names")
    pdl.add_argument("--overwrite", action="store_true", help="re-download files that already exist")
    pdl.add_argument("--token", help="Synapse personal access token (default: SYNAPSE_AUTH_TOKEN)")
    pdl.set_defaults(func=cmd_download)

    pi = sub.add_parser("inspect", help="load a release folder and summarise its tables")
    pi.add_argument("path")
    pi.add_argument("--cohort")
    pi.add_argument("--version")
    pi.add_argument("--no-genomic", action="store_true", help="skip genomic files")
    pi.set_defaults(func=cmd_inspect)

    pc = sub.add_parser("crosswalk", help="list the harmonisation rules")
    pc.add_argument("--file", help="a custom crosswalk YAML (default: the packaged one)")
    pc.add_argument(
        "--kind", action="append", choices=list(harmonize.crosswalk.RULE_KINDS), help="restrict to rule kinds"
    )
    pc.add_argument("--basis", action="store_true", help="include the basis text")
    pc.add_argument("--validate", action="store_true", help="check the rules against the dictionary")
    pc.set_defaults(func=cmd_crosswalk)

    ph = sub.add_parser("harmonize", help="harmonise one or more release folders and write the result")
    ph.add_argument("paths", nargs="+", help="release folders (one per cohort)")
    ph.add_argument("--out", required=True, help="output directory")
    ph.add_argument("--compress", action="store_true", help="write .csv.gz")
    ph.add_argument("--keep-original", action="store_true", help="keep renamed columns as <name>_orig")
    ph.add_argument("--no-genomic", action="store_true", help="skip genomic files")
    ph.set_defaults(func=cmd_harmonize)

    pl = sub.add_parser("lot", help="derive lines of therapy from an index event the user chooses")
    pl.add_argument("paths", nargs="+", help="release folders (one per cohort)")
    pl.add_argument("--out", required=True, help="output directory")
    pl.add_argument(
        "--index",
        default="advanced",
        choices=list(lot.INDEX_EVENTS),
        help="built-in index event (default: advanced disease)",
    )
    pl.add_argument(
        "--index-table",
        help="CSV with record_id, ca_seq and index_days (and optionally index_source); overrides --index",
    )
    pl.add_argument("--rules", default="default", help=f"rules preset: {', '.join(sorted(lot.PRESETS))}")
    pl.add_argument("--combination-window", type=int, help="days within which added drugs join the line")
    pl.add_argument("--restart-gap", type=int, help="same agents after a longer gap (days) start a new line")
    pl.add_argument("--progression-aware", action="store_true", help="require a documented progression")
    pl.add_argument("--pre-index", choices=["exclude", "line_zero"], help="handling of pre-index regimens")
    pl.add_argument("--cancers", default="index", choices=["index", "non_index", "all"])
    pl.add_argument("--compress", action="store_true", help="write .csv.gz")
    pl.set_defaults(func=cmd_lot)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
