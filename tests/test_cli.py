from __future__ import annotations

import json
import warnings

import pytest

from geniekit import cli


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "geniekit" in out


def test_releases_command(capsys):
    assert cli.main(["releases"]) == 0
    out = capsys.readouterr().out
    assert "BLADDER" in out and "syn75201334" in out and "registry generated on" in out
    assert cli.main(["releases", "--cohort", "crc", "--files"]) == 0
    out = capsys.readouterr().out
    assert (
        "tm_level_dataset.csv" in out and "NSCLC" not in out.split("registry generated")[0].split("\n\n")[1]
    )


def test_dictionary_lookup(capsys):
    assert cli.main(["dictionary", "lookup", "ca_cigarette", "--table", "cancer", "--values"]) == 0
    out = capsys.readouterr().out
    assert "BrCa" in out and "NSCLC" not in out
    assert cli.main(["dictionary", "lookup", "no_such_variable"]) == 1


def test_dictionary_diff_and_presence(capsys):
    assert cli.main(["dictionary", "diff", "--table", "cpt", "--kind", "dtype"]) == 0
    out = capsys.readouterr().out
    assert "cpt_seq_date" in out
    assert cli.main(["dictionary", "presence", "--table", "patient", "--only-partial"]) == 0
    out = capsys.readouterr().out
    assert "race_ethnicity" in out and "record_id" not in out


def test_inspect_fixture(capsys):
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "nsclc_sim"
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error"
        )  # the fixture matches the dictionary once not-released columns are known
        assert cli.main(["inspect", str(fixture), "--no-genomic"]) == 0
    out = capsys.readouterr().out
    assert "CohortData(NSCLC v2.0-public" in out and "regimen" in out and "column differences" in out
    assert "cpt: missing 0, unexpected 0, declared but not released 1" in out
    assert cli.main(["inspect", str(fixture / "missing")]) == 1


def test_access_and_download_without_token(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("SYNAPSE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("SYNAPSE_PAT", raising=False)
    monkeypatch.setenv("SYNAPSE_CONFIG_FILE", str(tmp_path / "none"))
    assert cli.main(["access"]) == 1
    assert "No Synapse personal access token" in capsys.readouterr().err
    assert cli.main(["download", "crc", "--dest", str(tmp_path)]) == 1


def test_access_and_download_with_stub(monkeypatch, tmp_path, capsys):
    from geniekit.io import synapse as syn_mod

    from .test_io_synapse import TOKEN, FakeHttp

    monkeypatch.setattr(syn_mod.requests, "Session", lambda: FakeHttp(approved=False))
    assert cli.main(["access", "--token", TOKEN]) == 2
    assert "SUBMITTED" in capsys.readouterr().out
    monkeypatch.setattr(syn_mod.requests, "Session", lambda: FakeHttp())
    assert cli.main(["access", "--token", TOKEN]) == 0
    assert (
        cli.main(
            [
                "download",
                "crc",
                "--dest",
                str(tmp_path),
                "--groups",
                "clinical",
                "--tables",
                "patient",
                "--token",
                TOKEN,
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "release folder" in out and (tmp_path / "CRC_v2.0-public" / "manifest.json").exists()


def test_crosswalk_command(capsys):
    assert cli.main(["crosswalk", "--validate"]) == 0
    out = capsys.readouterr().out
    assert "cigarette-rename" in out and "validation: clean" in out
    assert cli.main(["crosswalk", "--kind", "derive", "--basis"]) == 0
    out = capsys.readouterr().out
    assert "race-ethnicity-derived" in out and "cigarette-rename" not in out


def test_harmonize_command(tmp_path, capsys):
    from geniekit import synthetic

    a = synthetic.generate("NSCLC", n_patients=8, seed=1).to_dir(tmp_path / "NSCLC_v2.0-public")
    b = synthetic.generate("PANC", n_patients=8, seed=2).to_dir(tmp_path / "PANC_v1.0-public")
    out = tmp_path / "out"
    assert cli.main(["harmonize", str(a), "--out", str(out / "one")]) == 0
    assert (out / "one" / "cancer_index.csv").exists() and (out / "one" / "harmonization_report.csv").exists()
    assert cli.main(["harmonize", str(a), str(b), "--out", str(out / "two"), "--compress"]) == 0
    text = capsys.readouterr().out
    assert "StackedData" in text and (out / "two" / "patient.csv.gz").exists()
    assert (out / "two" / "availability_cancer_index.csv").exists()
    assert cli.main(["harmonize", str(tmp_path / "missing"), "--out", str(out)]) == 1


def test_lot_command(tmp_path, capsys):
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "nsclc_sim"
    out = tmp_path / "lot"
    assert (
        cli.main(["lot", str(fixture), "--out", str(out), "--index", "advanced", "--restart-gap", "90"]) == 0
    )
    text = capsys.readouterr().out
    assert "LinesOfTherapy(NSCLC" in text and "written to" in text
    assert (out / "lot_lines.csv").exists() and (out / "lot_settings.json").exists()
    settings = json.loads((out / "lot_settings.json").read_text(encoding="utf-8"))
    assert settings["rules"]["restart_gap_days"] == 90 and settings["rules"]["name"] == "default+custom"
    # a custom index table and the other options
    table = tmp_path / "index.csv"
    table.write_text("record_id,ca_seq,index_days\nGENIE-TEST-0005,2,100\n", encoding="utf-8")
    args = ["lot", str(fixture), "--out", str(out / "custom"), "--index-table", str(table)]
    args += ["--rules", "progression", "--pre-index", "line_zero", "--combination-window", "14", "--compress"]
    assert cli.main(args) == 0
    assert (out / "custom" / "lot_lines.csv.gz").exists()
    assert cli.main(["lot", str(fixture), "--out", str(out), "--rules", "nope"]) == 1
    assert cli.main(["lot", str(fixture / "missing"), "--out", str(out)]) == 1
