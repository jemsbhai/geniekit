"""Tests for the Synapse client using a stub HTTP layer (no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from geniekit.io import synapse
from geniekit.io.synapse import AccessDeniedError, AccessStatus, SynapseError, SynapseSession, resolve_token

TOKEN = "test-token"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b"", text=""):
        self.status_code = status_code
        self._payload = payload
        self._content = content
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHttp:
    """Routes requests to canned responses; records calls."""

    def __init__(self, approved=True, restricted=False, content=b"record_id\nGENIE-TEST-1\n", bad_md5=False):
        self.approved = approved
        self.restricted = restricted
        self.content = content
        self.bad_md5 = bad_md5
        self.calls: list[tuple[str, str]] = []

    def _handle(self):
        md5 = hashlib.md5(self.content).hexdigest()
        if self.bad_md5:
            md5 = "0" * 32
        return {
            "id": "123",
            "fileName": "patient_level_dataset.csv",
            "contentMd5": md5,
            "contentSize": len(self.content),
        }

    def get(self, url, headers=None, timeout=None, stream=False):
        self.calls.append(("GET", url))
        if headers is None or "Authorization" not in headers:
            # pre-signed download URL: no auth header
            return FakeResponse(content=self.content)
        if headers["Authorization"] != f"Bearer {TOKEN}":
            return FakeResponse(401, text="unauthorized")
        if url.endswith("/userProfile"):
            return FakeResponse(
                payload={"ownerId": "1", "userName": "tester", "firstName": "T", "lastName": "U"}
            )
        if url.endswith("/status"):
            return FakeResponse(
                payload={
                    "isApproved": self.approved,
                    "currentSubmissionStatus": None
                    if self.approved
                    else {
                        "submissionId": "18115",
                        "state": "SUBMITTED",
                        "modifiedOn": "2026-10-17T17:17:07Z",
                    },
                }
            )
        if "/entity/" in url:
            return FakeResponse(payload={"id": url.rsplit("/", 1)[-1], "name": "thing"})
        return FakeResponse(404, text="not found")

    def post(self, url, headers=None, data=None, timeout=None):
        self.calls.append(("POST", url))
        if url.endswith("/bundle2"):
            return FakeResponse(
                payload={
                    "restrictionInformation": {"hasUnmetAccessRequirement": self.restricted},
                    "fileHandles": [] if self.restricted else [self._handle()],
                }
            )
        if url.endswith("/fileHandle/batch"):
            return FakeResponse(
                payload={
                    "requestedFiles": [
                        {"preSignedURL": "https://example.invalid/presigned", "fileHandle": self._handle()}
                    ]
                }
            )
        return FakeResponse(500, text="boom")


def test_resolve_token_order(monkeypatch, tmp_path):
    monkeypatch.delenv("SYNAPSE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("SYNAPSE_PAT", raising=False)
    monkeypatch.setenv("SYNAPSE_CONFIG_FILE", str(tmp_path / "missing"))
    with pytest.raises(SynapseError, match="No Synapse personal access token"):
        resolve_token()
    assert resolve_token(" explicit ") == "explicit"
    monkeypatch.setenv("SYNAPSE_PAT", "from-pat")
    assert resolve_token() == "from-pat"
    monkeypatch.setenv("SYNAPSE_AUTH_TOKEN", "from-auth")
    assert resolve_token() == "from-auth"
    monkeypatch.delenv("SYNAPSE_AUTH_TOKEN")
    monkeypatch.delenv("SYNAPSE_PAT")
    cfg = tmp_path / ".synapseConfig"
    cfg.write_text("[authentication]\nauthtoken = from-config\n", encoding="utf-8")
    monkeypatch.setenv("SYNAPSE_CONFIG_FILE", str(cfg))
    assert resolve_token() == "from-config"


def test_whoami_and_access_status():
    s = SynapseSession(TOKEN, http=FakeHttp(approved=False))
    assert s.whoami()["userName"] == "tester"
    status = s.access_status()
    assert status == AccessStatus("9606645", False, "SUBMITTED", "18115", "2026-10-17T17:17:07Z")
    assert "SUBMITTED" in status.describe()
    assert "approved" in SynapseSession(TOKEN, http=FakeHttp()).access_status().describe()
    none = AccessStatus("9606645", False, None, None, None)
    assert "Request Access" in none.describe()


def test_bad_token_is_reported():
    s = SynapseSession("wrong", http=FakeHttp())
    with pytest.raises(SynapseError, match="401"):
        s.whoami()


def test_download_file_verifies_md5(tmp_path):
    http = FakeHttp()
    s = SynapseSession(TOKEN, http=http)
    path, handle = s.download_file("syn1", tmp_path)
    assert path == tmp_path / "patient_level_dataset.csv"
    assert path.read_bytes() == http.content and handle["contentMd5"]
    # second call with a matching checksum does not re-download
    n_calls = len(http.calls)
    s.download_file("syn1", tmp_path)
    assert not any(c == ("GET", "https://example.invalid/presigned") for c in http.calls[n_calls:])
    # explicit path target
    path2, _ = s.download_file("syn1", tmp_path / "sub" / "renamed.csv")
    assert path2.name == "renamed.csv" and path2.exists()


def test_download_file_md5_mismatch_removes_file(tmp_path):
    s = SynapseSession(TOKEN, http=FakeHttp(bad_md5=True))
    with pytest.raises(SynapseError, match="MD5 mismatch"):
        s.download_file("syn1", tmp_path)
    assert not (tmp_path / "patient_level_dataset.csv").exists()


def test_restricted_file_raises_access_denied(tmp_path):
    s = SynapseSession(TOKEN, http=FakeHttp(approved=False, restricted=True))
    with pytest.raises(AccessDeniedError, match="not approved"):
        s.download_file("syn1", tmp_path)


def test_download_release_writes_manifest(tmp_path, monkeypatch):
    http = FakeHttp()
    s = SynapseSession(TOKEN, http=http)
    messages: list[str] = []
    folder = s.download_release(
        "crc", dest=tmp_path, groups=["clinical"], names=["patient", "cpt"], progress=messages.append
    )
    assert folder == tmp_path / "CRC_v2.0-public"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert {f["name"] for f in manifest["files"]} == {"patient", "cpt"}
    assert manifest["release_folder"] == "syn30991637"
    assert len(messages) == 2 and "syn39802589" in messages[0] and "syn39802578" in messages[1]
    # files land under <group>/<filename> and the manifest paths are relative
    for f in manifest["files"]:
        # manifest paths are POSIX-style on every platform
        assert "\\" not in f["path"] and f["path"].startswith("clinical/")
        assert (folder / f["path"]).exists()
    # the convenience wrapper builds its own HTTP session; stub the factory so no network is touched,
    # and check that the default destination honours GENIEKIT_DATA_DIR
    monkeypatch.setattr(synapse.requests, "Session", lambda: FakeHttp())
    monkeypatch.setenv("GENIEKIT_DATA_DIR", str(tmp_path / "envdir"))
    folder2 = synapse.download_release(
        "crc", token=TOKEN, groups=["clinical"], names=["patient"], progress=None
    )
    assert folder2 == tmp_path / "envdir" / "CRC_v2.0-public"
    assert isinstance(folder2, Path) and (folder2 / "clinical" / "patient_level_dataset.csv").exists()


def test_download_release_requires_approval(tmp_path):
    s = SynapseSession(TOKEN, http=FakeHttp(approved=False))
    with pytest.raises(AccessDeniedError, match="SUBMITTED"):
        s.download_release("NSCLC", dest=tmp_path)
    assert not (tmp_path / "NSCLC_v2.0-public").exists()
