"""Download GENIE BPC release files from Synapse with a personal access token.

This is a small client over the Synapse REST API (the same endpoints the R
package ``genieBPC`` uses), so the official ``synapseclient`` package is not
required. Authentication is a personal access token (PAT) with ``view`` and
``download`` scope, resolved in this order:

1. the ``token`` argument,
2. the ``SYNAPSE_AUTH_TOKEN`` environment variable (the name ``synapseclient``
   uses), then ``SYNAPSE_PAT`` (the name ``genieBPC`` uses),
3. ``authtoken`` in the ``[authentication]`` section of ``~/.synapseConfig``.

Downloading any GENIE file also requires an approved GENIE access
requirement (id 9606645) on the Synapse account. :meth:`SynapseSession.access_status`
reports where a request stands; :class:`AccessDeniedError` is raised when a download
is attempted without approval.
"""

from __future__ import annotations

import configparser
import datetime as dt
import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from ..releases import Release, get_release
from .cohort import CLINICAL_TABLES, GENOMIC_TABLES

__all__ = [
    "GENIE_ACCESS_REQUIREMENT",
    "AccessDeniedError",
    "AccessStatus",
    "DownloadedFile",
    "SynapseError",
    "SynapseSession",
    "download_release",
    "resolve_token",
]

REPO_ENDPOINT = "https://repo-prod.prod.sagebase.org/repo/v1"
FILE_ENDPOINT = "https://file-prod.prod.sagebase.org/file/v1"
# responses that are retried with a fresh signed URL (transient storage-side refusals)
RETRY_STATUSES = frozenset({403, 408, 429, 500, 502, 503, 504})
GENIE_ACCESS_REQUIREMENT = "9606645"
GENIE_REQUEST_ACCESS_URL = "https://www.synapse.org/Synapse:syn27056700"

_ENV_VARS = ("SYNAPSE_AUTH_TOKEN", "SYNAPSE_PAT")


class SynapseError(RuntimeError):
    """A Synapse API call failed."""


class AccessDeniedError(SynapseError):
    """The account has not met the GENIE access requirement."""


@dataclass(frozen=True)
class AccessStatus:
    requirement_id: str
    approved: bool
    state: str | None
    submission_id: str | None
    modified_on: str | None

    def describe(self) -> str:
        if self.approved:
            return f"access requirement {self.requirement_id}: approved"
        if self.state:
            return (
                f"access requirement {self.requirement_id}: not approved; latest submission "
                f"{self.submission_id} is {self.state} (updated {self.modified_on})"
            )
        return (
            f"access requirement {self.requirement_id}: not approved and no submission found. "
            f"Open {GENIE_REQUEST_ACCESS_URL}, click 'Request Access', and complete the ACT form."
        )


@dataclass(frozen=True)
class DownloadedFile:
    name: str
    group: str
    filename: str
    synapse_id: str
    file_handle_id: str
    md5: str | None
    bytes: int
    path: str


def resolve_token(token: str | None = None) -> str:
    """Return a Synapse personal access token (see module docstring for the search order)."""
    if token:
        return token.strip()
    for var in _ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    config_path = Path(os.environ.get("SYNAPSE_CONFIG_FILE", Path.home() / ".synapseConfig"))
    if config_path.exists():
        parser = configparser.ConfigParser()
        try:
            parser.read(config_path, encoding="utf-8")
            value = parser.get("authentication", "authtoken", fallback="").strip()
        except configparser.Error:
            value = ""
        if value:
            return value
    raise SynapseError(
        "No Synapse personal access token found. Pass token=..., set SYNAPSE_AUTH_TOKEN, "
        "or add authtoken to the [authentication] section of ~/.synapseConfig."
    )


class SynapseSession:
    """Authenticated access to the Synapse REST API.

    ``http`` may be any object with ``get``/``post`` methods compatible with
    :class:`requests.Session`; tests inject a stub.
    """

    def __init__(self, token: str | None = None, http: Any = None, timeout: float = 120.0) -> None:
        self.token = resolve_token(token)
        self.http = http or requests.Session()
        self.timeout = timeout

    # -- low level -----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def _get(self, url: str) -> Any:
        resp = self.http.get(url, headers=self._headers(), timeout=self.timeout)
        return self._parse(resp, url)

    def _post(self, url: str, body: dict[str, Any]) -> Any:
        resp = self.http.post(url, headers=self._headers(), data=json.dumps(body), timeout=self.timeout)
        return self._parse(resp, url)

    @staticmethod
    def _parse(resp: Any, url: str) -> Any:
        status = getattr(resp, "status_code", 200)
        if status == 401:
            raise SynapseError(f"Synapse rejected the token (HTTP 401) for {url}")
        if status == 403:
            raise AccessDeniedError(
                f"Synapse refused access (HTTP 403) for {url}: {getattr(resp, 'text', '')[:300]}"
            )
        if status >= 400:
            raise SynapseError(f"Synapse returned HTTP {status} for {url}: {getattr(resp, 'text', '')[:300]}")
        try:
            return resp.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise SynapseError(f"Non-JSON response from {url}") from exc

    # -- account and access --------------------------------------------------
    def whoami(self) -> dict[str, Any]:
        profile = self._get(f"{REPO_ENDPOINT}/userProfile")
        return {k: profile.get(k) for k in ("ownerId", "userName", "firstName", "lastName")}

    def access_status(self, requirement_id: str = GENIE_ACCESS_REQUIREMENT) -> AccessStatus:
        payload = self._get(f"{REPO_ENDPOINT}/accessRequirement/{requirement_id}/status")
        sub = payload.get("currentSubmissionStatus") or {}
        return AccessStatus(
            requirement_id=str(requirement_id),
            approved=bool(payload.get("isApproved", False)),
            state=sub.get("state"),
            submission_id=sub.get("submissionId"),
            modified_on=sub.get("modifiedOn"),
        )

    def entity(self, synapse_id: str) -> dict[str, Any]:
        return self._get(f"{REPO_ENDPOINT}/entity/{synapse_id}")

    def file_handle(self, synapse_id: str) -> dict[str, Any]:
        """The file handle of a FileEntity; raises :class:`AccessDeniedError` if restricted."""
        bundle = self._post(
            f"{REPO_ENDPOINT}/entity/{synapse_id}/bundle2",
            {"includeEntity": True, "includeFileHandles": True, "includeRestrictionInformation": True},
        )
        restriction = bundle.get("restrictionInformation") or {}
        if restriction.get("hasUnmetAccessRequirement"):
            status = self.access_status()
            raise AccessDeniedError(
                f"{synapse_id} is controlled by the GENIE access requirement and this account "
                f"is not approved. {status.describe()}"
            )
        handles = bundle.get("fileHandles") or []
        if not handles:
            raise SynapseError(f"{synapse_id} has no downloadable file handle (is it a folder?)")
        return handles[0]

    def presigned_url(self, synapse_id: str, file_handle_id: str) -> tuple[str, dict[str, Any]]:
        payload = self._post(
            f"{FILE_ENDPOINT}/fileHandle/batch",
            {
                "includeFileHandles": True,
                "includePreSignedURLs": True,
                "requestedFiles": [
                    {
                        "fileHandleId": str(file_handle_id),
                        "associateObjectId": synapse_id,
                        "associateObjectType": "FileEntity",
                    }
                ],
            },
        )
        requested = (payload.get("requestedFiles") or [{}])[0]
        url = requested.get("preSignedURL")
        if not url:
            failure = requested.get("failureCode") or "no pre-signed URL returned"
            if str(failure).upper() == "UNAUTHORIZED":
                raise AccessDeniedError(
                    f"Not authorised to download {synapse_id}: {self.access_status().describe()}"
                )
            raise SynapseError(f"Could not obtain a download URL for {synapse_id}: {failure}")
        return url, requested.get("fileHandle") or {}

    # -- downloads -----------------------------------------------------------
    def download_file(
        self,
        synapse_id: str,
        dest: str | Path,
        overwrite: bool = False,
        max_attempts: int = 5,
        backoff: float = 1.0,
    ) -> tuple[Path, dict[str, Any]]:
        """Download one file to ``dest`` (a directory or a full path); verify its MD5.

        Synapse issues short-lived signed URLs for its storage back end, and a
        freshly issued URL is occasionally answered with HTTP 403 (observed in
        practice) or a transient 5xx. Such responses, and an MD5 mismatch, are
        retried with a new signed URL up to ``max_attempts`` times with
        exponential backoff starting at ``backoff`` seconds. Other HTTP errors
        raise immediately.
        """
        handle = self.file_handle(synapse_id)
        dest = Path(dest)
        target = dest / handle["fileName"] if dest.is_dir() or not dest.suffix else dest
        target.parent.mkdir(parents=True, exist_ok=True)
        url, handle2 = self.presigned_url(synapse_id, handle["id"])
        handle = {**handle, **handle2}
        expected_md5 = handle.get("contentMd5")
        if target.exists() and not overwrite and expected_md5 and _md5(target) == expected_md5:
            return target, handle
        attempts = max(1, int(max_attempts))
        failure = ""
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                time.sleep(backoff * 2 ** (attempt - 2))
                url, handle2 = self.presigned_url(synapse_id, handle["id"])
                handle = {**handle, **handle2}
                expected_md5 = handle.get("contentMd5")
            with self.http.get(url, stream=True, timeout=self.timeout) as resp:
                status = getattr(resp, "status_code", 200)
                if status in RETRY_STATUSES:
                    failure = f"HTTP {status}"
                    continue
                if status >= 400:
                    raise SynapseError(f"Download of {synapse_id} failed with HTTP {status}")
                digest = hashlib.md5()
                with open(target, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                            digest.update(chunk)
            if expected_md5 and digest.hexdigest() != expected_md5:
                target.unlink(missing_ok=True)
                failure = "MD5 mismatch"
                continue
            return target, handle
        name = handle.get("fileName", "?")
        raise SynapseError(f"Download of {synapse_id} ({name}) failed after {attempts} attempt(s): {failure}")

    def download_release(
        self,
        release: Release | str,
        version: str | None = None,
        dest: str | Path | None = None,
        groups: Iterable[str] = ("clinical", "genomic"),
        names: Iterable[str] | None = None,
        overwrite: bool = False,
        progress: Callable[[str], None] | None = print,
        strict: bool = False,
    ) -> Path:
        """Download the files of one release into ``dest/<cohort>_<version>/<group>/``.

        Returns the release folder, which :func:`geniekit.io.load_release`
        accepts directly. A ``manifest.json`` records Synapse ids, file handle
        ids, MD5 checksums and sizes of the downloaded files, and under
        ``"failed"`` any file that could not be downloaded after the retries
        in :meth:`download_file`. Failures do not stop the other files; with
        ``strict=True`` they raise :class:`SynapseError` once every file has
        been tried (the manifest is written first). An access failure
        (:class:`AccessDeniedError`) always raises.
        """
        rel = release if isinstance(release, Release) else get_release(release, version)
        base = Path(dest or os.environ.get("GENIEKIT_DATA_DIR") or Path("data") / "genie_bpc")
        folder = base / f"{rel.cohort}_{rel.version}"
        wanted = set(names) if names else None
        status = self.access_status()
        if not status.approved:
            raise AccessDeniedError(status.describe())
        downloaded: list[DownloadedFile] = []
        failed: list[dict[str, str]] = []
        order = {n: i for i, n in enumerate((*CLINICAL_TABLES, *GENOMIC_TABLES))}
        for group in groups:
            refs = sorted(rel.files_in(group).items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))
            for name, ref in refs:
                if wanted is not None and name not in wanted:
                    continue
                target_dir = folder / group
                if progress:
                    progress(f"{rel.label}: {group}/{ref.filename} ({ref.synapse_id})")
                try:
                    path, handle = self.download_file(
                        ref.synapse_id, target_dir / ref.filename, overwrite=overwrite
                    )
                except AccessDeniedError:
                    raise
                except SynapseError as exc:
                    failed.append(
                        {
                            "name": name,
                            "group": group,
                            "filename": ref.filename,
                            "synapse_id": ref.synapse_id,
                            "error": str(exc),
                        }
                    )
                    if progress:
                        progress(f"  failed: {exc}")
                    continue
                downloaded.append(
                    DownloadedFile(
                        name=name,
                        group=group,
                        filename=ref.filename,
                        synapse_id=ref.synapse_id,
                        file_handle_id=str(handle.get("id", "")),
                        md5=handle.get("contentMd5"),
                        bytes=int(handle.get("contentSize") or path.stat().st_size),
                        path=path.relative_to(folder).as_posix(),
                    )
                )
        manifest = {
            "cohort": rel.cohort,
            "version": rel.version,
            "release_folder": rel.release_folder,
            "downloaded_on": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "files": [asdict(f) for f in downloaded],
            "failed": failed,
        }
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if failed and strict:
            names_failed = ", ".join(f["filename"] for f in failed)
            raise SynapseError(
                f"{len(failed)} file(s) of {rel.label} could not be downloaded: {names_failed}"
            )
        return folder


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_release(
    cohort: str,
    version: str | None = None,
    dest: str | Path | None = None,
    token: str | None = None,
    groups: Iterable[str] = ("clinical", "genomic"),
    names: Iterable[str] | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = print,
    strict: bool = False,
) -> Path:
    """Convenience wrapper: ``SynapseSession(token).download_release(...)``."""
    return SynapseSession(token).download_release(
        cohort,
        version=version,
        dest=dest,
        groups=groups,
        names=names,
        overwrite=overwrite,
        progress=progress,
        strict=strict,
    )
