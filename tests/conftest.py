"""Shared fixtures.

Tests never touch Synapse or any patient-level data. Everything runs from the
packaged metadata and, later, from synthetic cohorts.
"""

from __future__ import annotations

import pytest

from geniekit import dictionary, releases


@pytest.fixture(scope="session")
def dd() -> dictionary.DataDictionary:
    return dictionary.load()


@pytest.fixture(scope="session")
def all_releases() -> list[releases.Release]:
    return releases.list_releases()
