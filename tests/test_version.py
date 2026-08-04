"""Tests keeping ``jraphx.__version__`` and the published changelog in step."""

import re
from pathlib import Path

import jraphx

CHANGELOG = Path(__file__).resolve().parents[1] / "docs" / "source" / "changelog.rst"

_VERSION_HEADING = re.compile(r"^Version (\d+\.\d+\.\d+)", re.MULTILINE)


def _changelog_versions() -> list[str]:
    """Collects the version headings of the changelog, newest first.

    Returns:
        list[str]: Every ``X.Y.Z`` string that heads a changelog section, in the
        order they appear in the file.
    """
    text = CHANGELOG.read_text(encoding="utf-8")
    return _VERSION_HEADING.findall(text)


def test_version_is_a_release_triple() -> None:
    """``__version__`` is a three-component version string."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", jraphx.__version__)


def test_changelog_documents_the_current_version() -> None:
    """The newest changelog section describes the version the package reports."""
    versions = _changelog_versions()
    assert versions, f"No 'Version X.Y.Z' headings found in {CHANGELOG}"
    assert versions[0] == jraphx.__version__


def test_changelog_versions_are_strictly_decreasing() -> None:
    """Changelog sections are ordered newest first, without duplicates."""
    versions = _changelog_versions()
    parsed = [tuple(int(part) for part in version.split(".")) for version in versions]
    assert parsed == sorted(set(parsed), reverse=True)
