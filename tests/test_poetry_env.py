"""The Poetry environment matches ``poetry.lock``.

A relock without ``poetry sync`` leaves the env on the old package set, and a
test that only imports a package can still pass on it. This compares the
installed distributions with the lock, in both directions:

- every package the lock expects on this interpreter in the default groups is
  installed (sibling path dependencies included);
- every installed package the lock pins is at a version a lock entry allows here
  (sibling path dependencies excepted, as are pip and the project when no lock
  entry for them applies in the default groups; see ``_classify``);
- nothing is installed that the lock does not list, or lists only for other
  platforms or for groups pyproject.toml does not declare (packages of a declared
  optional group may be installed), except the project and pip, which
  ``poetry sync`` does not remove on Poetry 2's default install path
  (``installer.re-resolve = false``).

No CI job runs it yet (this repo has none): after a relock, run ``poetry sync``
and then ``pytest`` by hand.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.markers import Marker

ROOT = Path(__file__).resolve().parents[1]

# Lists every installed distribution as raw [name, version] pairs, in sys.path
# order. -I keeps the caller's cwd and environment off the child's sys.path, so it
# sees only the env; -B stops it writing bytecode.
_INSTALLED = """
import json, sys
from importlib.metadata import distributions
json.dump([[d.metadata["Name"], d.version] for d in distributions()], sys.stdout)
"""


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed() -> dict[str, str]:
    proc = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _INSTALLED],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    installed: dict[str, str] = {}
    for name, version in json.loads(proc.stdout):
        # Python imports the first distribution of a name on sys.path; keep that one.
        installed.setdefault(_canonical(name), version)
    return installed


def _groups(pyproject: dict[str, Any]) -> tuple[set[str], set[str]]:
    """(default groups, all groups), canonicalised as Poetry does.

    The default groups are what `poetry sync` installs without --with: main and
    every non-optional group.
    """
    assert "dependency-groups" not in pyproject, (
        "PEP 735 [dependency-groups] are not read by this test; teach _groups about them"
    )
    declared = pyproject.get("tool", {}).get("poetry", {}).get("group", {})
    every = {"main"} | {_canonical(name) for name in declared}
    default = {"main"} | {
        _canonical(name)
        for name, spec in declared.items()
        if not spec.get("optional", False)
    }
    return default, every


def _project_name(pyproject: dict[str, Any]) -> str:
    project = pyproject.get("project", {}).get("name")
    return _canonical(project or pyproject["tool"]["poetry"]["name"])


def _applies_here(entry: dict[str, Any], groups: set[str]) -> bool:
    """Whether this lock entry applies on this interpreter in any of ``groups``."""
    markers = entry.get("markers")
    for group in entry["groups"]:
        if _canonical(group) not in groups:
            continue
        marker = markers.get(group) if isinstance(markers, dict) else markers
        if marker is None or Marker(marker).evaluate():
            return True
    return False


@dataclass
class Drift:
    missing: list[str] = field(default_factory=list)
    stale: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    not_here: list[str] = field(default_factory=list)
    extraneous: list[str] = field(default_factory=list)


def _classify(
    lock_entries: list[dict[str, Any]],
    installed: dict[str, str],
    default_groups: set[str],
    all_groups: set[str],
    kept_by_sync: set[str],
) -> Drift:
    """Compare the installed distributions with the lock's entries."""
    # An entry that applies in a default group must have an allowed version below.
    assert default_groups <= all_groups, (default_groups, all_groups)
    # Poetry can write several entries for one name, split by marker.
    entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in lock_entries:
        entries[_canonical(entry["name"])].append(entry)

    drift = Drift()
    for name, candidates in entries.items():
        if name not in installed:
            if any(_applies_here(entry, default_groups) for entry in candidates):
                drift.missing.append(name)
            continue
        # A directory entry (a sibling checkout) is built from that tree, so its
        # installed version is the tree's, and `poetry sync` does not re-read it;
        # only a relock does. Its presence is checked above, its version is not.
        if any(
            entry.get("source", {}).get("type") == "directory" for entry in candidates
        ):
            continue
        # On Poetry 2's default install path (installer.re-resolve = false), `poetry
        # sync` never removes the project or pip, and pins pip only when pip is
        # locked here in a group it syncs; outside those groups it leaves them be.
        if name in kept_by_sync and not any(
            _applies_here(entry, default_groups) for entry in candidates
        ):
            continue
        allowed = [e["version"] for e in candidates if _applies_here(e, all_groups)]
        if not allowed:
            # Locked only for other platforms or groups: `poetry sync` uninstalls it.
            drift.not_here.append(name)
        elif installed[name] not in allowed:
            drift.stale[name] = (installed[name], allowed)

    drift.extraneous = sorted(set(installed) - set(entries) - kept_by_sync)
    drift.missing.sort()
    drift.not_here.sort()
    return drift


def test_poetry_env_matches_lock() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "poetry.lock").read_text(encoding="utf-8"))
    assert all("groups" in entry for entry in lock["package"]), (
        "poetry.lock predates format 2.1 (no per-entry groups): relock with Poetry 2"
    )
    assert any(_canonical(e["name"]) == "ultralytics" for e in lock["package"]), (
        "ultralytics is not in poetry.lock"
    )
    default_groups, all_groups = _groups(pyproject)
    # `poetry sync` keeps the project and pip on its default install path (see
    # `_classify`).
    kept_by_sync = {"pip", _project_name(pyproject)}

    drift = _classify(
        lock["package"], _installed(), default_groups, all_groups, kept_by_sync
    )

    assert not drift.missing, (
        f"locked but not installed: {drift.missing}: run `poetry sync`"
    )
    assert not drift.stale, (
        f"env differs from poetry.lock (installed, locked): {drift.stale}: run `poetry sync`"
    )
    assert not drift.not_here, (
        "installed, but poetry.lock lists them only for other platforms or groups: "
        f"{drift.not_here}: run `poetry sync`"
    )
    assert not drift.extraneous, (
        f"installed but not in poetry.lock: {drift.extraneous}: run `poetry sync`"
    )


# Unit tests of the pure helpers (_classify on synthetic locks, _groups, _project_name).

_DEFAULT = {"main", "dev"}
_ALL = {"main", "dev", "security-floors"}
_KEPT = {"pip", "qontinui-train"}


def _entry(name: str, version: str, groups: list[str], **extra: Any) -> dict[str, Any]:
    return {"name": name, "version": version, "groups": groups, **extra}


def _run(entries: list[dict[str, Any]], installed: dict[str, str]) -> Drift:
    return _classify(entries, installed, _DEFAULT, _ALL, _KEPT)


def test_classify_matching_env_is_clean() -> None:
    drift = _run([_entry("numpy", "2.0", ["main"])], {"numpy": "2.0", "pip": "26.0"})
    assert drift == Drift()


def test_classify_reports_a_version_drift() -> None:
    drift = _run([_entry("numpy", "2.0", ["main"])], {"numpy": "1.0"})
    assert drift.stale == {"numpy": ("1.0", ["2.0"])}


def test_classify_reports_a_missing_default_package_but_not_an_optional_one() -> None:
    entries = [
        _entry("numpy", "2.0", ["main"]),
        _entry("ujson", "6.0", ["security-floors"]),
    ]
    assert _run(entries, {}).missing == ["numpy"]


def test_classify_skips_a_package_locked_for_another_platform_when_absent() -> None:
    entries = [
        _entry("colorama", "0.4", ["main"], markers='sys_platform == "nonexistent"')
    ]
    assert _run(entries, {}) == Drift()


def test_classify_reports_an_installed_package_locked_only_for_another_platform() -> (
    None
):
    entries = [
        _entry("colorama", "0.4", ["main"], markers='sys_platform == "nonexistent"')
    ]
    assert _run(entries, {"colorama": "0.4"}).not_here == ["colorama"]


def test_classify_accepts_the_marker_split_entry_that_applies_here() -> None:
    entries = [
        _entry("numpy", "1.0", ["main"]),
        _entry("numpy", "2.0", ["main"], markers='sys_platform == "nonexistent"'),
    ]
    assert _run(entries, {"numpy": "1.0"}) == Drift()
    assert _run(entries, {"numpy": "2.0"}).stale == {"numpy": ("2.0", ["1.0"])}


def test_classify_checks_a_sibling_for_presence_but_not_version() -> None:
    sibling = _entry(
        "qontinui", "0.1", ["main"], source={"type": "directory", "url": "../q"}
    )
    assert _run([sibling], {}).missing == ["qontinui"]
    assert _run([sibling], {"qontinui": "9.9"}) == Drift()


def test_classify_reports_an_unlocked_package_except_what_sync_keeps() -> None:
    drift = _run([], {"stray": "1.0", "pip": "26.0", "qontinui-train": "0.1"})
    assert drift.extraneous == ["stray"]


def test_classify_reads_lock_group_names_canonically() -> None:
    entries = [
        _entry(
            "pytest", "9.0", ["Dev"], markers={"Dev": 'sys_platform != "nonexistent"'}
        )
    ]
    assert _run(entries, {"pytest": "9.0"}) == Drift()
    assert _run(entries, {}).missing == ["pytest"]


def test_classify_accepts_an_installed_package_of_an_optional_group() -> None:
    entries = [_entry("ujson", "6.0", ["security-floors"])]
    assert _run(entries, {"ujson": "6.0"}) == Drift()
    assert _run(entries, {"ujson": "5.0"}).stale == {"ujson": ("5.0", ["6.0"])}


def test_classify_keeps_pip_locked_elsewhere_but_checks_a_pip_locked_here() -> None:
    elsewhere = [
        _entry("pip", "25.0", ["main"], markers='sys_platform == "nonexistent"')
    ]
    assert _run(elsewhere, {"pip": "26.0"}) == Drift()
    here = [_entry("pip", "25.0", ["main"])]
    assert _run(here, {"pip": "26.0"}).stale == {"pip": ("26.0", ["25.0"])}


def test_classify_reads_a_marker_table_by_the_locks_own_group_spelling() -> None:
    entries = [
        _entry(
            "pytest", "9.0", ["Dev"], markers={"Dev": 'sys_platform == "nonexistent"'}
        )
    ]
    assert _run(entries, {}) == Drift()
    assert _run(entries, {"pytest": "9.0"}).not_here == ["pytest"]


def test_classify_treats_a_group_missing_from_a_marker_table_as_unconditional() -> None:
    entries = [
        _entry(
            "typing-extensions",
            "4.15",
            ["main", "security-floors"],
            markers={"security-floors": 'sys_platform == "nonexistent"'},
        )
    ]
    assert _run(entries, {}).missing == ["typing-extensions"]


def test_project_name_and_groups_read_a_pep_621_only_pyproject() -> None:
    pyproject: dict[str, Any] = {"project": {"name": "Demo_Pkg"}, "tool": {"ruff": {}}}
    assert _project_name(pyproject) == "demo-pkg"
    assert _groups(pyproject) == ({"main"}, {"main"})


def test_classify_leaves_pip_locked_only_in_an_optional_group_alone() -> None:
    # On Poetry 2's default install path, plain `poetry sync` neither pins nor
    # removes it, so the test must not demand it.
    entries = [_entry("pip", "25.0", ["security-floors"])]
    assert _run(entries, {"pip": "26.0"}) == Drift()
