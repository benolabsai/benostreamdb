"""Version of the BenoStreamDB dbt adapter.

Since dbt 1.8 decoupled adapters from dbt-core, an adapter uses independent
SemVer and does NOT mirror dbt-core's version. This package therefore simply
carries the BenoStreamDB engine version, whose single source of truth is the
repository's ``Cargo.toml`` ``[workspace.package].version``.

dbt-core compatibility is declared separately, not through this number:

  * the ``dbt-adapters`` dependency in ``pyproject.toml``, and
  * ``require-dbt-version`` in ``dbt/include/benostreamdb/dbt_project.yml``.
"""

from __future__ import annotations

import os
import re
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path


def _from_cargo() -> str | None:
    """Read ``[workspace.package].version`` from the nearest Cargo.toml.

    Works while building from the repo (the source tree is present); returns
    ``None`` from an installed wheel, where no Cargo.toml exists.
    """
    for parent in Path(__file__).resolve().parents:
        cargo = parent / "Cargo.toml"
        if cargo.is_file():
            match = re.search(
                r'(?ms)^\[workspace\.package\].*?^version\s*=\s*"([0-9][^"]*)"',
                cargo.read_text(),
            )
            if match:
                return match.group(1)
    return None


def _detect() -> str:
    # Explicit release override wins: the release workflow sets this so a PyPI
    # repack can be <base>.postN while crates.io stays at <base>.
    forced = os.environ.get("BSDB_RELEASE_VERSION")
    if forced:
        return forced
    # Building from source: the repo Cargo.toml is authoritative.
    from_cargo = _from_cargo()
    if from_cargo:
        return from_cargo
    # Installed wheel: the distribution metadata is authoritative.
    try:
        return _dist_version("dbt-benostreamdb")
    except PackageNotFoundError:
        return "0.0.0"


version = _detect()
