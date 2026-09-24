# Copyright (c) 2026 Richard Albright. All rights reserved.

"""Fixtures and gating for the soak / stress suite.

These tests intentionally saturate memory, CPU, and disk I/O, so they take
minutes and are **skipped unless `BSDB_STRESS=1`**. That keeps the normal
`pytest tests/` CI run (and a developer's quick run) fast, while the nightly
`soak` workflow runs them at full length.

Environment:
  BSDB_STRESS                set to "1" to enable this directory
  BSDB_SOAK_SECONDS          wall-clock budget for the mixed soak (default 30)
  BSDB_STRESS_RSS_CEILING_GB RSS ceiling asserted by the soak (default 16)
"""

import os

import pytest

BSDB_STRESS = os.environ.get("BSDB_STRESS") == "1"
DEFAULT_RSS_CEILING_GB = float(os.environ.get("BSDB_STRESS_RSS_CEILING_GB", "16"))


def pytest_collection_modifyitems(config, items):
    """Skip the whole directory unless explicitly enabled."""
    if BSDB_STRESS:
        return
    skip = pytest.mark.skip(reason="soak/stress suite; set BSDB_STRESS=1 to run")
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def soak_seconds() -> float:
    return float(os.environ.get("BSDB_SOAK_SECONDS", "30"))


@pytest.fixture(scope="session")
def rss_ceiling_gb() -> float:
    return DEFAULT_RSS_CEILING_GB


@pytest.fixture()
def table_uri(tmp_path) -> str:
    return f"file://{tmp_path / 'tbl'}"


@pytest.fixture(scope="session")
def rss_gb():
    """Current process RSS in GB. Requires psutil."""
    psutil = pytest.importorskip("psutil")

    def _rss() -> float:
        return psutil.Process(os.getpid()).memory_info().rss / 1e9

    return _rss


@pytest.fixture(scope="session")
def rows_of():
    """Row count of whatever the Python API returns.

    Accepts a pyarrow Table/RecordBatch, a list of batches, a pandas DataFrame
    (the `search()` API returns one), or any object with `num_rows`/`shape`.
    """

    def _rows(obj) -> int:
        if obj is None:
            return 0
        if hasattr(obj, "num_rows"):
            return int(obj.num_rows)
        if isinstance(obj, (list, tuple)):
            return sum(_rows(x) for x in obj)
        if hasattr(obj, "to_batches"):
            return sum(b.num_rows for b in obj.to_batches())
        # pandas.DataFrame (and anything 2-D): `read()` returns Arrow, but
        # `search()` returns a DataFrame.
        if hasattr(obj, "shape"):
            return int(obj.shape[0])
        if hasattr(obj, "__len__"):
            return len(obj)
        raise TypeError(f"cannot count rows of {type(obj)!r}")

    return _rows
