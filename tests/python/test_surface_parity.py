# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

"""Surface parity beyond functions: **index types**, **DDL statements**, and
**stored procedures / table actions** must be consistent across the engine and
the connector surfaces (Python, dbt, Spark, Trino, Flight SQL).

The Rust core is the source of truth:
  * `benostreamdb.registered_index_algorithms()` — `IndexAlgorithm::all_names()`
  * `benostreamdb.registered_ddl_statements()` — `catalog_ddl::handled_ddl_statements()`
  * `benostreamdb.registered_table_actions()` — `catalog_ddl::table_action_names()`

Adding an index algorithm / DDL statement / action to the core without exposing
it in the other surfaces fails here, so the surfaces cannot silently drift.
"""

import pathlib
import re

import pyarrow as pa
import pytest

import benostreamdb

ROOT = pathlib.Path(__file__).resolve().parents[2]
MACROS_DIR = ROOT / "dbt-benostreamdb" / "dbt" / "include" / "benostreamdb" / "macros"
SPARK_DOC = ROOT / "docs" / "integrations" / "spark.md"
# Spark 4.x exposes stored procedures via the native DSv2 ProcedureCatalog;
# Spark 3.5 uses the Iceberg shim instead. Both live in per-minor source roots.
SPARK_PROCEDURE_CATALOG = (
    ROOT
    / "spark-benostreamdb"
    / "src"
    / "main"
    / "spark-4"
    / "com"
    / "benostreamdb"
    / "spark"
    / "BenoStreamCatalog.scala"
)
SPARK_PROCEDURE_TEST = (
    ROOT
    / "spark-benostreamdb"
    / "src"
    / "test"
    / "scala"
    / "com"
    / "benostreamdb"
    / "spark"
    / "SparkProcedureCatalogTest.scala"
)
TRINO_PAGE_SOURCE = (
    ROOT
    / "trino-benostreamdb"
    / "src"
    / "main"
    / "java"
    / "com"
    / "benostreamdb"
    / "trino"
    / "BenoStreamDBPageSource.java"
)
FLIGHT_SERVER = ROOT / "server" / "flight_sql" / "src" / "server.rs"

# The engine's index algorithms (IndexAlgorithm::all_names()).
INDEX_ALGORITHMS = {
    "hnsw",
    "hnsw_pq",
    "hnsw_tq4",
    "hnsw_tq8",
    "bm25",
    "bloom",
    "bitmap",
    "composite_bitmap",
    "csr_graph",
    "json_path",
}

# The engine's intercepted DDL / maintenance statements.
DDL_STATEMENTS = {
    "CREATE DATABASE",
    "CREATE SCHEMA",
    "CREATE TABLE",
    "CREATE INDEX",
    "DROP TABLE",
    "DROP SCHEMA",
    "DROP DATABASE",
    "ALTER TABLE",
    "OPTIMIZE TABLE",
    "TRUNCATE TABLE",
    "VACUUM",
}

# The engine's `ALTER TABLE ... EXECUTE <action>` procedures.
TABLE_ACTIONS = {
    "remove_orphan_files",
    "recover_indexes",
    "rollback",
    "rollback_to_snapshot",
    "preload_indexes",
    "verify_integrity",
    "checkpoint",
    "rewrite_data_files",
    "compact",
    "expire_snapshots",
}


def dbt_macros():
    names = set()
    for path in MACROS_DIR.glob("*.sql"):
        for match in re.finditer(
            r"^\{%-?\s*macro\s+([a-z_][a-z0-9_]*)\s*\(", path.read_text(), re.M
        ):
            name = match.group(1)
            if name.startswith(("default__", "benostreamdb__", "_")):
                continue
            names.add(name)
    return names


# ---------------------------------------------------------------------------
# Index algorithms
# ---------------------------------------------------------------------------


def test_index_algorithms_registered():
    assert set(benostreamdb.registered_index_algorithms()) == INDEX_ALGORITHMS


def test_python_accepts_every_index_algorithm(tmp_path):
    """Every engine index algorithm must be accepted by the Python parser
    (a string name or a dict config)."""
    table = benostreamdb.Table.from_arrow(
        f"file://{tmp_path}/idx",
        pa.table(
            {
                "a": [1, 2],
                "b": [3, 4],
                "source": [0, 1],
                "target": [1, 2],
                "text": ["x", "y"],
                "embedding": [[1.0, 0.0], [0.0, 1.0]],
            }
        ),
    )
    # A representative config per algorithm (string where the name is enough,
    # dict where the algorithm needs parameters).
    configs = {
        "hnsw": "hnsw",
        "hnsw_pq": "hnsw_pq",
        "hnsw_tq4": "hnsw_tq4",
        "hnsw_tq8": "hnsw_tq8",
        "bm25": "bm25",
        "bloom": "bloom",
        "bitmap": "bitmap",
        "composite_bitmap": {"type": "composite_bitmap", "columns": ["a", "b"]},
        "csr_graph": {"type": "graph", "src_column": "source", "dst_column": "target"},
        "json_path": {"type": "json_path", "paths": ["$.a"]},
    }
    for name in sorted(INDEX_ALGORITHMS):
        try:
            table.add_index("a", configs[name])
        except Exception as exc:  # noqa: BLE001 - the message is inspected
            assert "Unknown index type" not in str(exc), (
                f"index algorithm '{name}' is not accepted by the Python parser: {exc}"
            )


def test_dbt_exposes_index_creation():
    """dbt must have a macro that creates an index (so index types are usable)."""
    macros = dbt_macros()
    assert {"add_index", "create_index"} & macros, (
        f"dbt has no index-creation macro; macros: {sorted(macros)}"
    )


# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------


def test_ddl_statements_registered():
    assert set(benostreamdb.registered_ddl_statements()) == DDL_STATEMENTS


def test_ddl_reachable_from_each_sql_surface():
    """Every SQL surface forwards statements to the engine session, so the
    engine's DDL is reachable from all of them."""
    # Spark: the pass-through reader runs engine SQL via openQuery.
    spark_query = (
        ROOT
        / "spark-benostreamdb"
        / "src"
        / "main"
        / "scala"
        / "com"
        / "benostreamdb"
        / "spark"
        / "BenoStreamQueryTable.scala"
    )
    assert "openQuery" in spark_query.read_text()
    # Trino: the page source forwards SQL via openQuery.
    assert "openQuery(" in TRINO_PAGE_SOURCE.read_text()
    # Flight: built on the engine session.
    assert "BenoStreamSession" in FLIGHT_SERVER.read_text()
    # dbt: the adapter executes through the embedded session.
    connections = (
        ROOT / "dbt-benostreamdb" / "dbt" / "adapters" / "benostreamdb" / "connections.py"
    )
    assert "session.sql(" in connections.read_text()


# ---------------------------------------------------------------------------
# Stored procedures / table actions
# ---------------------------------------------------------------------------


def test_table_actions_registered():
    assert set(benostreamdb.registered_table_actions()) == TABLE_ACTIONS


def test_table_actions_reachable_via_execute(tmp_path):
    """`ALTER TABLE ... EXECUTE <action>` is the engine's procedure surface and
    must be callable from the embedded session."""
    table = benostreamdb.Table.from_arrow(
        f"file://{tmp_path}/t", pa.table({"id": [1, 2, 3]})
    )
    session = benostreamdb.Session()
    session.register("t", table)
    # A harmless, side-effect-free action.
    session.sql("ALTER TABLE t EXECUTE checkpoint")


# Spark 4.x stored procedures (BenoStreamProcedureCatalog.loadProcedure).
SPARK_PROCEDURES = {
    "add_index",
    "drop_index",
    "build_index",
    "rebuild_index",
    "compact",
    "show_indexes",
    "set_primary_key",
    "drop_primary_key",
    "regional_drift_search",
}


def test_spark_exposes_procedures():
    """Spark 4.x implements the native DSv2 `ProcedureCatalog`, exposing
    `CALL benostream.system.*`. The Spark 3.5 build uses the Iceberg shim."""
    catalog = SPARK_PROCEDURE_CATALOG.read_text()
    assert "extends BenoStreamCatalog with ProcedureCatalog" in catalog, (
        "spark-4 catalog no longer implements the native ProcedureCatalog"
    )
    # Parse the `case "<name>" =>` arms of loadProcedure.
    found = set(re.findall(r'case\s+"([a-z_]+)"', catalog))
    assert SPARK_PROCEDURES <= found, (
        f"Spark catalog is missing procedures: {sorted(SPARK_PROCEDURES - found)}"
    )
    # The Spark 3.5 variant must resolve the Iceberg shim (no native API pre-4.0).
    spark3 = (
        ROOT
        / "spark-benostreamdb"
        / "src"
        / "main"
        / "spark-3"
        / "com"
        / "benostreamdb"
        / "spark"
        / "BenoStreamCatalog.scala"
    ).read_text()
    assert "iceberg.catalog" in spark3, "spark-3 catalog must use the Iceberg shim"


def test_spark_procedures_are_tested():
    """The procedure surface must have a test, and it must cover every
    documented procedure so the catalog cannot regress silently."""
    assert SPARK_PROCEDURE_TEST.exists(), (
        "no Spark procedure catalog test; CALL benostream.system.* is untested"
    )
    test_src = SPARK_PROCEDURE_TEST.read_text()
    for name in sorted(SPARK_PROCEDURES):
        assert name in test_src, f"Spark procedure test does not cover '{name}'"


def test_spark_shared_procedure_name_matches_engine_action():
    """`compact` is the one procedure name shared with the engine's
    `ALTER TABLE ... EXECUTE <action>` surface; the names must agree."""
    assert "compact" in SPARK_PROCEDURES
    assert "compact" in TABLE_ACTIONS


# ---------------------------------------------------------------------------
# Reactive subscriptions (Theme 4) — must be reachable from every surface
# ---------------------------------------------------------------------------


def test_subscribe_reachable_from_all_surfaces():
    """`Table::subscribe()` is exposed as a SQL table function (universal), a
    native Python object, a Flight streaming ticket, and an MCP tool."""
    # Engine session registers the table function (so every SQL surface has it).
    assert "register_subscribe_table_functions" in (
        ROOT / "src" / "core" / "sql" / "session.rs"
    ).read_text()
    # Native Python binding.
    assert "PySubscription" in (ROOT / "src" / "python" / "subscribe.rs").read_text()
    # MCP tool.
    assert "subscribe_events" in (
        ROOT / "contrib" / "benostreamdb-mcp" / "src" / "server.rs"
    ).read_text()
    # Flight SQL streaming ticket.
    assert "benostreamdb.Subscribe" in (
        ROOT / "server" / "flight_sql" / "src" / "server.rs"
    ).read_text()


def test_gpu_context_wired_in_all_surfaces():
    """GPU device selection must be reachable from every connector, through the
    one shared core mapping (`context_from_device_str`)."""
    assert "context_from_device_str" in (
        ROOT / "src" / "core" / "index" / "gpu.rs"
    ).read_text()
    # Flight + MCP servers pin the device from the environment at startup.
    assert "apply_gpu_context_from_env" in (
        ROOT / "server" / "flight_sql" / "src" / "main.rs"
    ).read_text()
    assert "apply_gpu_context_from_env" in (
        ROOT / "contrib" / "benostreamdb-mcp" / "src" / "server.rs"
    ).read_text()
    # dbt adapter applies the profile `gpu_device`.
    assert "set_gpu_device" in (
        ROOT / "dbt-benostreamdb" / "dbt" / "adapters" / "benostreamdb" / "connections.py"
    ).read_text()
    # Spark + Trino JNI bridges use the shared mapping.
    assert "context_from_device_str" in (ROOT / "src" / "core" / "ffi.rs").read_text()
