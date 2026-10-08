"""Function-surface parity across the core, Python, dbt, Trino, and Spark.

The Rust core is the source of truth: `benostreamdb.registered_functions()`
returns every custom UDF/UDAF the engine registers with DataFusion. Adding a
function to the core without exposing it in the other surfaces fails here, so
the surfaces cannot silently drift apart.

Surfaces:
  * core   — `src/core/sql/udf/mod.rs::registered_function_names()`
  * python — the embedded session (every function must resolve in SQL)
  * dbt    — `dbt-benostreamdb/dbt/include/benostreamdb/macros/*.sql`
  * spark  — `BenoStreamCatalogFunctions` (DSv2 `FunctionCatalog`)
  * trino  — the connector forwards SQL to the engine session (`openQuery`);
             Trino has no connector function SPI, so its user-facing function
             surface is empty by design (tracked in TRINO_KNOWN_GAPS)
  * flight — the Arrow Flight SQL server is built on `BenoStreamSession`, so it
             inherits the full function surface (and the graph table functions)
             with no per-surface wiring
"""

import pathlib
import re

import pytest

import benostreamdb

ROOT = pathlib.Path(__file__).resolve().parents[2]
MACROS_DIR = ROOT / "dbt-benostreamdb" / "dbt" / "include" / "benostreamdb" / "macros"
SPARK_FUNCTIONS_SCALA = (
    ROOT
    / "spark-benostreamdb"
    / "src"
    / "main"
    / "scala"
    / "com"
    / "benostreamdb"
    / "spark"
    / "BenoStreamCatalogFunctions.scala"
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

# Graph traversal table functions (`FROM graph_*(...)`). These are a separate
# surface from the scalar/aggregate UDFs: they are registered as DataFusion
# table functions and reachable from every SQL surface (Python, dbt, Spark
# pass-through, Trino, Flight SQL).
GRAPH_TABLE_FUNCTIONS = {
    "graph_neighbors",
    "graph_shortest_path",
    "graph_all_shortest_paths",
    "graph_subgraph",
    "graph_connecting_paths",
}

# Engine function -> dbt macro that exposes it. A macro may cover several
# engine functions (e.g. `community_detect` covers louvain/leiden/label
# propagation, `node_similarity` covers jaccard/preferential attachment).
DBT_MACRO_FOR = {
    # --- vector distances ---
    "dist_l2": "l2_distance",
    "dist_cosine": "cosine_distance",
    "dist_ip": "inner_product",
    "dist_l1": "l1_distance",
    "dist_hamming": "hamming_distance",
    "dist_jaccard": "jaccard_distance",
    # --- vector transforms ---
    "vector_add": "vector_add",
    "vector_sub": "vector_sub",
    "vector_mul": "vector_mul",
    "vector_concat": "vector_concat",
    "vector_dims": "vector_dims",
    "vector_norm": "vector_norm",
    "l2_normalize": "l2_normalize",
    "binary_quantize": "binary_quantize",
    "subvector": "subvector",
    "vector_to_binary": "vector_to_binary",
    # --- sparse ---
    "vector_to_sparse": "vector_to_sparse",
    "sparse_to_vector": "sparse_to_vector",
    # --- vector aggregates ---
    "vector_sum": "vector_sum",
    "vector_avg": "vector_avg",
    "centroid": "centroid",
    "vector_median": "vector_median",
    "vector_stddev": "vector_stddev",
    "vector_min": "vector_min",
    "vector_max": "vector_max",
    # --- lexical ---
    "bm25_score": "bm25_score",
    "tf_idf": "tf_idf",
    # --- json ---
    "json_extract_path": "json_extract_path",
    "json_extract_path_text": "json_extract_path_text",
    "json_contains": "json_contains",
    "json_exists": "json_exists",
    "json_typeof": "json_typeof",
    "json_path_exists": "json_path_exists",
    "json_path_query": "json_path_query",
    # --- graph ---
    "graph_pagerank": "pagerank",
    "graph_personalized_pagerank": "personalized_pagerank",
    "graph_louvain_communities": "community_detect",
    "graph_leiden_communities": "community_detect",
    "graph_label_propagation": "community_detect",
    "graph_neighbors": "graph_neighbors",
    "graph_subgraph": "subgraph",
    "graph_connecting_paths": "connecting_paths",
    "graph_shortest_path": "shortest_path",
    "graph_all_shortest_paths": "all_shortest_paths",
    "graph_connected_components": "connected_components",
    "graph_strongly_connected_components": "connected_components",
    "graph_degree_centrality": "degree_centrality",
    "graph_closeness_centrality": "closeness_centrality",
    "graph_betweenness_centrality": "betweenness_centrality",
    "graph_jaccard_coefficient": "node_similarity",
    "graph_preferential_attachment": "node_similarity",
    "graph_triangle_count": "triangle_count",
    "graph_modularity": "modularity",
    "drift_search": "drift_search",
    "regional_drift": "regional_drift",
}

# Engine functions exposed by Spark under a different catalog-function name.
SPARK_ALIAS = {
    "dist_l2": "l2_distance",
    "dist_cosine": "cosine_distance",
    "dist_ip": "inner_product",
    "dist_l1": "l1_distance",
    "dist_hamming": "hamming_distance",
    "dist_jaccard": "jaccard_distance",
}

# Engine functions reachable from Spark only through the pass-through reader
# (`option("query", sql)`), because Spark's DSv2 `FunctionCatalog` has no
# aggregate/UDAF equivalent: the vector aggregates and the graph UDAFs.
SPARK_PASSTHROUGH_ONLY = {
    "vector_sum", "vector_avg", "centroid", "vector_median", "vector_stddev",
    "vector_min", "vector_max",
    "vector_to_sparse", "sparse_to_vector",
    "graph_pagerank", "graph_personalized_pagerank", "graph_louvain_communities",
    "graph_leiden_communities", "graph_label_propagation", "graph_neighbors",
    "graph_subgraph", "graph_connecting_paths", "graph_shortest_path",
    "graph_all_shortest_paths", "graph_connected_components",
    "graph_strongly_connected_components", "graph_degree_centrality",
    "graph_closeness_centrality", "graph_betweenness_centrality",
    "graph_jaccard_coefficient", "graph_preferential_attachment",
    "graph_triangle_count", "graph_modularity", "drift_search", "regional_drift",
}

# Trino has no connector function SPI: the connector only forwards generated
# SQL (projection + pushed-down filters) to the engine session, so no engine
# function is callable from Trino SQL. Every function is therefore a known gap.
TRINO_KNOWN_GAPS = set(DBT_MACRO_FOR)


def engine_functions():
    return set(benostreamdb.registered_functions())


def dbt_macros():
    """User-facing macro names (dispatch triplets collapse to one name)."""
    names = set()
    for path in MACROS_DIR.glob("*.sql"):
        for match in re.finditer(r"^\{%-?\s*macro\s+([a-z_][a-z0-9_]*)\s*\(", path.read_text(), re.M):
            name = match.group(1)
            if name.startswith(("default__", "benostreamdb__", "_")):
                continue
            names.add(name)
    return names


def spark_functions():
    """Names listed in `BenoStreamCatalogFunctions.names`."""
    text = SPARK_FUNCTIONS_SCALA.read_text()
    # Match the Seq(...) up to the line that closes it (a bare `)`), then drop
    # `//` comments: they contain parentheses that would otherwise truncate the
    # non-greedy match before any quoted name is seen.
    block = re.search(
        r"val names:\s*Seq\[String\]\s*=\s*Seq\((.*?)\n\s*\)", text, re.S
    )
    assert block, "could not find the Spark catalog function list"
    body = "\n".join(line.split("//", 1)[0] for line in block.group(1).splitlines())
    return set(re.findall(r'"([a-z_][a-z0-9_]*)"', body))


def test_core_functions_are_all_mapped():
    """A new core UDF must be added to DBT_MACRO_FOR (and thus to a macro)."""
    unmapped = engine_functions() - set(DBT_MACRO_FOR)
    assert not unmapped, (
        f"core functions with no dbt macro mapping: {sorted(unmapped)} — "
        "add a macro and map it in DBT_MACRO_FOR"
    )


def test_mapping_has_no_stale_entries():
    """DBT_MACRO_FOR must not reference functions the core no longer registers."""
    stale = set(DBT_MACRO_FOR) - engine_functions()
    assert not stale, f"DBT_MACRO_FOR references unknown core functions: {sorted(stale)}"


def test_dbt_macros_exist():
    macros = dbt_macros()
    missing = {fn: macro for fn, macro in DBT_MACRO_FOR.items() if macro not in macros}
    assert not missing, f"dbt macros missing for core functions: {missing}"


def test_python_resolves_every_core_function():
    """Every core function must resolve in the embedded session."""
    session = benostreamdb.Session()
    unresolved = []
    for fn in sorted(engine_functions()):
        try:
            session.sql(f"select {fn}()")
        except Exception as exc:  # noqa: BLE001 - any error is inspected
            message = str(exc)
            if "Invalid function" in message or "unknown function" in message.lower():
                unresolved.append(fn)
    assert not unresolved, f"core functions not registered in the session: {unresolved}"


def test_spark_surface_covers_core_or_is_a_known_gap():
    """Every core function is a Spark catalog function, an alias, or pass-through."""
    spark = spark_functions()
    # `SPARK_ALIAS` keys are engine names exposed under a different Spark name.
    uncovered = engine_functions() - spark - set(SPARK_ALIAS) - SPARK_PASSTHROUGH_ONLY
    assert not uncovered, (
        f"core functions neither exposed by Spark nor listed in "
        f"SPARK_PASSTHROUGH_ONLY: {sorted(uncovered)}"
    )


def test_spark_aliases_are_real():
    spark = spark_functions()
    for engine_fn, spark_fn in SPARK_ALIAS.items():
        assert engine_fn in engine_functions(), f"SPARK_ALIAS maps unknown core function {engine_fn}"
        assert spark_fn in spark, f"SPARK_ALIAS target {spark_fn} is not a Spark catalog function"


def test_spark_passthrough_only_are_real():
    """Pass-through entries must be real core functions not already in the catalog."""
    spark = spark_functions()
    bogus = SPARK_PASSTHROUGH_ONLY - engine_functions()
    assert not bogus, f"SPARK_PASSTHROUGH_ONLY references unknown functions: {sorted(bogus)}"
    already = SPARK_PASSTHROUGH_ONLY & spark
    assert not already, (
        f"SPARK_PASSTHROUGH_ONLY lists functions Spark already exposes: {sorted(already)}"
    )


def test_spark_passthrough_reader_exists():
    """The pass-through reader is what makes aggregates/graph UDAFs reachable."""
    source = (
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
    assert source.exists(), "the Spark pass-through reader (BenoStreamQueryTable) is missing"
    text = source.read_text()
    assert "openQuery" in text, "the pass-through reader no longer runs engine SQL"


def test_trino_forwards_sql_to_the_engine_session():
    """Trino's function surface is the engine's SQL dialect via `openQuery`."""
    text = TRINO_PAGE_SOURCE.read_text()
    assert "openQuery(" in text, (
        "the Trino page source no longer forwards SQL to the engine session; "
        "the function-parity assumptions in this test need revisiting"
    )


def test_trino_gaps_are_documented():
    """Every core function is a documented Trino gap (no connector function SPI)."""
    undocumented = engine_functions() - TRINO_KNOWN_GAPS
    assert not undocumented, (
        f"core functions not accounted for in TRINO_KNOWN_GAPS: {sorted(undocumented)}"
    )


def test_graph_table_functions_are_registered():
    """The graph traversal table functions are registered in the engine."""
    registered = set(benostreamdb.registered_table_functions())
    assert registered == GRAPH_TABLE_FUNCTIONS, (
        f"graph table functions drifted: registered={sorted(registered)} "
        f"expected={sorted(GRAPH_TABLE_FUNCTIONS)}"
    )


def test_graph_table_functions_resolve_in_the_session():
    """Each table function is callable; a missing table is a *table* error, not
    an unknown-function error (which would mean it was never registered)."""
    session = benostreamdb.Session()
    # Per-function argument shapes (seeds are strings; path endpoints are ints).
    calls = {
        "graph_neighbors": "('__no_such_table__', '1', 1)",
        "graph_subgraph": "('__no_such_table__', '1', 1)",
        "graph_shortest_path": "('__no_such_table__', 1, 2)",
        "graph_all_shortest_paths": "('__no_such_table__', 1, 2)",
        "graph_connecting_paths": "('__no_such_table__', '1')",
    }
    for fn in sorted(GRAPH_TABLE_FUNCTIONS):
        try:
            session.sql(f"select * from {fn}{calls[fn]}")
        except Exception as exc:  # noqa: BLE001 - the message is inspected
            message = str(exc)
            assert "not found" in message and "table" in message.lower(), (
                f"{fn} did not resolve as a table function: {message}"
            )
        else:
            raise AssertionError(f"{fn} unexpectedly succeeded on a missing table")


def test_flight_surface_inherits_the_engine_session():
    """The Flight SQL server is built on `BenoStreamSession`, so it exposes the
    full function surface (and the graph table functions) with no extra wiring."""
    text = FLIGHT_SERVER.read_text()
    assert "BenoStreamSession" in text, (
        "the Flight SQL server no longer uses BenoStreamSession; the function "
        "parity assumptions in this test need revisiting"
    )
