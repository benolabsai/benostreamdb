package com.benostreamdb.trino;

import io.trino.testing.DistributedQueryRunner;
import io.trino.testing.MaterializedResult;
import io.trino.testing.TestingSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import io.trino.testing.MaterializedRow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Runs the connector inside a real Trino engine.
 *
 * <p>The native library must be on {@code java.library.path}; surefire points at
 * {@code ../target/debug}. If it is absent, the class is skipped rather than
 * failing unrelated builds.</p>
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestBenoStreamDBQueries {

    private DistributedQueryRunner runner;
    private Path warehouse;

    @BeforeAll
    void setUp() throws Exception {
        assumeTrue(BenoStreamDBJNIBridge.isLoaded(), "libbenostreamdb is not on java.library.path");
        warehouse = Files.createTempDirectory("benostream-trino-test-");
        runner = DistributedQueryRunner.builder(
                        TestingSession.testSessionBuilder()
                                .setCatalog("benostreamdb")
                                .setSchema("default")
                                .build())
                .build();
        runner.installPlugin(new BenoStreamDBPlugin());
        runner.createCatalog(
                "benostreamdb",
                "benostreamdb",
                Map.of("benostream.warehouse", warehouse.toUri().toString()));
    }

    @AfterAll
    void tearDown() {
        if (runner != null) {
            runner.close();
        }
    }

    @Test
    void catalogIsVisible() {
        assertTrue(runner.execute("SHOW CATALOGS").getOnlyColumnAsSet().contains("benostreamdb"));
    }

    @Test
    void createsInsertsAndScans() {
        runner.execute("CREATE TABLE benostreamdb.default.docs (id bigint, title varchar)");
        runner.execute(
                "INSERT INTO benostreamdb.default.docs VALUES "
                        + "(1, 'graph rag'), (2, 'vector search')");

        MaterializedResult result = runner.execute(
                "SELECT id, title FROM benostreamdb.default.docs ORDER BY id");
        assertEquals(2, result.getRowCount());
        assertEquals(1L, result.getMaterializedRows().get(0).getField(0));
        assertEquals("graph rag", result.getMaterializedRows().get(0).getField(1));
    }

    @Test
    void addIndexProcedureRuns() {
        runner.execute("CREATE TABLE benostreamdb.default.indexed (id bigint, title varchar)");
        runner.execute(
                "INSERT INTO benostreamdb.default.indexed VALUES (1, 'graph rag')");

        runner.execute("CALL benostreamdb.system.add_index('default.indexed', 'title', 'bm25')");
    }

    @Test
    void allSparkProceduresAreRegistered() {
        // Trino has no SHOW PROCEDURES statement, so assert through the connector
        // API instead of SQL. This also fails loudly if a handle stops binding.
        var connector = new BenoStreamDBConnectorFactory().create(
                "benostreamdb",
                Map.of("benostream.warehouse", warehouse.toUri().toString()),
                null);
        java.util.Set<String> registered = new java.util.HashSet<>();
        connector.getProcedures().forEach(procedure -> registered.add(procedure.getName()));
        // Name-for-name parity with Spark's BenoStreamProcedureCatalog, including
        // the `optimize` alias Spark accepts for `compact`.
        assertTrue(
                registered.containsAll(java.util.List.of(
                        "add_index",
                        "build_index",
                        "rebuild_index",
                        "drop_index",
                        "compact",
                        "optimize",
                        "set_primary_key",
                        "drop_primary_key")),
                "registered procedures were: " + registered);
    }

    @Test
    void maintenanceProceduresRun() {
        runner.execute("CREATE TABLE benostreamdb.default.maint (id bigint, title varchar)");
        runner.execute("INSERT INTO benostreamdb.default.maint VALUES (1, 'graph rag')");

        runner.execute("CALL benostreamdb.system.add_index('default.maint', 'title', 'bm25')");
        runner.execute("CALL benostreamdb.system.rebuild_index('default.maint', 'title')");
        runner.execute("CALL benostreamdb.system.drop_index('default.maint', 'title')");
        runner.execute("CALL benostreamdb.system.set_primary_key('default.maint', 'id')");
        runner.execute("CALL benostreamdb.system.drop_primary_key('default.maint', 'id')");
        runner.execute("CALL benostreamdb.system.compact('default.maint')");
        runner.execute("CALL benostreamdb.system.optimize('default.maint')");

        assertEquals(
                1,
                runner.execute("SELECT id FROM benostreamdb.default.maint").getRowCount());
    }

    @Test
    void scalarFunctionsAppearInShowFunctions() {
        java.util.Set<String> registered = new java.util.HashSet<>();
        for (io.trino.testing.MaterializedRow row : runner.execute("SHOW FUNCTIONS").getMaterializedRows()) {
            registered.add(String.valueOf(row.getField(0)));
        }
        // Five are ours. cosine_distance and dot_product are Trino 484
        // built-ins with the same signature, and plugin functions may not
        // shadow built-ins, so those two are intentionally not registered here.
        java.util.Set<String> expected = new java.util.TreeSet<>(java.util.List.of(
                "l2_distance", "vector_distance", "sparse_dot_product",
                "hybrid_score", "reciprocal_rank_fusion"));
        java.util.Set<String> missing = new java.util.TreeSet<>(expected);
        missing.removeAll(registered);
        assertTrue(missing.isEmpty(), "not visible in SHOW FUNCTIONS: " + missing);

        // The two built-ins still resolve, so Spark's full surface is available.
        assertTrue(registered.containsAll(java.util.List.of("cosine_distance", "dot_product")),
                "built-in distance functions missing: " + registered);
    }

    @Test
    void scalarDistanceFunctions() {
        // cosine_distance and dot_product resolve to Trino built-ins; the
        // values here are asserted through our own l2/vector/sparse/hybrid/rrf.
        assertEquals(5.0, scalar("SELECT l2_distance("
                + "CAST(ARRAY[0.0, 0.0] AS array(double)), "
                + "CAST(ARRAY[3.0, 4.0] AS array(double)))"));
        assertEquals(11.0, scalar("SELECT dot_product("
                + "CAST(ARRAY[1.0, 2.0] AS array(double)), "
                + "CAST(ARRAY[3.0, 4.0] AS array(double)))"));
        assertEquals(-11.0, scalar("SELECT vector_distance("
                + "CAST(ARRAY[1.0, 2.0] AS array(double)), "
                + "CAST(ARRAY[3.0, 4.0] AS array(double)), 'dot')"));
        assertEquals(0.25, scalar("SELECT hybrid_score(1.0, 0.0, 0.25)"));
        assertEquals(6.0, scalar("SELECT sparse_dot_product("
                + "CAST(ARRAY[0, 5] AS array(bigint)), CAST(ARRAY[1.5, 2.0] AS array(double)), "
                + "CAST(ARRAY[5, 9] AS array(bigint)), CAST(ARRAY[3.0, 4.0] AS array(double)))"));
        assertEquals(1.0 / 61 + 1.0 / 62, scalar(
                "SELECT reciprocal_rank_fusion(CAST(ARRAY[1, 2] AS array(integer)), 60)"));
    }

    private double scalar(String sql) {
        Object value = runner.execute(sql).getMaterializedRows().get(0).getField(0);
        assertTrue(value instanceof Number, "expected a numeric result from: " + sql);
        return ((Number) value).doubleValue();
    }

    @Test
    void trinoBuiltinVectorFunctionsSemantics() {
        // This plugin contributes l2_distance, vector_distance,
        // sparse_dot_product, hybrid_score and reciprocal_rank_fusion. It cannot
        // contribute cosine_distance or dot_product: Trino registers functions
        // globally and rejects a duplicate signature with "Function already
        // registered", so the connector cannot overload a built-in per schema.
        // These calls therefore pin Trino's own behaviour, which matters because
        // users will reach for these names on our tables.
        assertEquals(0.0, scalar("SELECT cosine_distance("
                + "CAST(ARRAY[1.0, 0.0] AS array(double)), "
                + "CAST(ARRAY[1.0, 0.0] AS array(double)))"));
        assertEquals(1.0, scalar("SELECT cosine_distance("
                + "CAST(ARRAY[1.0, 0.0] AS array(double)), "
                + "CAST(ARRAY[0.0, 1.0] AS array(double)))"));
        // Standard 1 - cos: opposing vectors give 2.0, i.e. Trino does not clamp
        // at 1 (it matches Spark's floor-only behaviour in this case).
        assertEquals(2.0, scalar("SELECT cosine_distance("
                + "CAST(ARRAY[1.0, 0.0] AS array(double)), "
                + "CAST(ARRAY[-1.0, 0.0] AS array(double)))"));
        assertEquals(11.0, scalar("SELECT dot_product("
                + "CAST(ARRAY[1.0, 2.0] AS array(double)), "
                + "CAST(ARRAY[3.0, 4.0] AS array(double)))"));
        // Spark's l2_distance equivalent.
        assertEquals(5.0, scalar("SELECT euclidean_distance("
                + "CAST(ARRAY[0.0, 0.0] AS array(double)), "
                + "CAST(ARRAY[3.0, 4.0] AS array(double)))"));
    }

    @Test
    void vectorSearchEndToEnd() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.vectors "
                        + "(id bigint, title varchar, embedding array(real))");
        runner.execute(
                "INSERT INTO benostreamdb.default.vectors VALUES "
                        + "(1, 'graph rag', CAST(ARRAY[1.0, 0.0, 0.0] AS array(real))), "
                        + "(2, 'vector search', CAST(ARRAY[0.0, 1.0, 0.0] AS array(real))), "
                        + "(3, 'graph database', CAST(ARRAY[0.9, 0.1, 0.0] AS array(real)))");
        runner.execute(
                "CALL benostreamdb.system.add_index('default.vectors', 'embedding', 'hnsw')");
        // add_index registers the index; building it happens on a detached
        // background task, so a query issued immediately afterwards can see no
        // index at all. build_index makes the two-phase contract explicit and
        // the test deterministic.
        runner.execute("CALL benostreamdb.system.build_index('default.vectors')");

        MaterializedResult result = runner.execute(
                "SELECT id FROM TABLE(system.vector_search("
                        + "table_name => 'default.vectors', column => 'embedding', "
                        + "query => '[1.0, 0.0, 0.0]', k => 1))");
        assertEquals(1, result.getRowCount());
        assertEquals(1L, result.getMaterializedRows().get(0).getField(0));
    }

    // Pushdown is asserted at the layer that owns it rather than through
    // EXPLAIN, whose row shape in DistributedQueryRunner is not stable enough to
    // parse: VectorSearchSqlTest pins the generated KNN statement (dist_cosine,
    // ORDER BY, LIMIT), and TrinoConnectorTest pins the filter that travels in
    // the split.

    @Test
    void allTableFunctionsAreRegistered() {
        var connector = new BenoStreamDBConnectorFactory().create(
                "benostreamdb",
                Map.of("benostream.warehouse", warehouse.toUri().toString()),
                null);
        java.util.Set<String> registered = new java.util.HashSet<>();
        connector.getTableFunctions().forEach(function -> registered.add(function.getName()));
        assertTrue(
                registered.containsAll(java.util.List.of(
                        "vector_search",
                        "hybrid_search",
                        "graph_neighbors",
                        "graph_shortest_path",
                        "graph_subgraph",
                        "list_indexes",
                        "regional_drift_search",
                        "graph_rag_search")),
                "registered table functions were: " + registered);
    }

    @Test
    void graphRagSearchJoinsVectorAndGraph() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.rag_docs "
                        + "(id bigint, title varchar, embedding array(real))");
        runner.execute(
                "INSERT INTO benostreamdb.default.rag_docs VALUES "
                        + "(1, 'graph rag', CAST(ARRAY[1.0, 0.0] AS array(real))), "
                        + "(2, 'vector search', CAST(ARRAY[0.0, 1.0] AS array(real))), "
                        + "(3, 'graph database', CAST(ARRAY[0.9, 0.1] AS array(real)))");
        runner.execute(
                "CALL benostreamdb.system.add_index('default.rag_docs', 'embedding', 'hnsw')");
        runner.execute("CALL benostreamdb.system.build_index('default.rag_docs')");

        runner.execute(
                "CREATE TABLE benostreamdb.default.rag_edges (source bigint, target bigint)");
        runner.execute("INSERT INTO benostreamdb.default.rag_edges VALUES (1, 2), (2, 3)");

        // Two tables in one statement: documents ranked by distance, joined to a
        // traversal of the separately registered edge table.
        MaterializedResult result = runner.execute(
                "SELECT id, hop FROM TABLE(system.graph_rag_search("
                        + "table_name => 'default.rag_docs', "
                        + "edge_table => 'default.rag_edges', "
                        + "vector_column => 'embedding', query_vector => '1.0, 0.0', "
                        + "seeds => '1', id_column => 'id', hops => 1, k => 2))");

        assertTrue(result.getRowCount() >= 1, "expected at least one reachable document");
        assertEquals(1L, result.getMaterializedRows().get(0).getField(0));
    }

    @Test
    void regionalDriftSearchRunsOnAnEmptyGraph() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.drift_edges (source bigint, target bigint)");
        // An empty graph must yield no rows rather than fail: the engine returns
        // a negative status, which the page source treats as "no results".
        MaterializedResult result = runner.execute(
                "SELECT node_id FROM TABLE(system.regional_drift_search("
                        + "table_name => 'default.drift_edges', query => 'related nodes', "
                        + "seeds => '1'))");
        assertEquals(0, result.getRowCount());
    }

    @Test
    void listIndexesReturnsEngineMetadata() {
        runner.execute("CREATE TABLE benostreamdb.default.lidx (id bigint, title varchar)");
        runner.execute("INSERT INTO benostreamdb.default.lidx VALUES (1, 'graph rag')");
        runner.execute("CALL benostreamdb.system.add_index('default.lidx', 'title', 'bm25')");
        runner.execute("CALL benostreamdb.system.build_index('default.lidx')");

        // Spark's show_indexes calls the same JNI entry point but discards the
        // rows; here they are actually returned.
        MaterializedResult result = runner.execute(
                "SELECT column_name, index_category, algorithm, index_id, format_version "
                        + "FROM TABLE(system.list_indexes(table_name => 'default.lidx'))");

        assertTrue(result.getRowCount() >= 1, "expected at least one index row");
        assertEquals("title", result.getMaterializedRows().get(0).getField(0));
    }

    @Test
    void graphNeighborsEndToEnd() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.edges (source bigint, target bigint)");
        runner.execute(
                "INSERT INTO benostreamdb.default.edges VALUES (1, 2), (2, 3), (3, 4)");

        MaterializedResult result = runner.execute(
                "SELECT node, hop FROM TABLE(system.graph_neighbors("
                        + "table_name => 'default.edges', seeds => '1', hops => 2)) ORDER BY node");
        assertEquals(3, result.getRowCount());
        assertEquals(1L, result.getMaterializedRows().get(0).getField(0));
        assertEquals(0, ((Number) result.getMaterializedRows().get(0).getField(1)).intValue());
    }

    /**
     * Run a query until it returns rows (bounded).
     *
     * <p>Index builds are engine-side and can still be finishing when the
     * procedure returns under load, so a hybrid/vector assertion is an
     * eventual-consistency check, not an instant one.</p>
     */
    private MaterializedResult awaitRows(String sql) {
        MaterializedResult result = runner.execute(sql);
        for (int attempt = 0; attempt < 40 && result.getRowCount() == 0; attempt++) {
            try {
                Thread.sleep(250);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
            result = runner.execute(sql);
        }
        return result;
    }

    /**
     * Highest non-null `rrf_score` in the result (0 when there is none).
     *
     * <p>Index builds are engine-side and asynchronous, and a hybrid search that
     * runs while the BM25/vector sidecars are still being attached returns rows
     * with no fused score yet. Waiting for a row count alone is not enough: the
     * count can be non-zero while every score is still zero.</p>
     */
    private static double maxScore(MaterializedResult result) {
        double max = 0;
        for (MaterializedRow row : result.getMaterializedRows()) {
            if (row.getFieldCount() < 2) {
                continue;
            }
            Object score = row.getField(1);
            if (score instanceof Number) {
                max = Math.max(max, ((Number) score).doubleValue());
            }
        }
        return max;
    }

    @Test
    void hybridSearchEndToEnd() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.hybrid "
                        + "(id bigint, title varchar, embedding array(real))");
        runner.execute(
                "INSERT INTO benostreamdb.default.hybrid VALUES "
                        + "(1, 'graph rag', CAST(ARRAY[1.0, 0.0, 0.0] AS array(real))), "
                        + "(2, 'vector search', CAST(ARRAY[0.0, 1.0, 0.0] AS array(real))), "
                        + "(3, 'graph database', CAST(ARRAY[0.9, 0.1, 0.0] AS array(real)))");
        runner.execute(
                "CALL benostreamdb.system.add_index('default.hybrid', 'embedding', 'hnsw')");
        runner.execute(
                "CALL benostreamdb.system.add_index('default.hybrid', 'title', 'bm25')");
        runner.execute("CALL benostreamdb.system.build_index('default.hybrid')");

        String sql =
                "SELECT id, rrf_score FROM TABLE(system.hybrid_search("
                        + "table_name => 'default.hybrid', text_column => 'title', "
                        + "query_text => 'graph', vector_column => 'embedding', "
                        + "query_vector => '[1.0, 0.0, 0.0]', k => 2))";
        MaterializedResult result = runner.execute(sql);
        for (int attempt = 0; attempt < 40 && maxScore(result) <= 0; attempt++) {
            try {
                Thread.sleep(250);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
            result = runner.execute(sql);
        }
        assertTrue(result.getRowCount() >= 1);
        assertTrue(
                maxScore(result) > 0,
                "expected a fused score, got " + result.getMaterializedRows());
    }

    @Test
    void typeMatrixRoundTrip() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.types "
                        + "(id bigint, i integer, r real, d double, b boolean, "
                        + "s varchar, a array(real), dt date)");
        runner.execute(
                "INSERT INTO benostreamdb.default.types VALUES "
                        + "(1, 2, REAL '3.5', 4.5, true, 'x', "
                        + "CAST(ARRAY[1.0, 2.0] AS array(real)), DATE '2026-01-02')");

        MaterializedResult result = runner.execute(
                "SELECT id, i, r, d, b, s, a, dt FROM benostreamdb.default.types");
        assertEquals(1, result.getRowCount());
        assertEquals(1L, result.getMaterializedRows().get(0).getField(0));
        assertEquals(true, result.getMaterializedRows().get(0).getField(4));
        assertEquals(2, ((java.util.List<?>) result.getMaterializedRows().get(0).getField(6)).size());
    }

    @Test
    void vectorSearchRejectsMissingColumn() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.bad_vectors "
                        + "(id bigint, embedding array(real))");
        org.junit.jupiter.api.Assertions.assertThrows(
                io.trino.testing.QueryFailedException.class,
                () -> runner.execute(
                        "SELECT id FROM TABLE(system.vector_search("
                                + "table_name => 'default.bad_vectors', column => 'missing', "
                                + "query => '[1.0]', k => 1))"));
    }

    @Test
    void hybridSearchHonoursFilter() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.filtered "
                        + "(id bigint, title varchar, embedding array(real))");
        runner.execute(
                "INSERT INTO benostreamdb.default.filtered VALUES "
                        + "(1, 'graph rag', CAST(ARRAY[1.0, 0.0] AS array(real))), "
                        + "(2, 'graph database', CAST(ARRAY[0.9, 0.1] AS array(real))), "
                        + "(3, 'vector search', CAST(ARRAY[0.0, 1.0] AS array(real)))");
        runner.execute(
                "CALL benostreamdb.system.add_index('default.filtered', 'embedding', 'hnsw')");
        runner.execute("CALL benostreamdb.system.add_index('default.filtered', 'title', 'bm25')");
        runner.execute("CALL benostreamdb.system.build_index('default.filtered')");

        // The predicate must be applied to both retrieval paths, so only id = 2
        // can survive the fusion.
        MaterializedResult filtered = awaitRows(
                "SELECT id, rrf_score FROM TABLE(system.hybrid_search("
                        + "table_name => 'default.filtered', text_column => 'title', "
                        + "query_text => 'graph', vector_column => 'embedding', "
                        + "query_vector => '[1.0, 0.0]', k => 5, filter => 'id = 2'))");
        assertTrue(filtered.getRowCount() >= 1, "the matching row must be returned");
        for (MaterializedRow row : filtered.getMaterializedRows()) {
            assertEquals(2L, ((Number) row.getField(0)).longValue(), "filtered row escaped the predicate");
        }

        // A predicate that matches nothing must return no rows rather than the
        // unfiltered neighbours.
        MaterializedResult empty = runner.execute(
                "SELECT id FROM TABLE(system.hybrid_search("
                        + "table_name => 'default.filtered', text_column => 'title', "
                        + "query_text => 'graph', vector_column => 'embedding', "
                        + "query_vector => '[1.0, 0.0]', k => 5, filter => 'id > 100'))");
        assertEquals(0, empty.getRowCount(), "impossible filter must not fall back to unfiltered rows");
    }

    @Test
    void vectorSearchHonoursMetric() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.metrics "
                        + "(id bigint, embedding array(real))");
        runner.execute(
                "INSERT INTO benostreamdb.default.metrics VALUES "
                        + "(1, CAST(ARRAY[1.0, 0.0] AS array(real))), "
                        + "(2, CAST(ARRAY[0.0, 1.0] AS array(real)))");

        MaterializedResult l2 = runner.execute(
                "SELECT id, distance FROM TABLE(system.vector_search("
                        + "table_name => 'default.metrics', column => 'embedding', "
                        + "query => '[1.0, 0.0]', k => 2, metric => 'l2'))");
        assertEquals(2, l2.getRowCount());
        assertEquals(1L, ((Number) l2.getMaterializedRows().get(0).getField(0)).longValue());
        assertEquals(0.0, ((Number) l2.getMaterializedRows().get(0).getField(1)).doubleValue(), 1e-6);

        org.junit.jupiter.api.Assertions.assertThrows(
                io.trino.testing.QueryFailedException.class,
                () -> runner.execute(
                        "SELECT id FROM TABLE(system.vector_search("
                                + "table_name => 'default.metrics', column => 'embedding', "
                                + "query => '[1.0, 0.0]', k => 1, metric => 'levenshtein'))"));
    }

    @Test
    void listIndexesReturnsOneRowPerIndex() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.listed "
                        + "(id bigint, title varchar, embedding array(real))");
        runner.execute(
                "INSERT INTO benostreamdb.default.listed VALUES "
                        + "(1, 'graph rag', CAST(ARRAY[1.0, 0.0] AS array(real)))");
        runner.execute("CALL benostreamdb.system.add_index('default.listed', 'embedding', 'hnsw')");
        runner.execute("CALL benostreamdb.system.add_index('default.listed', 'title', 'bm25')");
        runner.execute("CALL benostreamdb.system.build_index('default.listed')");

        MaterializedResult indexes = runner.execute(
                "SELECT column_name, index_category, algorithm, index_id, format_version "
                        + "FROM TABLE(system.list_indexes(table_name => 'default.listed'))");
        assertTrue(indexes.getRowCount() >= 1, "built indexes must be listed");
        // The five columns are the connector's contract with Spark's
        // `show_indexes`; a shape change must fail here.
        assertEquals(5, indexes.getTypes().size());
        assertEquals("varchar", indexes.getTypes().get(0).getDisplayName());
        assertEquals("bigint", indexes.getTypes().get(4).getDisplayName());
    }

    @Test
    void vectorTransformScalarFunctions() {
        MaterializedResult result = runner.execute(
                "SELECT vector_dims(CAST(ARRAY[1.0, 2.0, 3.0] AS array(double))) AS dims, "
                        + "vector_norm(CAST(ARRAY[3.0, 4.0] AS array(double))) AS norm, "
                        + "l2_normalize(CAST(ARRAY[3.0, 4.0] AS array(double))) AS unit, "
                        + "subvector(CAST(ARRAY[10.0, 20.0, 30.0] AS array(double)), 2, 2) AS slice, "
                        + "vector_add(CAST(ARRAY[1.0, 2.0] AS array(double)), "
                        + "CAST(ARRAY[3.0, 4.0] AS array(double))) AS sum, "
                        + "vector_sub(CAST(ARRAY[1.0, 2.0] AS array(double)), "
                        + "CAST(ARRAY[3.0, 4.0] AS array(double))) AS diff");

        MaterializedRow row = result.getMaterializedRows().get(0);
        assertEquals(3L, ((Number) row.getField(0)).longValue());
        assertEquals(5.0, ((Number) row.getField(1)).doubleValue(), 1e-9);
        assertEquals(java.util.List.of(0.6, 0.8), row.getField(2));
        assertEquals(java.util.List.of(20.0, 30.0), row.getField(3));
        assertEquals(java.util.List.of(4.0, 6.0), row.getField(4));
        assertEquals(java.util.List.of(-2.0, -2.0), row.getField(5));
    }
}
