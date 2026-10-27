package com.benostreamdb.trino;

import io.trino.plugin.tpch.TpchPlugin;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Proves the product claim inside a real engine: a BenoStreamDB table joins a
 * foreign catalog in one SQL statement, with no export, copy, or ETL step.
 *
 * <p>The foreign catalog is {@code tpch} because it is a first-party connector
 * published at every Trino release, so the test pins exactly the same version
 * as the engine under test. {@code trino-postgresql} is deliberately not used:
 * Trino stopped publishing the JDBC connectors to Maven Central after 476
 * (they now ship only inside the server tarball), so a Maven-scoped PostgreSQL
 * catalog test would silently pin an SPI-incompatible plugin against a 482/484
 * engine. The PostgreSQL path is covered by the server-distribution tier
 * described in {@code plans/trino_version_bands.md}.</p>
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestCrossCatalogJoin {

    private DistributedQueryRunner runner;
    private Path warehouse;

    @BeforeAll
    void setUp() throws Exception {
        assumeTrue(BenoStreamDBJNIBridge.isLoaded(), "libbenostreamdb is not on java.library.path");
        warehouse = Files.createTempDirectory("benostream-trino-join-");
        runner = DistributedQueryRunner.builder(
                        TestingSession.testSessionBuilder()
                                .setCatalog("benostreamdb")
                                .setSchema("default")
                                .build())
                .build();
        runner.installPlugin(new BenoStreamDBPlugin());
        runner.installPlugin(new TpchPlugin());
        runner.createCatalog(
                "benostreamdb",
                "benostreamdb",
                Map.of("benostream.warehouse", warehouse.toUri().toString()));
        runner.createCatalog("tpch", "tpch", Map.of());
    }

    @AfterAll
    void tearDown() {
        if (runner != null) {
            runner.close();
        }
    }

    @Test
    void scalarJoinAcrossCatalogs() {
        runner.execute("CREATE TABLE benostreamdb.default.joined (id bigint, title varchar)");
        runner.execute(
                "INSERT INTO benostreamdb.default.joined VALUES "
                        + "(1, 'graph rag'), (2, 'vector search')");

        MaterializedResult result = runner.execute(
                "SELECT d.title, o.orderstatus "
                        + "FROM benostreamdb.default.joined d "
                        + "JOIN tpch.tiny.orders o ON d.id = o.orderkey "
                        + "ORDER BY d.id");

        assertEquals(2, result.getRowCount());
        assertEquals("graph rag", result.getMaterializedRows().get(0).getField(0));
        assertNotNull(result.getMaterializedRows().get(0).getField(1));
    }

    @Test
    void vectorSearchResultJoinsForeignCatalog() {
        runner.execute(
                "CREATE TABLE benostreamdb.default.join_vectors "
                        + "(id bigint, title varchar, embedding array(real))");
        runner.execute(
                "INSERT INTO benostreamdb.default.join_vectors VALUES "
                        + "(1, 'graph rag', CAST(ARRAY[1.0, 0.0, 0.0] AS array(real))), "
                        + "(2, 'vector search', CAST(ARRAY[0.0, 1.0, 0.0] AS array(real)))");
        runner.execute(
                "CALL benostreamdb.system.add_index('default.join_vectors', 'embedding', 'hnsw')");
        // Index building is detached; build explicitly so the search below is
        // not racing a background task.
        runner.execute("CALL benostreamdb.system.build_index('default.join_vectors')");

        // The nearest neighbour feeds directly into a join against tpch; the
        // vector index result is never materialised outside the engine.
        MaterializedResult result = runner.execute(
                "SELECT v.id, v.title, o.totalprice "
                        + "FROM TABLE(system.vector_search("
                        + "  table_name => 'default.join_vectors', column => 'embedding', "
                        + "  query => '[1.0, 0.0, 0.0]', k => 1)) v "
                        + "JOIN tpch.tiny.orders o ON v.id = o.orderkey");

        assertEquals(1, result.getRowCount());
        assertEquals(1L, result.getMaterializedRows().get(0).getField(0));
        assertEquals("graph rag", result.getMaterializedRows().get(0).getField(1));
    }

    @Test
    void aggregateSpansBothCatalogs() {
        runner.execute("CREATE TABLE benostreamdb.default.grouped (id bigint)");
        runner.execute("INSERT INTO benostreamdb.default.grouped VALUES (1), (2), (3)");

        MaterializedResult result = runner.execute(
                "SELECT count(*) FROM benostreamdb.default.grouped g "
                        + "JOIN tpch.tiny.orders o ON g.id = o.orderkey");

        assertEquals(1, result.getRowCount());
        assertEquals(3L, result.getMaterializedRows().get(0).getField(0));
    }
}
