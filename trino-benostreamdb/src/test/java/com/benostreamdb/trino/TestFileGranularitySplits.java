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
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * End-to-end coverage for {@code benostream.split.file-granularity=true}.
 *
 * <p>With the property on, a plain scan emits one split per Parquet file and each
 * split is read through a file-scoped engine session instead of the table-wide
 * query. The important property is that fan-out changes <em>where</em> the rows
 * come from, not which rows: a two-file table must still produce exactly the rows
 * the single-split path produces, with no duplicates and no drops.</p>
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestFileGranularitySplits {

    private DistributedQueryRunner runner;

    @BeforeAll
    void setUp() throws Exception {
        assumeTrue(BenoStreamDBJNIBridge.isLoaded(), "libbenostreamdb is not on java.library.path");
        Path warehouse = Files.createTempDirectory("benostream-trino-splits-");
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
                Map.of(
                        "benostream.warehouse", warehouse.toUri().toString(),
                        "benostream.split.file-granularity", "true"));

        runner.execute("CREATE TABLE benostreamdb.default.split_t (id bigint, label varchar)");
        // Two commits => two data files => two splits.
        runner.execute("INSERT INTO benostreamdb.default.split_t VALUES (1, 'a'), (2, 'b')");
        runner.execute("INSERT INTO benostreamdb.default.split_t VALUES (3, 'c'), (4, 'd')");
    }

    @AfterAll
    void tearDown() {
        if (runner != null) {
            runner.close();
        }
    }

    @Test
    void fanOutReturnsEveryRowExactlyOnce() {
        MaterializedResult result = runner.execute(
                "SELECT count(*) AS c, sum(id) AS s FROM benostreamdb.default.split_t");
        assertEquals(1, result.getRowCount());
        assertEquals(4L, ((Number) result.getMaterializedRows().get(0).getField(0)).longValue());
        assertEquals(10L, ((Number) result.getMaterializedRows().get(0).getField(1)).longValue());

        MaterializedResult rows = runner.execute(
                "SELECT id FROM benostreamdb.default.split_t ORDER BY id");
        assertEquals(4, rows.getRowCount());
        List<Long> ids = rows.getMaterializedRows().stream()
                .map(row -> ((Number) row.getField(0)).longValue())
                .toList();
        assertEquals(List.of(1L, 2L, 3L, 4L), ids);
    }

    @Test
    void filteredScanStillUsesTheSingleSplitPath() {
        // A pushed predicate disables fan-out: the filter has to run against the
        // whole table, so the connector keeps one split.
        MaterializedResult result = runner.execute(
                "SELECT id FROM benostreamdb.default.split_t WHERE id > 2 ORDER BY id");
        assertEquals(2, result.getRowCount());
        assertEquals(3L, ((Number) result.getMaterializedRows().get(0).getField(0)).longValue());
        assertEquals(4L, ((Number) result.getMaterializedRows().get(1).getField(0)).longValue());
    }
}
