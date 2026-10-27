package com.benostreamdb.trino;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.trino.spi.connector.*;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Emits a single split per table.
 *
 * The engine's session does the parallelism internally, so there is no
 * file-range split fan-out. The predicate pushed down by
 * {@link AbstractBenoStreamDBMetadata#applyFilter} travels as a SQL {@code WHERE}
 * clause in the split, so the engine applies scalar and inverted-index pruning
 * before decoding Parquet.
 *
 * <p>Table functions ({@code vector_search}, {@code hybrid_search},
 * {@code graph_*}) hand over a complete statement instead: when
 * {@code pushdownSql} is present it replaces the generated
 * {@code SELECT * FROM t [WHERE ...]}, which is what lets the native engine run
 * KNN, RRF, or traversal itself rather than post-filtering in Trino.</p>
 *
 * <p>Multi-split fan-out ({@code benostream.split.file-granularity}) applies to
 * plain, unfiltered table scans only: the split then names one Parquet file and
 * the page source reads it through a file-scoped session. A filtered or
 * function-driven scan keeps a single split, because its SQL has to run against
 * the whole table.</p>
 */
public class BenoStreamDBSplitManager implements ConnectorSplitManager {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String warehouse;
    private final String gpuDevice;
    private final boolean fileGranularity;

    public BenoStreamDBSplitManager(String warehouse, String gpuDevice) {
        this(warehouse, gpuDevice, false);
    }

    public BenoStreamDBSplitManager(String warehouse, String gpuDevice, boolean fileGranularity) {
        this.warehouse = warehouse;
        this.gpuDevice = gpuDevice;
        this.fileGranularity = fileGranularity;
    }

    @Override
    public ConnectorSplitSource getSplits(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorTableHandle table,
            Set<ColumnHandle> columns,
            Constraint constraint) {

        if (table instanceof RegionalDriftFunction.Handle driftHandle) {
            // The whole search is one unit of work; the split carries the
            // resolved request so the page source performs it exactly once.
            String uri = RegionalDriftFunction.resolveUri(warehouse, driftHandle);
            return new FixedSplitSource(List.of(driftHandle.toSplit(uri)));
        }

        if (table instanceof ListIndexesTableHandle listHandle) {
            // Index metadata is produced by the engine's manifest, not by a scan,
            // so it is fetched here and carried in the split as JSON.
            String uri = ListIndexesFunction.resolveUri(warehouse, listHandle);
            String json = BenoStreamDBJNIBridge.listIndexes(uri);
            return new FixedSplitSource(List.of(new ListIndexesSplit(json)));
        }

        BenoStreamDBTableHandle tableHandle = (BenoStreamDBTableHandle) table;
        String uri = BenoStreamDBTableUri.of(warehouse, tableHandle.getSchemaName(), tableHandle.getTableName());
        String filter = tableHandle.getFilterString().orElse("").trim();
        String sql = tableHandle.getPushdownSql().orElseGet(
                () -> filter.isEmpty() ? "SELECT * FROM t" : "SELECT * FROM t WHERE " + filter);

        // Fan out one split per file only when nothing has to be evaluated
        // table-wide (no pushed SQL, no hybrid search, no multi-table statement,
        // no pushed filter). Otherwise the single-split SQL path is the correct
        // one.
        Optional<HybridSearchHandle> hybrid = tableHandle.getHybridSearch();
        boolean plainScan = tableHandle.getPushdownSql().isEmpty()
                && hybrid.isEmpty()
                && tableHandle.getTablesJson().isEmpty()
                && filter.isEmpty();
        if (fileGranularity && plainScan) {
            List<BenoStreamDBSplit> splits = new ArrayList<>();
            for (String file : listSplits(uri)) {
                splits.add(new BenoStreamDBSplit(
                        uri, sql, Optional.empty(), Optional.empty(), Optional.of(file)));
            }
            if (splits.size() > 1) {
                return new FixedSplitSource(splits);
            }
        }

        return new FixedSplitSource(List.of(new BenoStreamDBSplit(
                uri, sql, tableHandle.getHybridSearch(), tableHandle.getTablesJson())));
    }

    /** Data files backing the table, as reported by the engine's manifest. */
    private static List<String> listSplits(String tableUri) {
        String json = BenoStreamDBJNIBridge.listSplits(tableUri);
        if (json == null || json.isBlank()) {
            return List.of();
        }
        try {
            JsonNode root = MAPPER.readTree(json);
            if (root == null || !root.isArray()) {
                return List.of();
            }
            List<String> files = new ArrayList<>(root.size());
            for (JsonNode node : root) {
                // A file with merge-on-read deletes cannot be read raw: the
                // per-file session does not apply delete files, so fall back to
                // the single-split SQL scan for the whole table.
                if (node.path("has_deletes").asBoolean(false)) {
                    return List.of();
                }
                JsonNode path = node.get("file_path");
                if (path != null && !path.asText().isBlank()) {
                    files.add(path.asText());
                }
            }
            return files;
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to parse BenoStreamDB split listing", e);
        }
    }
}
