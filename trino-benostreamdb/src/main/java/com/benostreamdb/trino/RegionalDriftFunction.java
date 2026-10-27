package com.benostreamdb.trino;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.trino.spi.StandardErrorCode;
import io.trino.spi.TrinoException;
import io.trino.spi.connector.ConnectorAccessControl;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorSplit;
import io.trino.spi.connector.ConnectorTableHandle;
import io.trino.spi.connector.ConnectorTransactionHandle;
import io.trino.spi.function.table.Argument;
import io.trino.spi.function.table.ArgumentSpecification;
import io.trino.spi.function.table.ConnectorTableFunction;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;
import io.trino.spi.function.table.Descriptor;
import io.trino.spi.function.table.ReturnTypeSpecification;
import io.trino.spi.function.table.ScalarArgument;
import io.trino.spi.function.table.ScalarArgumentSpecification;
import io.trino.spi.function.table.TableFunctionAnalysis;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.IntegerType;
import io.trino.spi.type.VarcharType;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * {@code system.regional_drift_search(...)}: DRIFT graph traversal.
 *
 * <p>Spark registers this as a procedure whose {@code call} runs the search and
 * then returns an empty iterator, so {@code CALL regional_drift_search(...)}
 * produces no rows even though the work is done. Exposing it as a table
 * function is what makes the result usable.</p>
 *
 * <p>The engine returns a single Arrow batch with a {@code node_id} column; that
 * is the same column the Spark bridge reads via the Arrow C data interface, so
 * the Trino page source mirrors it exactly.</p>
 */
public final class RegionalDriftFunction implements ConnectorTableFunction {

    public static final String NAME = "regional_drift_search";

    private final String warehouse;

    public RegionalDriftFunction(String warehouse) {
        this.warehouse = warehouse;
    }

    @Override
    public String getSchema() {
        return "system";
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public List<ArgumentSpecification> getArguments() {
        return List.of(
                ScalarArgumentSpecification.builder()
                        .name("TABLE_NAME").type(VarcharType.VARCHAR).build(),
                ScalarArgumentSpecification.builder()
                        .name("QUERY").type(VarcharType.VARCHAR).build(),
                ScalarArgumentSpecification.builder()
                        .name("SEEDS").type(VarcharType.VARCHAR).build(),
                ScalarArgumentSpecification.builder()
                        .name("TOP_K").type(IntegerType.INTEGER).defaultValue(5L).build(),
                ScalarArgumentSpecification.builder()
                        .name("HOPS").type(IntegerType.INTEGER).defaultValue(2L).build(),
                ScalarArgumentSpecification.builder()
                        .name("N_DEPTH").type(IntegerType.INTEGER).defaultValue(2L).build(),
                ScalarArgumentSpecification.builder()
                        .name("K_FOLLOWUPS").type(IntegerType.INTEGER).defaultValue(3L).build(),
                ScalarArgumentSpecification.builder()
                        .name("MODE").type(VarcharType.VARCHAR).defaultValue(
                                io.airlift.slice.Slices.utf8Slice("auto")).build());
    }

    @Override
    public ReturnTypeSpecification getReturnTypeSpecification() {
        return ReturnTypeSpecification.GenericTable.GENERIC_TABLE;
    }

    @Override
    public TableFunctionAnalysis analyze(
            ConnectorSession session,
            ConnectorTransactionHandle transactionHandle,
            Map<String, Argument> arguments,
            ConnectorAccessControl accessControl) {
        String table = stringArgument(arguments, "TABLE_NAME");
        String query = stringArgument(arguments, "QUERY");
        String seeds = stringArgument(arguments, "SEEDS");
        if (table == null || table.isBlank()) {
            throw new TrinoException(StandardErrorCode.GENERIC_USER_ERROR, "table_name is required");
        }
        if (query == null || query.isBlank()) {
            throw new TrinoException(StandardErrorCode.GENERIC_USER_ERROR, "query is required");
        }
        if (seeds == null || seeds.isBlank()) {
            throw new TrinoException(StandardErrorCode.GENERIC_USER_ERROR, "seeds is required");
        }

        String[] schemaTable = BenoStreamDBTableUri.splitTable(table);
        Handle handle = new Handle(
                schemaTable[0],
                schemaTable[1],
                query,
                seeds,
                intArgument(arguments, "TOP_K", 5),
                intArgument(arguments, "HOPS", 2),
                intArgument(arguments, "N_DEPTH", 2),
                intArgument(arguments, "K_FOLLOWUPS", 3),
                defaulted(stringArgument(arguments, "MODE"), "auto"));

        return TableFunctionAnalysis.builder()
                .returnedType(descriptor())
                .handle(handle)
                .build();
    }

    static Descriptor descriptor() {
        return Descriptor.descriptor(
                List.of(RegionalDriftPageSource.NODE_ID),
                List.of(BigintType.BIGINT));
    }

    static String resolveUri(String warehouse, Handle handle) {
        return BenoStreamDBTableUri.of(warehouse, handle.getSchemaName(), handle.getTableName());
    }

    private static String defaulted(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    private static String stringArgument(Map<String, Argument> arguments, String name) {
        Argument argument = arguments.get(name);
        if (!(argument instanceof ScalarArgument)) {
            return null;
        }
        Object value = ((ScalarArgument) argument).getValue();
        if (value instanceof io.airlift.slice.Slice) {
            return ((io.airlift.slice.Slice) value).toStringUtf8();
        }
        return value == null ? null : value.toString();
    }

    private static int intArgument(Map<String, Argument> arguments, String name, int fallback) {
        Argument argument = arguments.get(name);
        if (!(argument instanceof ScalarArgument)) {
            return fallback;
        }
        Object value = ((ScalarArgument) argument).getValue();
        return value instanceof Number ? ((Number) value).intValue() : fallback;
    }

    /**
     * Serves as both the table-function handle (returned from analysis) and the
     * connector table handle (returned from {@code applyTableFunction}).
     */
    public static final class Handle implements ConnectorTableHandle, ConnectorTableFunctionHandle {
        private final String schemaName;
        private final String tableName;
        private final String query;
        private final String seeds;
        private final int topK;
        private final int hops;
        private final int nDepth;
        private final int kFollowups;
        private final String mode;

        @JsonCreator
        public Handle(
                @JsonProperty("schemaName") String schemaName,
                @JsonProperty("tableName") String tableName,
                @JsonProperty("query") String query,
                @JsonProperty("seeds") String seeds,
                @JsonProperty("topK") int topK,
                @JsonProperty("hops") int hops,
                @JsonProperty("nDepth") int nDepth,
                @JsonProperty("kFollowups") int kFollowups,
                @JsonProperty("mode") String mode) {
            this.schemaName = Objects.requireNonNull(schemaName, "schemaName is null");
            this.tableName = Objects.requireNonNull(tableName, "tableName is null");
            this.query = Objects.requireNonNull(query, "query is null");
            this.seeds = Objects.requireNonNull(seeds, "seeds is null");
            this.topK = topK;
            this.hops = hops;
            this.nDepth = nDepth;
            this.kFollowups = kFollowups;
            this.mode = Objects.requireNonNull(mode, "mode is null");
        }

        @JsonProperty public String getSchemaName() { return schemaName; }
        @JsonProperty public String getTableName() { return tableName; }
        @JsonProperty public String getQuery() { return query; }
        @JsonProperty public String getSeeds() { return seeds; }
        @JsonProperty public int getTopK() { return topK; }
        @JsonProperty public int getHops() { return hops; }
        @JsonProperty public int getNDepth() { return nDepth; }
        @JsonProperty public int getKFollowups() { return kFollowups; }
        @JsonProperty public String getMode() { return mode; }

        Split toSplit(String tableUri) {
            return new Split(tableUri, query, seeds, topK, hops, nDepth, kFollowups, mode);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof Handle that)) {
                return false;
            }
            return topK == that.topK && hops == that.hops && nDepth == that.nDepth
                    && kFollowups == that.kFollowups
                    && schemaName.equals(that.schemaName) && tableName.equals(that.tableName)
                    && query.equals(that.query) && seeds.equals(that.seeds)
                    && mode.equals(that.mode);
        }

        @Override
        public int hashCode() {
            return Objects.hash(schemaName, tableName, query, seeds, topK, hops, nDepth, kFollowups, mode);
        }

        @Override
        public String toString() {
            return schemaName + "." + tableName + " drift(" + query + ")";
        }
    }

    /** Carries the resolved request so the search runs once per split. */
    public static final class Split implements ConnectorSplit {
        private final String tableUri;
        private final String query;
        private final String seeds;
        private final int topK;
        private final int hops;
        private final int nDepth;
        private final int kFollowups;
        private final String mode;

        @JsonCreator
        public Split(
                @JsonProperty("tableUri") String tableUri,
                @JsonProperty("query") String query,
                @JsonProperty("seeds") String seeds,
                @JsonProperty("topK") int topK,
                @JsonProperty("hops") int hops,
                @JsonProperty("nDepth") int nDepth,
                @JsonProperty("kFollowups") int kFollowups,
                @JsonProperty("mode") String mode) {
            this.tableUri = Objects.requireNonNull(tableUri, "tableUri is null");
            this.query = Objects.requireNonNull(query, "query is null");
            this.seeds = Objects.requireNonNull(seeds, "seeds is null");
            this.topK = topK;
            this.hops = hops;
            this.nDepth = nDepth;
            this.kFollowups = kFollowups;
            this.mode = Objects.requireNonNull(mode, "mode is null");
        }

        @JsonProperty public String getTableUri() { return tableUri; }
        @JsonProperty public String getQuery() { return query; }
        @JsonProperty public String getSeeds() { return seeds; }
        @JsonProperty public int getTopK() { return topK; }
        @JsonProperty public int getHops() { return hops; }
        @JsonProperty public int getNDepth() { return nDepth; }
        @JsonProperty public int getKFollowups() { return kFollowups; }
        @JsonProperty public String getMode() { return mode; }

        @Override
        public boolean isRemotelyAccessible() {
            return true;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof Split that)) {
                return false;
            }
            return topK == that.topK && hops == that.hops && nDepth == that.nDepth
                    && kFollowups == that.kFollowups && tableUri.equals(that.tableUri)
                    && query.equals(that.query) && seeds.equals(that.seeds)
                    && mode.equals(that.mode);
        }

        @Override
        public int hashCode() {
            return Objects.hash(tableUri, query, seeds, topK, hops, nDepth, kFollowups, mode);
        }
    }
}
