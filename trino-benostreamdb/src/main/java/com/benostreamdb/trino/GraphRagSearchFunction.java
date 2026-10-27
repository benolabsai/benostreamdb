package com.benostreamdb.trino;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.collect.ImmutableMap;
import io.trino.spi.StandardErrorCode;
import io.trino.spi.TrinoException;
import io.trino.spi.connector.ConnectorAccessControl;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorTransactionHandle;
import io.trino.spi.function.table.Argument;
import io.trino.spi.function.table.ArgumentSpecification;
import io.trino.spi.function.table.ConnectorTableFunction;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;
import io.trino.spi.function.table.ReturnTypeSpecification;
import io.trino.spi.function.table.ScalarArgument;
import io.trino.spi.function.table.ScalarArgumentSpecification;
import io.trino.spi.function.table.TableFunctionAnalysis;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.VarcharType;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * {@code system.graph_rag_search(...)}: vector retrieval over a document table,
 * expanded through a graph edge table.
 *
 * <p>This is the multi-table function Phase 3b was blocked on. Both tables are
 * registered in one engine session (as {@code t} and {@code e}) so a single
 * statement can rank documents by distance and join them to a traversal:</p>
 *
 * <pre>{@code
 * SELECT d.* FROM t d
 * JOIN graph_neighbors('e', '<seeds>', <hops>, '<mode>') g ON d.<id> = g.node
 * ORDER BY dist_cosine(d.<vector>, '[<query>]') LIMIT <k>
 * }</pre>
 */
public final class GraphRagSearchFunction implements ConnectorTableFunction {

    public static final String NAME = "graph_rag_search";

    private static final Pattern SEEDS = Pattern.compile("[0-9]+(?:,[0-9]+)*");
    private static final String DOCS_ALIAS = "t";
    private static final String EDGE_ALIAS = "e";

    private final String warehouse;

    public GraphRagSearchFunction(String warehouse) {
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
                stringArg("TABLE_NAME", null),
                stringArg("EDGE_TABLE", null),
                stringArg("VECTOR_COLUMN", null),
                stringArg("QUERY_VECTOR", null),
                stringArg("SEEDS", null),
                stringArg("ID_COLUMN", "id"),
                integerArg("HOPS", 1L),
                integerArg("K", 10L),
                stringArg("MODE", "auto"));
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
        String docsTable = required(arguments, "TABLE_NAME");
        String edgeTable = required(arguments, "EDGE_TABLE");
        String vectorColumn = required(arguments, "VECTOR_COLUMN");
        String queryVector = required(arguments, "QUERY_VECTOR");
        String seeds = required(arguments, "SEEDS");
        if (!SEEDS.matcher(seeds.trim()).matches()) {
            throw userError("seeds must be a comma-separated list of node ids");
        }

        String[] docs = BenoStreamDBTableUri.splitTable(docsTable);
        String[] edges = BenoStreamDBTableUri.splitTable(edgeTable);

        Handle handle = new Handle(
                docs[0], docs[1],
                edges[0], edges[1],
                stringArgument(arguments, "ID_COLUMN", "id"),
                vectorColumn,
                queryVector,
                seeds.trim(),
                intArgument(arguments, "HOPS", 1),
                intArgument(arguments, "K", 10),
                stringArgument(arguments, "MODE", "auto"));

        // The return schema is the document table's columns plus `hop`, so it has
        // to be resolved from the table before planning; build the statement too
        // so invalid references fail at analysis time rather than on a worker.
        List<io.trino.spi.connector.ColumnMetadata> documentColumns =
                new BenoStreamDBMetadata(warehouse).resolveColumns(docs[0], docs[1]);
        GraphRagSql.build(handle);

        return TableFunctionAnalysis.builder()
                .returnedType(GraphRagSql.descriptor(documentColumns))
                .handle(handle)
                .build();
    }

    /** The alias-to-URI map handed to {@code openQueryMulti}. */
    static String tablesJson(String docsUri, String edgeUri) {
        return "{\"" + DOCS_ALIAS + "\":\"" + docsUri + "\",\""
                + EDGE_ALIAS + "\":\"" + edgeUri + "\"}";
    }

    private static ScalarArgumentSpecification stringArg(String name, String defaultValue) {
        ScalarArgumentSpecification.Builder builder =
                ScalarArgumentSpecification.builder().name(name).type(VarcharType.VARCHAR);
        if (defaultValue != null) {
            builder = builder.defaultValue(io.airlift.slice.Slices.utf8Slice(defaultValue));
        }
        return builder.build();
    }

    private static ScalarArgumentSpecification integerArg(String name, long defaultValue) {
        return ScalarArgumentSpecification.builder()
                .name(name)
                .type(io.trino.spi.type.IntegerType.INTEGER)
                .defaultValue(defaultValue)
                .build();
    }

    private static String required(Map<String, Argument> arguments, String name) {
        String value = stringArgument(arguments, name, null);
        if (value == null || value.isBlank()) {
            throw userError(name.toLowerCase(java.util.Locale.ENGLISH) + " is required");
        }
        return value;
    }

    private static String stringArgument(Map<String, Argument> arguments, String name, String fallback) {
        Argument argument = arguments.get(name);
        if (!(argument instanceof ScalarArgument)) {
            return fallback;
        }
        Object value = ((ScalarArgument) argument).getValue();
        String text = value instanceof io.airlift.slice.Slice
                ? ((io.airlift.slice.Slice) value).toStringUtf8()
                : (value == null ? null : value.toString());
        return text == null || text.isBlank() ? fallback : text;
    }

    private static int intArgument(Map<String, Argument> arguments, String name, int fallback) {
        Argument argument = arguments.get(name);
        if (!(argument instanceof ScalarArgument)) {
            return fallback;
        }
        Object value = ((ScalarArgument) argument).getValue();
        return value instanceof Number ? ((Number) value).intValue() : fallback;
    }

    private static TrinoException userError(String message) {
        return new TrinoException(StandardErrorCode.GENERIC_USER_ERROR, message);
    }

    /** Plan-time handle; also serves as the connector table handle. */
    public static final class Handle implements ConnectorTableFunctionHandle {
        private final String schemaName;
        private final String tableName;
        private final String edgeSchemaName;
        private final String edgeTableName;
        private final String idColumn;
        private final String vectorColumn;
        private final String queryVector;
        private final String seeds;
        private final int hops;
        private final int k;
        private final String mode;

        @JsonCreator
        public Handle(
                @JsonProperty("schemaName") String schemaName,
                @JsonProperty("tableName") String tableName,
                @JsonProperty("edgeSchemaName") String edgeSchemaName,
                @JsonProperty("edgeTableName") String edgeTableName,
                @JsonProperty("idColumn") String idColumn,
                @JsonProperty("vectorColumn") String vectorColumn,
                @JsonProperty("queryVector") String queryVector,
                @JsonProperty("seeds") String seeds,
                @JsonProperty("hops") int hops,
                @JsonProperty("k") int k,
                @JsonProperty("mode") String mode) {
            this.schemaName = Objects.requireNonNull(schemaName, "schemaName is null");
            this.tableName = Objects.requireNonNull(tableName, "tableName is null");
            this.edgeSchemaName = Objects.requireNonNull(edgeSchemaName, "edgeSchemaName is null");
            this.edgeTableName = Objects.requireNonNull(edgeTableName, "edgeTableName is null");
            this.idColumn = Objects.requireNonNull(idColumn, "idColumn is null");
            this.vectorColumn = Objects.requireNonNull(vectorColumn, "vectorColumn is null");
            this.queryVector = Objects.requireNonNull(queryVector, "queryVector is null");
            this.seeds = Objects.requireNonNull(seeds, "seeds is null");
            this.hops = hops;
            this.k = k;
            this.mode = Objects.requireNonNull(mode, "mode is null");
        }

        @JsonProperty public String getSchemaName() { return schemaName; }
        @JsonProperty public String getTableName() { return tableName; }
        @JsonProperty public String getEdgeSchemaName() { return edgeSchemaName; }
        @JsonProperty public String getEdgeTableName() { return edgeTableName; }
        @JsonProperty public String getIdColumn() { return idColumn; }
        @JsonProperty public String getVectorColumn() { return vectorColumn; }
        @JsonProperty public String getQueryVector() { return queryVector; }
        @JsonProperty public String getSeeds() { return seeds; }
        @JsonProperty public int getHops() { return hops; }
        @JsonProperty public int getK() { return k; }
        @JsonProperty public String getMode() { return mode; }

        Map<String, String> tables(String docsUri, String edgeUri) {
            return ImmutableMap.of(DOCS_ALIAS, docsUri, EDGE_ALIAS, edgeUri);
        }
    }
}
