package com.benostreamdb.trino;

import io.airlift.slice.Slices;
import io.trino.spi.StandardErrorCode;
import io.trino.spi.TrinoException;
import io.trino.spi.connector.ColumnMetadata;
import io.trino.spi.connector.ConnectorAccessControl;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorTransactionHandle;
import io.trino.spi.function.table.Argument;
import io.trino.spi.function.table.ArgumentSpecification;
import io.trino.spi.function.table.ConnectorTableFunction;
import io.trino.spi.function.table.Descriptor;
import io.trino.spi.function.table.ReturnTypeSpecification;
import io.trino.spi.function.table.ScalarArgument;
import io.trino.spi.function.table.ScalarArgumentSpecification;
import io.trino.spi.function.table.TableFunctionAnalysis;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.RealType;
import io.trino.spi.type.Type;
import io.trino.spi.type.VarcharType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * {@code system.hybrid_search(table, text_column, query_text, vector_column,
 * query_vector, k, rrf_k, filter)}.
 *
 * <p>Returns every table column plus `rrf_score` (REAL). The fused score is
 * computed by the engine's `HybridSearchCoordinator`; the connector only
 * carries the request and maps the Arrow result.</p>
 */
public final class HybridSearchFunction implements ConnectorTableFunction {

    private final String warehouse;

    public HybridSearchFunction(String warehouse) {
        this.warehouse = warehouse;
    }

    @Override
    public String getSchema() {
        return "system";
    }

    @Override
    public String getName() {
        return "hybrid_search";
    }

    @Override
    public List<ArgumentSpecification> getArguments() {
        return List.of(
                ScalarArgumentSpecification.builder()
                        .name("TABLE_NAME")
                        .type(VarcharType.VARCHAR)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("TEXT_COLUMN")
                        .type(VarcharType.VARCHAR)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("QUERY_TEXT")
                        .type(VarcharType.VARCHAR)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("VECTOR_COLUMN")
                        .type(VarcharType.VARCHAR)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("QUERY_VECTOR")
                        .type(VarcharType.VARCHAR)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("K")
                        .type(BigintType.BIGINT)
                        .defaultValue(10L)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("RRF_K")
                        .type(BigintType.BIGINT)
                        .defaultValue(60L)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("FILTER")
                        .type(VarcharType.VARCHAR)
                        .defaultValue(Slices.utf8Slice(""))
                        .build());
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
        String textColumn = stringArgument(arguments, "TEXT_COLUMN");
        String queryText = stringArgument(arguments, "QUERY_TEXT");
        String vectorColumn = stringArgument(arguments, "VECTOR_COLUMN");
        String queryVector = stringArgument(arguments, "QUERY_VECTOR");
        long k = longArgument(arguments, "K", 10L);
        long rrfK = longArgument(arguments, "RRF_K", 60L);
        String filter = stringArgument(arguments, "FILTER");

        if (table == null || table.isBlank()) {
            throw userError("table is required");
        }
        if (textColumn == null || textColumn.isBlank()) {
            throw userError("text_column is required");
        }
        if (queryText == null || queryText.isBlank()) {
            throw userError("query_text is required");
        }
        if (vectorColumn == null || vectorColumn.isBlank()) {
            throw userError("vector_column is required");
        }
        if (k < 1 || k > 10_000) {
            throw userError("k must be between 1 and 10000");
        }
        if (rrfK < 1 || rrfK > 10_000) {
            throw userError("rrf_k must be between 1 and 10000");
        }
        // A non-empty filter is pushed into the engine, which resolves it to a
        // row-id bitmap and applies it to both retrieval paths before RRF.
        Optional<String> filterArgument =
                Optional.ofNullable(filter).map(String::trim).filter(value -> !value.isBlank());

        String[] schemaTable = BenoStreamDBTableUri.splitTable(table);
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata(warehouse);
        List<ColumnMetadata> tableColumns = metadata.resolveColumns(schemaTable[0], schemaTable[1]);

        boolean textFound = false;
        boolean vectorFound = false;
        List<String> names = new ArrayList<>(tableColumns.size() + 1);
        List<Type> types = new ArrayList<>(tableColumns.size() + 1);
        for (ColumnMetadata column : tableColumns) {
            names.add(column.getName());
            types.add(column.getType());
            if (column.getName().equals(textColumn)) {
                textFound = true;
            }
            if (column.getName().equals(vectorColumn)) {
                vectorFound = true;
            }
        }
        if (!textFound) {
            throw userError("text_column '" + textColumn + "' does not exist in " + table);
        }
        if (!vectorFound) {
            throw userError("vector_column '" + vectorColumn + "' does not exist in " + table);
        }
        VectorSearchSql.parseVector(queryVector);
        names.add("rrf_score");
        types.add(RealType.REAL);

        String tableUri = BenoStreamDBTableUri.of(warehouse, schemaTable[0], schemaTable[1]);
        HybridSearchHandle handle = new HybridSearchHandle(
                table,
                tableUri,
                textColumn,
                queryText,
                vectorColumn,
                queryVector,
                k,
                rrfK,
                filterArgument);

        return TableFunctionAnalysis.builder()
                .returnedType(Descriptor.descriptor(names, types))
                .handle(handle)
                .build();
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

    private static long longArgument(
            Map<String, Argument> arguments, String name, long defaultValue) {
        String value = stringArgument(arguments, name);
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            throw userError(name + " must be an integer");
        }
    }

    private static TrinoException userError(String message) {
        return new TrinoException(StandardErrorCode.GENERIC_USER_ERROR, message);
    }
}
