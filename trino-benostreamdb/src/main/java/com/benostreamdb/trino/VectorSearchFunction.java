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
import io.trino.spi.type.DoubleType;
import io.trino.spi.type.Type;
import io.trino.spi.type.VarcharType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * {@code system.vector_search(table, column, query, k, filter)}.
 *
 * <p>{@code query} is a JSON array of numbers. The function resolves the table
 * schema at analysis time, returns every table column plus a {@code distance}
 * column, and hands the KNN SQL to {@link BenoStreamDBMetadata#applyTableFunction}.</p>
 */
public final class VectorSearchFunction implements ConnectorTableFunction {

    private static final String SCHEMA = "system";
    private static final String NAME = "vector_search";

    private final String warehouse;

    public VectorSearchFunction(String warehouse) {
        this.warehouse = warehouse;
    }

    @Override
    public String getSchema() {
        return SCHEMA;
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public List<ArgumentSpecification> getArguments() {
        return List.of(
                ScalarArgumentSpecification.builder()
                        .name("TABLE_NAME")
                        .type(VarcharType.VARCHAR)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("COLUMN")
                        .type(VarcharType.VARCHAR)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("QUERY")
                        .type(VarcharType.VARCHAR)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("K")
                        .type(BigintType.BIGINT)
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
        String column = stringArgument(arguments, "COLUMN");
        String query = stringArgument(arguments, "QUERY");
        long k = longArgument(arguments, "K");
        String filter = stringArgument(arguments, "FILTER");

        if (table == null || table.isBlank()) {
            throw userError("table is required");
        }
        if (column == null || column.isBlank()) {
            throw userError("column is required");
        }

        String[] schemaTable = BenoStreamDBTableUri.splitTable(table);
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata(warehouse);
        List<ColumnMetadata> tableColumns = metadata.resolveColumns(schemaTable[0], schemaTable[1]);

        List<String> names = new ArrayList<>(tableColumns.size() + 1);
        List<Type> types = new ArrayList<>(tableColumns.size() + 1);
        boolean found = false;
        for (ColumnMetadata tableColumn : tableColumns) {
            names.add(tableColumn.getName());
            types.add(tableColumn.getType());
            if (tableColumn.getName().equals(column)) {
                found = true;
            }
        }
        if (!found) {
            throw userError("column '" + column + "' does not exist in " + table);
        }
        names.add("distance");
        types.add(DoubleType.DOUBLE);

        Optional<String> filterArgument = Optional.ofNullable(filter).filter(value -> !value.isBlank());
        VectorSearchHandle handle = new VectorSearchHandle(table, column, query, k, filterArgument);

        // Validate the vector, k, identifiers, and filter before planning.
        VectorSearchSql.build(handle);

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

    private static long longArgument(Map<String, Argument> arguments, String name) {
        Argument argument = arguments.get(name);
        if (!(argument instanceof ScalarArgument)) {
            throw userError(name + " is required");
        }
        Object value = ((ScalarArgument) argument).getValue();
        if (value instanceof io.airlift.slice.Slice) {
            value = ((io.airlift.slice.Slice) value).toStringUtf8();
        }
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        if (value instanceof String) {
            try {
                return Long.parseLong((String) value);
            } catch (NumberFormatException e) {
                throw userError(name + " must be an integer");
            }
        }
        throw userError(name + " must be an integer");
    }

    private static TrinoException userError(String message) {
        return new TrinoException(StandardErrorCode.GENERIC_USER_ERROR, message);
    }
}
