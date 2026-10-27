package com.benostreamdb.trino;

import io.trino.spi.StandardErrorCode;
import io.trino.spi.TrinoException;
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
import io.trino.spi.type.VarcharType;

import java.util.List;
import java.util.Map;

/**
 * {@code system.list_indexes(table_name)}: the engine's index listing as a
 * table.
 *
 * <p>The Spark connector's {@code show_indexes} procedure calls the same JNI
 * entry point but discards the result, so this is parity plus a useful
 * improvement: it returns one row per index instead of nothing.</p>
 */
public final class ListIndexesFunction implements ConnectorTableFunction {

    public static final String NAME = "list_indexes";

    private final String warehouse;

    public ListIndexesFunction(String warehouse) {
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
        return List.of(ScalarArgumentSpecification.builder()
                .name("TABLE_NAME")
                .type(VarcharType.VARCHAR)
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
        if (table == null || table.isBlank()) {
            throw new TrinoException(StandardErrorCode.GENERIC_USER_ERROR, "table_name is required");
        }
        String[] schemaTable = BenoStreamDBTableUri.splitTable(table);
        return TableFunctionAnalysis.builder()
                .returnedType(descriptor())
                .handle(new ListIndexesTableHandle(schemaTable[0], schemaTable[1]))
                .build();
    }

    static Descriptor descriptor() {
        return Descriptor.descriptor(
                List.of(
                        ListIndexesPageSource.COLUMN_NAME,
                        ListIndexesPageSource.INDEX_CATEGORY,
                        ListIndexesPageSource.ALGORITHM,
                        ListIndexesPageSource.INDEX_ID,
                        ListIndexesPageSource.FORMAT_VERSION),
                List.of(
                        VarcharType.VARCHAR,
                        VarcharType.VARCHAR,
                        VarcharType.VARCHAR,
                        VarcharType.VARCHAR,
                        BigintType.BIGINT));
    }

    /** Fully-qualified warehouse URI for a handle; used to call the JNI bridge. */
    static String resolveUri(String warehouse, ListIndexesTableHandle handle) {
        return BenoStreamDBTableUri.of(warehouse, handle.getSchemaName(), handle.getTableName());
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
}
