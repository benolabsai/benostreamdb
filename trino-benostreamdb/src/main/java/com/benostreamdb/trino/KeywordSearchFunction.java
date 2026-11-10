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
 * {@code system.keyword_search(table, column, query, k, filter)}.
 *
 * <p>BM25-only retrieval: the lexical half of {@code hybrid_search}, without the
 * dense side and without fusion. Returns every table column plus a {@code score}
 * column (BM25, higher is better). Rows come back in descending BM25 order; the
 * engine's keyword path ranks them, so the statement carries no {@code ORDER BY},
 * which would defeat the index scan.</p>
 */
public final class KeywordSearchFunction implements ConnectorTableFunction {

    private static final String SCHEMA = "system";
    private static final String NAME = "keyword_search";

    private final String warehouse;

    public KeywordSearchFunction(String warehouse) {
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
                        .name("TEXT_COLUMN")
                        .type(VarcharType.VARCHAR)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("QUERY_TEXT")
                        .type(VarcharType.VARCHAR)
                        .build(),
                ScalarArgumentSpecification.builder()
                        .name("K")
                        .type(BigintType.BIGINT)
                        .defaultValue(10L)
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
        long k = longArgument(arguments, "K", 10L);
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

        String[] schemaTable = BenoStreamDBTableUri.splitTable(table);
        BenoStreamDBMetadata metadata = new BenoStreamDBMetadata(warehouse);
        List<ColumnMetadata> tableColumns = metadata.resolveColumns(schemaTable[0], schemaTable[1]);

        boolean textFound = false;
        List<String> names = new ArrayList<>(tableColumns.size() + 1);
        List<Type> types = new ArrayList<>(tableColumns.size() + 1);
        for (ColumnMetadata column : tableColumns) {
            names.add(column.getName());
            types.add(column.getType());
            if (column.getName().equals(textColumn)) {
                textFound = true;
            }
        }
        if (!textFound) {
            throw userError("text_column '" + textColumn + "' does not exist in " + table);
        }
        names.add("score");
        types.add(DoubleType.DOUBLE);

        Optional<String> filterArgument =
                Optional.ofNullable(filter).map(String::trim).filter(value -> !value.isBlank());

        String tableUri = BenoStreamDBTableUri.of(warehouse, schemaTable[0], schemaTable[1]);
        KeywordSearchHandle handle = new KeywordSearchHandle(
                table, tableUri, textColumn, queryText, k, filterArgument);

        // Validate identifiers, the query, and k before the plan is built.
        KeywordSearchSql.build(handle);

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

    private static long longArgument(Map<String, Argument> arguments, String name, long fallback) {
        Argument argument = arguments.get(name);
        if (!(argument instanceof ScalarArgument)) {
            return fallback;
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
                return Long.parseLong(((String) value).trim());
            } catch (NumberFormatException e) {
                throw userError(name + " must be an integer");
            }
        }
        return fallback;
    }

    private static TrinoException userError(String message) {
        return new TrinoException(StandardErrorCode.GENERIC_USER_ERROR, message);
    }
}
