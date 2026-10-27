package com.benostreamdb.trino;

import io.airlift.slice.Slices;
import io.trino.spi.StandardErrorCode;
import io.trino.spi.TrinoException;
import io.trino.spi.connector.ConnectorAccessControl;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorTransactionHandle;
import io.trino.spi.function.table.Argument;
import io.trino.spi.function.table.ArgumentSpecification;
import io.trino.spi.function.table.ConnectorTableFunction;
import io.trino.spi.function.table.ReturnTypeSpecification;
import io.trino.spi.function.table.ScalarArgument;
import io.trino.spi.function.table.ScalarArgumentSpecification;
import io.trino.spi.function.table.TableFunctionAnalysis;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.VarcharType;

import java.util.List;
import java.util.Map;

/**
 * Single-edge-table graph traversal functions:
 * {@code graph_neighbors}, {@code graph_shortest_path}, and
 * {@code graph_subgraph}.
 */
public final class GraphTableFunction implements ConnectorTableFunction {

    private final String warehouse;
    private final GraphSql.Kind kind;

    public GraphTableFunction(String warehouse, GraphSql.Kind kind) {
        this.warehouse = warehouse;
        this.kind = kind;
    }

    @Override
    public String getSchema() {
        return "system";
    }

    @Override
    public String getName() {
        return kind.functionName();
    }

    @Override
    public List<ArgumentSpecification> getArguments() {
        switch (kind) {
            case SHORTEST_PATH:
                return List.of(
                        tableArgument(),
                        ScalarArgumentSpecification.builder()
                                .name("SOURCE")
                                .type(BigintType.BIGINT)
                                .build(),
                        ScalarArgumentSpecification.builder()
                                .name("TARGET")
                                .type(BigintType.BIGINT)
                                .build(),
                        modeArgument());
            case NEIGHBORS:
            case SUBGRAPH:
                return List.of(
                        tableArgument(),
                        ScalarArgumentSpecification.builder()
                                .name("SEEDS")
                                .type(VarcharType.VARCHAR)
                                .build(),
                        ScalarArgumentSpecification.builder()
                                .name("HOPS")
                                .type(BigintType.BIGINT)
                                .defaultValue(1L)
                                .build(),
                        modeArgument());
            default:
                throw new IllegalStateException("Unsupported graph kind: " + kind);
        }
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
            throw userError("table is required");
        }

        GraphTableFunctionHandle handle;
        switch (kind) {
            case SHORTEST_PATH: {
                String source = stringArgument(arguments, "SOURCE");
                String target = stringArgument(arguments, "TARGET");
                String mode = stringArgument(arguments, "MODE");
                handle = new GraphTableFunctionHandle(
                        table, kind.functionName(), List.of(source, target, mode));
                break;
            }
            case NEIGHBORS:
            case SUBGRAPH: {
                String seeds = stringArgument(arguments, "SEEDS");
                String hops = stringArgument(arguments, "HOPS");
                String mode = stringArgument(arguments, "MODE");
                handle = new GraphTableFunctionHandle(
                        table, kind.functionName(), List.of(seeds, hops, mode));
                break;
            }
            default:
                throw new IllegalStateException("Unsupported graph kind: " + kind);
        }

        // Validate the arguments and the table reference before planning.
        GraphSql.build(handle);

        return TableFunctionAnalysis.builder()
                .returnedType(GraphSql.descriptor(kind))
                .handle(handle)
                .build();
    }

    private static ScalarArgumentSpecification tableArgument() {
        return ScalarArgumentSpecification.builder()
                .name("TABLE_NAME")
                .type(VarcharType.VARCHAR)
                .build();
    }

    private static ScalarArgumentSpecification modeArgument() {
        return ScalarArgumentSpecification.builder()
                .name("MODE")
                .type(VarcharType.VARCHAR)
                .defaultValue(Slices.utf8Slice("auto"))
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

    private static TrinoException userError(String message) {
        return new TrinoException(StandardErrorCode.GENERIC_USER_ERROR, message);
    }
}
