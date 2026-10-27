package com.benostreamdb.trino;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;

import java.util.List;
import java.util.Objects;

/**
 * Carries a graph traversal table-function call to the worker.
 *
 * <p>{@code function} is one of {@code graph_neighbors},
 * {@code graph_shortest_path}, or {@code graph_subgraph}; {@code args} holds
 * the validated positional arguments in the order the engine expects.</p>
 */
public final class GraphTableFunctionHandle implements ConnectorTableFunctionHandle {

    private final String table;
    private final String function;
    private final List<String> args;

    @JsonCreator
    public GraphTableFunctionHandle(
            @JsonProperty("table") String table,
            @JsonProperty("function") String function,
            @JsonProperty("args") List<String> args) {
        this.table = table;
        this.function = function;
        this.args = List.copyOf(args);
    }

    @JsonProperty
    public String getTable() {
        return table;
    }

    @JsonProperty
    public String getFunction() {
        return function;
    }

    @JsonProperty
    public List<String> getArgs() {
        return args;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        GraphTableFunctionHandle that = (GraphTableFunctionHandle) o;
        return Objects.equals(table, that.table)
                && Objects.equals(function, that.function)
                && Objects.equals(args, that.args);
    }

    @Override
    public int hashCode() {
        return Objects.hash(table, function, args);
    }
}
