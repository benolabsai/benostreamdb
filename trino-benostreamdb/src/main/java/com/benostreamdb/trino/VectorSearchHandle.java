package com.benostreamdb.trino;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;

import java.util.Objects;
import java.util.Optional;

/**
 * Carries the arguments of a {@code system.vector_search(...)} call to the
 * worker that applies the table function.
 */
public final class VectorSearchHandle implements ConnectorTableFunctionHandle {

    private final String table;
    private final String column;
    private final String query;
    private final long k;
    private final Optional<String> filter;

    @JsonCreator
    public VectorSearchHandle(
            @JsonProperty("table") String table,
            @JsonProperty("column") String column,
            @JsonProperty("query") String query,
            @JsonProperty("k") long k,
            @JsonProperty("filter") Optional<String> filter) {
        this.table = table;
        this.column = column;
        this.query = query;
        this.k = k;
        this.filter = filter == null ? Optional.empty() : filter;
    }

    @JsonProperty
    public String getTable() {
        return table;
    }

    @JsonProperty
    public String getColumn() {
        return column;
    }

    @JsonProperty
    public String getQuery() {
        return query;
    }

    @JsonProperty
    public long getK() {
        return k;
    }

    @JsonProperty
    public Optional<String> getFilter() {
        return filter;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        VectorSearchHandle that = (VectorSearchHandle) o;
        return k == that.k
                && Objects.equals(table, that.table)
                && Objects.equals(column, that.column)
                && Objects.equals(query, that.query)
                && Objects.equals(filter, that.filter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(table, column, query, k, filter);
    }
}
