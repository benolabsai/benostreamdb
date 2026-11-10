package com.benostreamdb.trino;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;

import java.util.Objects;
import java.util.Optional;

/**
 * Carries a {@code system.keyword_search(...)} call from analysis to the worker
 * that runs the native BM25 query.
 */
public final class KeywordSearchHandle implements ConnectorTableFunctionHandle {

    private final String table;
    private final String tableUri;
    private final String column;
    private final String query;
    private final long k;
    private final Optional<String> filter;

    @JsonCreator
    public KeywordSearchHandle(
            @JsonProperty("table") String table,
            @JsonProperty("tableUri") String tableUri,
            @JsonProperty("column") String column,
            @JsonProperty("query") String query,
            @JsonProperty("k") long k,
            @JsonProperty("filter") Optional<String> filter) {
        this.table = table;
        this.tableUri = tableUri;
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
    public String getTableUri() {
        return tableUri;
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
        if (!(o instanceof KeywordSearchHandle that)) {
            return false;
        }
        return k == that.k
                && Objects.equals(table, that.table)
                && Objects.equals(tableUri, that.tableUri)
                && Objects.equals(column, that.column)
                && Objects.equals(query, that.query)
                && Objects.equals(filter, that.filter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(table, tableUri, column, query, k, filter);
    }

    @Override
    public String toString() {
        return "keyword_search(" + table + ", " + column + ", k=" + k + ")";
    }
}
