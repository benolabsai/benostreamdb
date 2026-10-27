package com.benostreamdb.trino;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;

import java.util.Objects;
import java.util.Optional;

/**
 * Carries a {@code system.hybrid_search(...)} call through the split to the
 * worker that runs the native RRF query.
 */
public final class HybridSearchHandle implements ConnectorTableFunctionHandle {

    private final String table;
    private final String tableUri;
    private final String textColumn;
    private final String queryText;
    private final String vectorColumn;
    private final String queryVector;
    private final long k;
    private final long rrfK;
    private final Optional<String> filter;

    @JsonCreator
    public HybridSearchHandle(
            @JsonProperty("table") String table,
            @JsonProperty("tableUri") String tableUri,
            @JsonProperty("textColumn") String textColumn,
            @JsonProperty("queryText") String queryText,
            @JsonProperty("vectorColumn") String vectorColumn,
            @JsonProperty("queryVector") String queryVector,
            @JsonProperty("k") long k,
            @JsonProperty("rrfK") long rrfK,
            @JsonProperty("filter") Optional<String> filter) {
        this.table = table;
        this.tableUri = tableUri;
        this.textColumn = textColumn;
        this.queryText = queryText;
        this.vectorColumn = vectorColumn;
        this.queryVector = queryVector;
        this.k = k;
        this.rrfK = rrfK;
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
    public String getTextColumn() {
        return textColumn;
    }

    @JsonProperty
    public String getQueryText() {
        return queryText;
    }

    @JsonProperty
    public String getVectorColumn() {
        return vectorColumn;
    }

    @JsonProperty
    public String getQueryVector() {
        return queryVector;
    }

    @JsonProperty
    public long getK() {
        return k;
    }

    @JsonProperty
    public long getRrfK() {
        return rrfK;
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
        HybridSearchHandle that = (HybridSearchHandle) o;
        return k == that.k
                && rrfK == that.rrfK
                && Objects.equals(table, that.table)
                && Objects.equals(tableUri, that.tableUri)
                && Objects.equals(textColumn, that.textColumn)
                && Objects.equals(queryText, that.queryText)
                && Objects.equals(vectorColumn, that.vectorColumn)
                && Objects.equals(queryVector, that.queryVector)
                && Objects.equals(filter, that.filter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                table, tableUri, textColumn, queryText, vectorColumn, queryVector, k, rrfK, filter);
    }
}
