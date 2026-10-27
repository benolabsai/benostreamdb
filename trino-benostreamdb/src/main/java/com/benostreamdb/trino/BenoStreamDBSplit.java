package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorSplit;
import io.trino.spi.HostAddress;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Collections;
import java.util.Objects;
import java.util.Optional;

/**
 * A single scan of a BenoStreamDB table.
 *
 * The split carries the table URI and the SQL query (with the pushed-down
 * predicate) that the page source runs through the engine's session. The engine
 * does the parallelism internally, so there is no file-range split fan-out.
 */
public class BenoStreamDBSplit implements ConnectorSplit {
    private final String tableUri;
    private final String sql;
    private final Optional<HybridSearchHandle> hybridSearch;
    /**
     * JSON map of {@code alias -> table URI}. Present only for multi-table
     * statements such as {@code graph_rag_search}; the page source then runs the
     * SQL through {@code openQueryMulti} instead of {@code openQuery}.
     */
    private final Optional<String> tablesJson;

    @JsonCreator
    public BenoStreamDBSplit(
        @JsonProperty("tableUri") String tableUri,
        @JsonProperty("sql") String sql,
        @JsonProperty("hybridSearch") Optional<HybridSearchHandle> hybridSearch,
        @JsonProperty("tablesJson") Optional<String> tablesJson) {
        this.tableUri = tableUri;
        this.sql = sql;
        this.hybridSearch = hybridSearch == null ? Optional.empty() : hybridSearch;
        this.tablesJson = tablesJson == null ? Optional.empty() : tablesJson;
    }

    public BenoStreamDBSplit(
            String tableUri, String sql, Optional<HybridSearchHandle> hybridSearch) {
        this(tableUri, sql, hybridSearch, Optional.empty());
    }

    public BenoStreamDBSplit(String tableUri, String sql) {
        this(tableUri, sql, Optional.empty(), Optional.empty());
    }

    @JsonProperty
    public Optional<String> getTablesJson() {
        return tablesJson;
    }

    @Override
    public boolean isRemotelyAccessible() {
        return true;
    }

    @Override
    public List<HostAddress> getAddresses() {
        return Collections.emptyList();
    }

    @JsonProperty
    public String getTableUri() {
        return tableUri;
    }

    @JsonProperty
    public String getSql() {
        return sql;
    }

    @JsonProperty
    public Optional<HybridSearchHandle> getHybridSearch() {
        return hybridSearch;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        BenoStreamDBSplit that = (BenoStreamDBSplit) o;
        return Objects.equals(tableUri, that.tableUri)
                && Objects.equals(sql, that.sql)
                && Objects.equals(hybridSearch, that.hybridSearch);
    }

    @Override
    public int hashCode() {
        return Objects.hash(tableUri, sql, hybridSearch);
    }
}
