package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorTableHandle;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Objects;
import java.util.Optional;

public class BenoStreamDBTableHandle implements ConnectorTableHandle {
    private final String schemaName;
    private final String tableName;
    private final Optional<String> filterString;
    private final Optional<String> pushdownSql;
    private final Optional<HybridSearchHandle> hybridSearch;
    /** Alias -> table URI map for multi-table statements (graph_rag_search). */
    private final Optional<String> tablesJson;

    @JsonCreator
    public BenoStreamDBTableHandle(
            @JsonProperty("schemaName") String schemaName,
            @JsonProperty("tableName") String tableName,
            @JsonProperty("filterString") Optional<String> filterString,
            @JsonProperty("pushdownSql") Optional<String> pushdownSql,
            @JsonProperty("hybridSearch") Optional<HybridSearchHandle> hybridSearch,
            @JsonProperty("tablesJson") Optional<String> tablesJson) {
        this.schemaName = schemaName;
        this.tableName = tableName;
        this.filterString = filterString;
        this.pushdownSql = pushdownSql == null ? Optional.empty() : pushdownSql;
        this.hybridSearch = hybridSearch == null ? Optional.empty() : hybridSearch;
        this.tablesJson = tablesJson == null ? Optional.empty() : tablesJson;
    }

    public BenoStreamDBTableHandle(
            String schemaName,
            String tableName,
            Optional<String> filterString,
            Optional<String> pushdownSql,
            Optional<HybridSearchHandle> hybridSearch) {
        this(schemaName, tableName, filterString, pushdownSql, hybridSearch, Optional.empty());
    }

    public BenoStreamDBTableHandle(
            String schemaName,
            String tableName,
            Optional<String> filterString,
            Optional<String> pushdownSql) {
        this(schemaName, tableName, filterString, pushdownSql, Optional.empty());
    }

    public BenoStreamDBTableHandle(
            String schemaName,
            String tableName,
            Optional<String> filterString) {
        this(schemaName, tableName, filterString, Optional.empty(), Optional.empty());
    }

    public BenoStreamDBTableHandle(String schemaName, String tableName) {
        this(schemaName, tableName, Optional.empty(), Optional.empty(), Optional.empty());
    }

    @JsonProperty
    public String getSchemaName() {
        return schemaName;
    }

    @JsonProperty
    public String getTableName() {
        return tableName;
    }

    @JsonProperty
    public Optional<String> getFilterString() {
        return filterString;
    }

    @JsonProperty
    public Optional<String> getPushdownSql() {
        return pushdownSql;
    }

    @JsonProperty
    public Optional<HybridSearchHandle> getHybridSearch() {
        return hybridSearch;
    }

    @JsonProperty
    public Optional<String> getTablesJson() {
        return tablesJson;
    }

    // Trino's planner relies on value equality for table handles (maps, sets,
    // and cross-call comparisons), so implement it explicitly.
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        BenoStreamDBTableHandle that = (BenoStreamDBTableHandle) o;
        return Objects.equals(schemaName, that.schemaName)
                && Objects.equals(tableName, that.tableName)
                && Objects.equals(filterString, that.filterString)
                && Objects.equals(pushdownSql, that.pushdownSql)
                && Objects.equals(hybridSearch, that.hybridSearch);
    }

    @Override
    public int hashCode() {
        return Objects.hash(schemaName, tableName, filterString, pushdownSql, hybridSearch);
    }
}
