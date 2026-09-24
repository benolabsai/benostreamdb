package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorTableHandle;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Optional;

public class BenoStreamDBTableHandle implements ConnectorTableHandle {
    private final String schemaName;
    private final String tableName;
    private final Optional<String> filterString;

    @JsonCreator
    public BenoStreamDBTableHandle(
            @JsonProperty("schemaName") String schemaName,
            @JsonProperty("tableName") String tableName,
            @JsonProperty("filterString") Optional<String> filterString) {
        this.schemaName = schemaName;
        this.tableName = tableName;
        this.filterString = filterString;
    }

    public BenoStreamDBTableHandle(String schemaName, String tableName) {
        this(schemaName, tableName, Optional.empty());
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
}
