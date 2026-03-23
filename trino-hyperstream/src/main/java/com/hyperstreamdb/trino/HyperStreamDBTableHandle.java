package com.hyperstreamdb.trino;

import io.trino.spi.connector.ConnectorTableHandle;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class HyperStreamDBTableHandle implements ConnectorTableHandle {
    private final String schemaName;
    private final String tableName;

    @JsonCreator
    public HyperStreamDBTableHandle(
            @JsonProperty("schemaName") String schemaName,
            @JsonProperty("tableName") String tableName) {
        this.schemaName = schemaName;
        this.tableName = tableName;
    }

    @JsonProperty
    public String getSchemaName() {
        return schemaName;
    }

    @JsonProperty
    public String getTableName() {
        return tableName;
    }
}
