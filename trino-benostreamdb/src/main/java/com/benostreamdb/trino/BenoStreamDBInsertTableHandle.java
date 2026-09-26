package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorInsertTableHandle;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Handle for an INSERT into a BenoStreamDB table. Carries the target table and
 * the ordered columns being written so the page sink can build the Arrow batch.
 */
public class BenoStreamDBInsertTableHandle implements ConnectorInsertTableHandle {
    private final String schemaName;
    private final String tableName;
    private final List<BenoStreamDBColumnHandle> columns;

    @JsonCreator
    public BenoStreamDBInsertTableHandle(
            @JsonProperty("schemaName") String schemaName,
            @JsonProperty("tableName") String tableName,
            @JsonProperty("columns") List<BenoStreamDBColumnHandle> columns) {
        this.schemaName = schemaName;
        this.tableName = tableName;
        this.columns = columns;
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
    public List<BenoStreamDBColumnHandle> getColumns() {
        return columns;
    }
}
