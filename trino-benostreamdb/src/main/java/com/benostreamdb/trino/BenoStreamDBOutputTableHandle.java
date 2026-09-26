package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorOutputTableHandle;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Handle for a CREATE TABLE AS SELECT. Carries the target table and the ordered
 * columns so the output page sink can create the table and write the rows.
 */
public class BenoStreamDBOutputTableHandle implements ConnectorOutputTableHandle {
    private final String schemaName;
    private final String tableName;
    private final List<BenoStreamDBColumnHandle> columns;

    @JsonCreator
    public BenoStreamDBOutputTableHandle(
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
