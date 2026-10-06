package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorInsertTableHandle;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Objects;

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

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        BenoStreamDBInsertTableHandle that = (BenoStreamDBInsertTableHandle) o;
        return Objects.equals(schemaName, that.schemaName)
                && Objects.equals(tableName, that.tableName)
                && Objects.equals(columns, that.columns);
    }

    @Override
    public int hashCode() {
        return Objects.hash(schemaName, tableName, columns);
    }
}
