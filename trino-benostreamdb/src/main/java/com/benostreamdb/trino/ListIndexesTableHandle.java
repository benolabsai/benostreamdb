package com.benostreamdb.trino;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.trino.spi.connector.ConnectorTableHandle;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;

import java.util.Objects;

/**
 * Handle for the {@code list_indexes} table function.
 *
 * <p>This is a distinct handle type rather than a mode of
 * {@link BenoStreamDBTableHandle} because the rows come from the engine's
 * manifest metadata, not from a scan: the split carries JSON from
 * {@code listIndexes} and the page source materialises it without touching
 * Arrow or Parquet.</p>
 */
public class ListIndexesTableHandle implements ConnectorTableHandle, ConnectorTableFunctionHandle {

    private final String schemaName;
    private final String tableName;

    @JsonCreator
    public ListIndexesTableHandle(
            @JsonProperty("schemaName") String schemaName,
            @JsonProperty("tableName") String tableName) {
        this.schemaName = Objects.requireNonNull(schemaName, "schemaName is null");
        this.tableName = Objects.requireNonNull(tableName, "tableName is null");
    }

    @JsonProperty
    public String getSchemaName() {
        return schemaName;
    }

    @JsonProperty
    public String getTableName() {
        return tableName;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof ListIndexesTableHandle that)) {
            return false;
        }
        return schemaName.equals(that.schemaName) && tableName.equals(that.tableName);
    }

    @Override
    public int hashCode() {
        return Objects.hash(schemaName, tableName);
    }

    @Override
    public String toString() {
        return schemaName + "." + tableName;
    }
}
