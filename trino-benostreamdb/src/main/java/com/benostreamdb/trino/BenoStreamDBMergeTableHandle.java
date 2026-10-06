package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorMergeTableHandle;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Objects;

/**
 * Handle for a MERGE into a BenoStreamDB table. Wraps the target table handle,
 * the insert handle used for the merged rows, and the hidden row-id column
 * (whose type follows the primary key, so non-numeric keys work).
 */
public class BenoStreamDBMergeTableHandle implements ConnectorMergeTableHandle {
    private final BenoStreamDBTableHandle tableHandle;
    private final BenoStreamDBInsertTableHandle insertHandle;
    private final BenoStreamDBColumnHandle rowIdColumn;

    @JsonCreator
    public BenoStreamDBMergeTableHandle(
            @JsonProperty("tableHandle") BenoStreamDBTableHandle tableHandle,
            @JsonProperty("insertHandle") BenoStreamDBInsertTableHandle insertHandle,
            @JsonProperty("rowIdColumn") BenoStreamDBColumnHandle rowIdColumn) {
        this.tableHandle = tableHandle;
        this.insertHandle = insertHandle;
        this.rowIdColumn = rowIdColumn;
    }

    @Override
    @JsonProperty
    public BenoStreamDBTableHandle getTableHandle() {
        return tableHandle;
    }

    @JsonProperty
    public BenoStreamDBInsertTableHandle getInsertHandle() {
        return insertHandle;
    }

    @JsonProperty
    public BenoStreamDBColumnHandle getRowIdColumn() {
        return rowIdColumn;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        BenoStreamDBMergeTableHandle that = (BenoStreamDBMergeTableHandle) o;
        return Objects.equals(tableHandle, that.tableHandle)
                && Objects.equals(insertHandle, that.insertHandle)
                && Objects.equals(rowIdColumn, that.rowIdColumn);
    }

    @Override
    public int hashCode() {
        return Objects.hash(tableHandle, insertHandle, rowIdColumn);
    }
}
