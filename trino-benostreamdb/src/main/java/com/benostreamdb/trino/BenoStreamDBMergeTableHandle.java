package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorMergeTableHandle;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Handle for a MERGE into a BenoStreamDB table. Wraps the target table handle
 * and the insert handle used for the merged rows.
 */
public class BenoStreamDBMergeTableHandle implements ConnectorMergeTableHandle {
    private final BenoStreamDBTableHandle tableHandle;
    private final BenoStreamDBInsertTableHandle insertHandle;

    @JsonCreator
    public BenoStreamDBMergeTableHandle(
            @JsonProperty("tableHandle") BenoStreamDBTableHandle tableHandle,
            @JsonProperty("insertHandle") BenoStreamDBInsertTableHandle insertHandle) {
        this.tableHandle = tableHandle;
        this.insertHandle = insertHandle;
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
}
