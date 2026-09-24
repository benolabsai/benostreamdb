package com.benostreamdb.trino;

import io.trino.spi.connector.ColumnHandle;
import io.trino.spi.type.Type;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class BenoStreamDBColumnHandle implements ColumnHandle {
    private final String columnName;
    private final Type columnType;

    @JsonCreator
    public BenoStreamDBColumnHandle(
        @JsonProperty("columnName") String columnName, 
        @JsonProperty("columnType") Type columnType) {
        this.columnName = columnName;
        this.columnType = columnType;
    }

    @JsonProperty
    public String getColumnName() { return columnName; }

    @JsonProperty
    public Type getColumnType() { return columnType; }
}
