package com.benostreamdb.trino;

import io.trino.spi.connector.ColumnHandle;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.Type;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Objects;

public class BenoStreamDBColumnHandle implements ColumnHandle {
    /** The name Trino uses for the hidden merge row-id column. */
    public static final String ROW_ID_COLUMN = "_bsdb_row_id";

    private final String columnName;
    private final Type columnType;
    /**
     * For the hidden row-id column, the physical column whose value identifies
     * the row (the primary key). Null for ordinary columns. Mirrors Iceberg's
     * hidden {@code $row_id}: the row id must NOT be one of the data columns,
     * otherwise Trino's delete-and-insert merge planning corrupts the new row.
     */
    private final String rowIdSourceColumn;

    @JsonCreator
    public BenoStreamDBColumnHandle(
        @JsonProperty("columnName") String columnName,
        @JsonProperty("columnType") Type columnType,
        @JsonProperty("rowIdSourceColumn") String rowIdSourceColumn) {
        this.columnName = columnName;
        this.columnType = columnType;
        this.rowIdSourceColumn = rowIdSourceColumn;
    }

    public BenoStreamDBColumnHandle(String columnName, Type columnType) {
        this(columnName, columnType, null);
    }

    /** The hidden merge row-id column, sourced from the given primary-key column. */
    public static BenoStreamDBColumnHandle rowId(String primaryKeyColumn) {
        return rowId(primaryKeyColumn, BigintType.BIGINT);
    }

    /**
     * The hidden merge row-id column, carrying the primary-key column's own type
     * so non-numeric keys (e.g. a VARCHAR primary key) round-trip correctly.
     */
    public static BenoStreamDBColumnHandle rowId(String primaryKeyColumn, Type type) {
        return new BenoStreamDBColumnHandle(ROW_ID_COLUMN, type, primaryKeyColumn);
    }

    @JsonProperty
    public String getColumnName() { return columnName; }

    @JsonProperty
    public Type getColumnType() { return columnType; }

    @JsonProperty
    public String getRowIdSourceColumn() { return rowIdSourceColumn; }

    public boolean isRowId() { return rowIdSourceColumn != null; }

    // Trino's planner compares ColumnHandles across separate getColumnHandles()
    // calls (e.g. MERGE's mergeCaseSetColumns.indexOf(dataColumnHandle)), so
    // value equality is mandatory. Without it, MERGE UPDATE silently falls back
    // to the pre-update target row.
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        BenoStreamDBColumnHandle that = (BenoStreamDBColumnHandle) o;
        return Objects.equals(columnName, that.columnName)
                && Objects.equals(columnType, that.columnType)
                && Objects.equals(rowIdSourceColumn, that.rowIdSourceColumn);
    }

    @Override
    public int hashCode() {
        return Objects.hash(columnName, columnType, rowIdSourceColumn);
    }
}
