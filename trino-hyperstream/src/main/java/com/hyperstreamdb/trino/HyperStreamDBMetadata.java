package com.hyperstreamdb.trino;

import io.trino.spi.connector.*;
import io.trino.spi.type.VarcharType;
import io.trino.spi.type.IntegerType;

import java.util.List;
import java.util.Map;
import java.util.Optional;

public class HyperStreamDBMetadata implements ConnectorMetadata {

    @Override
    public List<String> listSchemaNames(ConnectorSession session) {
        return List.of("default");
    }

    @Override
    public ConnectorTableHandle getTableHandle(ConnectorSession session, SchemaTableName tableName) {
        return new HyperStreamDBTableHandle(tableName.getSchemaName(), tableName.getTableName());
    }

    @Override
    public ConnectorTableMetadata getTableMetadata(ConnectorSession session, ConnectorTableHandle table) {
        HyperStreamDBTableHandle handle = (HyperStreamDBTableHandle) table;
        // Mock Schema for PoC: id (int), name (varchar)
        // In real implementation, this would call Rust to get Schema from S3 (.schema)

        return new ConnectorTableMetadata(
                new SchemaTableName(handle.getSchemaName(), handle.getTableName()),
                List.of(
                        new ColumnMetadata("id", IntegerType.INTEGER),
                        new ColumnMetadata("name", VarcharType.VARCHAR)));
    }

    @Override
    public List<SchemaTableName> listTables(ConnectorSession session, Optional<String> schemaName) {
        return List.of(new SchemaTableName("default", "test_table"));
    }

    @Override
    public Map<String, ColumnHandle> getColumnHandles(ConnectorSession session, ConnectorTableHandle tableHandle) {
        // Mock column handles
        return Map.of(
                "id", new HyperStreamDBColumnHandle("id", IntegerType.INTEGER),
                "name", new HyperStreamDBColumnHandle("name", VarcharType.VARCHAR));
    }

    @Override
    public ColumnMetadata getColumnMetadata(ConnectorSession session, ConnectorTableHandle tableHandle,
            ColumnHandle columnHandle) {
        HyperStreamDBColumnHandle handle = (HyperStreamDBColumnHandle) columnHandle;
        return new ColumnMetadata(handle.getColumnName(), handle.getColumnType());
    }
}
