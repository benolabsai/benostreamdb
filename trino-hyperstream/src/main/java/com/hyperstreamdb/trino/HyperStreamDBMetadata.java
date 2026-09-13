package com.hyperstreamdb.trino;

import io.trino.spi.connector.*;
import io.trino.spi.type.VarcharType;
import io.trino.spi.type.IntegerType;
import io.trino.spi.predicate.Domain;
import io.trino.spi.predicate.TupleDomain;

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

    @Override
    public Optional<ConstraintApplicationResult<ConnectorTableHandle>> applyFilter(
            ConnectorSession session,
            ConnectorTableHandle table,
            Constraint constraint) {
        
        HyperStreamDBTableHandle handle = (HyperStreamDBTableHandle) table;
        TupleDomain<ColumnHandle> summary = constraint.getSummary();
        
        if (summary.isAll()) {
            return Optional.empty();
        }

        // Simplistic predicate translation for PoC
        StringBuilder filterBuilder = new StringBuilder();
        if (summary.getDomains().isPresent()) {
            Map<ColumnHandle, Domain> domains = summary.getDomains().get();
            boolean first = true;
            for (Map.Entry<ColumnHandle, Domain> entry : domains.entrySet()) {
                HyperStreamDBColumnHandle column = (HyperStreamDBColumnHandle) entry.getKey();
                Domain domain = entry.getValue();
                
                if (domain.isSingleValue()) {
                    if (!first) {
                        filterBuilder.append(" AND ");
                    }
                    Object value = domain.getSingleValue();
                    if (value instanceof io.airlift.slice.Slice) {
                        value = ((io.airlift.slice.Slice) value).toStringUtf8();
                        filterBuilder.append(column.getColumnName()).append(" = '").append(value).append("'");
                    } else {
                        filterBuilder.append(column.getColumnName()).append(" = ").append(value);
                    }
                    first = false;
                }
            }
        }

        if (filterBuilder.length() == 0) {
            return Optional.empty();
        }

        String filterStr = filterBuilder.toString();
        // If the handle already has this filter, don't apply it again
        if (handle.getFilterString().isPresent() && handle.getFilterString().get().equals(filterStr)) {
            return Optional.empty();
        }

        HyperStreamDBTableHandle newHandle = new HyperStreamDBTableHandle(
                handle.getSchemaName(),
                handle.getTableName(),
                Optional.of(filterStr));

        return Optional.of(new ConstraintApplicationResult<>(
                newHandle,
                summary, // return the summary as remaining constraint so Trino still evaluates it safely
                false));
    }
}
