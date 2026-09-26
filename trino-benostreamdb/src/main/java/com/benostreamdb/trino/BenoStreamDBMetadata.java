package com.benostreamdb.trino;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.trino.spi.connector.*;
import io.trino.spi.predicate.Domain;
import io.trino.spi.statistics.ComputedStatistics;
import io.trino.spi.predicate.TupleDomain;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.BooleanType;
import io.trino.spi.type.DateType;
import io.trino.spi.type.DoubleType;
import io.trino.spi.type.IntegerType;
import io.trino.spi.type.RealType;
import io.trino.spi.type.Type;
import io.trino.spi.type.VarcharType;
import io.airlift.slice.Slice;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class BenoStreamDBMetadata implements ConnectorMetadata {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String warehouse;

    public BenoStreamDBMetadata() {
        this(BenoStreamDBTableUri.DEFAULT_WAREHOUSE);
    }

    public BenoStreamDBMetadata(String warehouse) {
        this.warehouse = warehouse;
    }

    @Override
    public List<String> listSchemaNames(ConnectorSession session) {
        return List.of("default");
    }

    @Override
    public ConnectorTableHandle getTableHandle(ConnectorSession session, SchemaTableName tableName) {
        return new BenoStreamDBTableHandle(tableName.getSchemaName(), tableName.getTableName());
    }

    @Override
    public ConnectorTableMetadata getTableMetadata(ConnectorSession session, ConnectorTableHandle table) {
        BenoStreamDBTableHandle handle = (BenoStreamDBTableHandle) table;
        return new ConnectorTableMetadata(
                new SchemaTableName(handle.getSchemaName(), handle.getTableName()),
                resolveColumns(handle.getSchemaName(), handle.getTableName()));
    }

    @Override
    public List<SchemaTableName> listTables(ConnectorSession session, Optional<String> schemaName) {
        return List.of(new SchemaTableName("default", "test_table"));
    }

    @Override
    public Map<String, ColumnHandle> getColumnHandles(ConnectorSession session, ConnectorTableHandle tableHandle) {
        BenoStreamDBTableHandle handle = (BenoStreamDBTableHandle) tableHandle;
        Map<String, ColumnHandle> result = new LinkedHashMap<>();
        for (ColumnMetadata col : resolveColumns(handle.getSchemaName(), handle.getTableName())) {
            result.put(col.getName(), new BenoStreamDBColumnHandle(col.getName(), col.getType()));
        }
        return result;
    }

    @Override
    public ColumnMetadata getColumnMetadata(ConnectorSession session, ConnectorTableHandle tableHandle,
            ColumnHandle columnHandle) {
        BenoStreamDBColumnHandle handle = (BenoStreamDBColumnHandle) columnHandle;
        return new ColumnMetadata(handle.getColumnName(), handle.getColumnType());
    }

    // -------------------------------------------------------------------------
    // Write path (INSERT)
    // -------------------------------------------------------------------------

    @Override
    public ConnectorInsertTableHandle beginInsert(ConnectorSession session, ConnectorTableHandle tableHandle,
            List<ColumnHandle> columns, RetryMode retryMode) {
        BenoStreamDBTableHandle handle = (BenoStreamDBTableHandle) tableHandle;
        List<BenoStreamDBColumnHandle> cols = new ArrayList<>();
        for (ColumnHandle c : columns) {
            cols.add((BenoStreamDBColumnHandle) c);
        }
        return new BenoStreamDBInsertTableHandle(handle.getSchemaName(), handle.getTableName(), cols);
    }

    @Override
    public Optional<ConnectorOutputMetadata> finishInsert(ConnectorSession session,
            ConnectorInsertTableHandle insertHandle, Collection<Slice> fragments,
            Collection<ComputedStatistics> computedStatistics) {
        return Optional.empty();
    }

    // -------------------------------------------------------------------------
    // DDL path (CREATE TABLE AS SELECT)
    // -------------------------------------------------------------------------

    @Override
    public ConnectorOutputTableHandle beginCreateTable(ConnectorSession session,
            ConnectorTableMetadata tableMetadata, Optional<ConnectorTableLayout> layout, RetryMode retryMode) {
        List<BenoStreamDBColumnHandle> cols = new ArrayList<>();
        for (ColumnMetadata col : tableMetadata.getColumns()) {
            cols.add(new BenoStreamDBColumnHandle(col.getName(), col.getType()));
        }
        return new BenoStreamDBOutputTableHandle(
                tableMetadata.getTable().getSchemaName(),
                tableMetadata.getTable().getTableName(),
                cols);
    }

    @Override
    public Optional<ConnectorOutputMetadata> finishCreateTable(ConnectorSession session,
            ConnectorOutputTableHandle tableHandle, Collection<Slice> fragments,
            Collection<ComputedStatistics> computedStatistics) {
        return Optional.empty();
    }

    // -------------------------------------------------------------------------
    // MERGE / row-level path
    // -------------------------------------------------------------------------

    @Override
    public ColumnHandle getMergeRowIdColumnHandle(ConnectorSession session, ConnectorTableHandle tableHandle) {
        BenoStreamDBTableHandle handle = (BenoStreamDBTableHandle) tableHandle;
        List<ColumnMetadata> columns = resolveColumns(handle.getSchemaName(), handle.getTableName());
        // The merge row id is exposed as the target's primary-key column, so the
        // merge sink can delete matched rows by it.
        String keyName = resolvePrimaryKeyColumn(handle.getSchemaName(), handle.getTableName(), columns);
        return new BenoStreamDBColumnHandle(keyName, BigintType.BIGINT);
    }

    /** The first primary-key column, or the first column when no PK is declared. */
    private String resolvePrimaryKeyColumn(String schemaName, String tableName, List<ColumnMetadata> columns) {
        String uri = BenoStreamDBTableUri.of(warehouse, schemaName, tableName);
        if (BenoStreamDBJNIBridge.isLoaded()) {
            String json = BenoStreamDBJNIBridge.getPrimaryKey(uri);
            if (json != null && !json.isEmpty()) {
                try {
                    List<String> pk = MAPPER.readValue(json, new TypeReference<>() {
                    });
                    if (!pk.isEmpty()) {
                        return pk.get(0);
                    }
                } catch (Exception ignored) {
                    // fall through to the first column
                }
            }
        }
        return columns.isEmpty() ? "_row_id" : columns.get(0).getName();
    }

    @Override
    public ConnectorMergeTableHandle beginMerge(ConnectorSession session, ConnectorTableHandle tableHandle,
            RetryMode retryMode) {
        BenoStreamDBTableHandle handle = (BenoStreamDBTableHandle) tableHandle;
        List<BenoStreamDBColumnHandle> cols = new ArrayList<>();
        for (ColumnMetadata col : resolveColumns(handle.getSchemaName(), handle.getTableName())) {
            cols.add(new BenoStreamDBColumnHandle(col.getName(), col.getType()));
        }
        BenoStreamDBInsertTableHandle insert = new BenoStreamDBInsertTableHandle(
                handle.getSchemaName(), handle.getTableName(), cols);
        return new BenoStreamDBMergeTableHandle(handle, insert);
    }

    @Override
    public void finishMerge(ConnectorSession session, ConnectorMergeTableHandle mergeHandle,
            Collection<Slice> fragments, Collection<ComputedStatistics> computedStatistics) {
        // The merge sink already applied the appends and deletes.
    }

    // -------------------------------------------------------------------------
    // Filter pushdown
    // -------------------------------------------------------------------------

    @Override
    public Optional<ConstraintApplicationResult<ConnectorTableHandle>> applyFilter(
            ConnectorSession session,
            ConnectorTableHandle table,
            Constraint constraint) {

        BenoStreamDBTableHandle handle = (BenoStreamDBTableHandle) table;
        TupleDomain<ColumnHandle> summary = constraint.getSummary();

        if (summary.isAll()) {
            return Optional.empty();
        }

        StringBuilder filterBuilder = new StringBuilder();
        if (summary.getDomains().isPresent()) {
            Map<ColumnHandle, Domain> domains = summary.getDomains().get();
            boolean first = true;
            for (Map.Entry<ColumnHandle, Domain> entry : domains.entrySet()) {
                BenoStreamDBColumnHandle column = (BenoStreamDBColumnHandle) entry.getKey();
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
        if (handle.getFilterString().isPresent() && handle.getFilterString().get().equals(filterStr)) {
            return Optional.empty();
        }

        BenoStreamDBTableHandle newHandle = new BenoStreamDBTableHandle(
                handle.getSchemaName(),
                handle.getTableName(),
                Optional.of(filterStr));

        return Optional.of(new ConstraintApplicationResult<>(
                newHandle,
                summary,
                false));
    }

    // -------------------------------------------------------------------------
    // Schema resolution
    // -------------------------------------------------------------------------

    private List<ColumnMetadata> resolveColumns(String schemaName, String tableName) {
        String uri = BenoStreamDBTableUri.of(warehouse, schemaName, tableName);
        String json = null;
        if (BenoStreamDBJNIBridge.isLoaded()) {
            json = BenoStreamDBJNIBridge.getTableSchema(uri);
        }
        if (json == null || json.isEmpty()) {
            // Fallback for tests / when the native library is unavailable.
            return List.of(
                    new ColumnMetadata("id", IntegerType.INTEGER),
                    new ColumnMetadata("name", VarcharType.VARCHAR));
        }
        try {
            List<Map<String, Object>> fields = MAPPER.readValue(json, new TypeReference<>() {
            });
            List<ColumnMetadata> columns = new ArrayList<>();
            for (Map<String, Object> f : fields) {
                String name = String.valueOf(f.get("name"));
                String type = String.valueOf(f.get("type"));
                columns.add(new ColumnMetadata(name, trinoType(type)));
            }
            return columns;
        } catch (Exception e) {
            throw new RuntimeException("Failed to parse BenoStreamDB schema: " + e.getMessage(), e);
        }
    }

    private static Type trinoType(String arrowType) {
        switch (arrowType) {
            case "Int8":
            case "Int16":
            case "Int32":
                return IntegerType.INTEGER;
            case "Int64":
            case "UInt8":
            case "UInt16":
            case "UInt32":
            case "UInt64":
                return BigintType.BIGINT;
            case "Float16":
            case "Float32":
                return RealType.REAL;
            case "Float64":
                return DoubleType.DOUBLE;
            case "Boolean":
                return BooleanType.BOOLEAN;
            case "Date32":
            case "Date64":
                return DateType.DATE;
            case "Utf8":
            case "LargeUtf8":
            case "Utf8View":
                return VarcharType.VARCHAR;
            default:
                return VarcharType.VARCHAR;
        }
    }
}
