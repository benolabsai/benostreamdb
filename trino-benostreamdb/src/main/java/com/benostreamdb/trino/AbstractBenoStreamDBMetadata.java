package com.benostreamdb.trino;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.trino.spi.StandardErrorCode;
import io.trino.spi.TrinoException;
import io.trino.spi.connector.*;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;
import io.trino.spi.function.table.Descriptor;
import io.trino.spi.predicate.Domain;
import io.trino.spi.statistics.ComputedStatistics;
import io.trino.spi.predicate.TupleDomain;
import io.trino.spi.type.ArrayType;
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

/**
 * Band-neutral connector metadata.
 *
 * <p>Every SPI method whose signature is identical across the supported Trino
 * bands lives here. The only member that cannot live here is
 * {@code finishMerge}, because its return type changed between bands:
 * Trino 482-483 declare {@code void finishMerge(...)}, while Trino 484+
 * declares {@code Optional<ConnectorOutputMetadata> finishMerge(...)}. Java
 * cannot overload on return type, so the band-specific subclass in
 * {@code src/main/java-trino-<band>} supplies exactly that one method.</p>
 *
 * <p>See {@code plans/trino_version_bands.md} for the measured band boundaries.</p>
 */
public abstract class AbstractBenoStreamDBMetadata implements ConnectorMetadata {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String warehouse;

    protected AbstractBenoStreamDBMetadata() {
        this(BenoStreamDBTableUri.DEFAULT_WAREHOUSE);
    }

    protected AbstractBenoStreamDBMetadata(String warehouse) {
        this.warehouse = warehouse;
    }

    @Override
    public List<String> listSchemaNames(ConnectorSession session) {
        requireNative();
        return parseStringList(BenoStreamDBJNIBridge.listSchemas(warehouse));
    }

    @Override
    public void createSchema(ConnectorSession session, String schemaName, Map<String, Object> properties,
            io.trino.spi.security.TrinoPrincipal owner) {
        requireNative();
        if (!BenoStreamDBJNIBridge.createSchema(warehouse, schemaName)) {
            throw new RuntimeException("BenoStreamDB createSchema failed for " + schemaName);
        }
    }

    @Override
    public void createTable(
            ConnectorSession session,
            ConnectorTableMetadata tableMetadata,
            SaveMode saveMode) {
        requireNative();
        SchemaTableName table = tableMetadata.getTable();
        String uri = BenoStreamDBTableUri.of(warehouse, table.getSchemaName(), table.getTableName());
        List<Map<String, Object>> fields = new ArrayList<>();
        for (ColumnMetadata column : tableMetadata.getColumns()) {
            Map<String, Object> field = new LinkedHashMap<>();
            field.put("name", column.getName());
            field.put("type", arrowTypeName(column.getType()));
            field.put("nullable", column.isNullable());
            fields.add(field);
        }
        try {
            String json = MAPPER.writeValueAsString(fields);
            if (!BenoStreamDBJNIBridge.createTable(uri, json)) {
                throw new TrinoException(
                        StandardErrorCode.GENERIC_INTERNAL_ERROR,
                        "BenoStreamDB createTable failed for " + table);
            }
        } catch (JsonProcessingException e) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "Failed to serialize table schema for " + table,
                    e);
        }
    }

    @Override
    public void dropTable(ConnectorSession session, ConnectorTableHandle tableHandle) {
        requireNative();
        BenoStreamDBTableHandle handle = (BenoStreamDBTableHandle) tableHandle;
        String uri = BenoStreamDBTableUri.of(warehouse, handle.getSchemaName(), handle.getTableName());
        if (!BenoStreamDBJNIBridge.dropTable(uri)) {
            throw new RuntimeException("BenoStreamDB dropTable failed for " + uri);
        }
    }

    @Override
    public ConnectorTableHandle getTableHandle(ConnectorSession session, SchemaTableName tableName,
            Optional<ConnectorTableVersion> startVersion, Optional<ConnectorTableVersion> endVersion) {
        // Verify the table exists so Trino reports "table not found" up front
        // instead of failing later during split generation.
        requireNative();
        String uri = BenoStreamDBTableUri.of(warehouse, tableName.getSchemaName(), tableName.getTableName());
        String json = BenoStreamDBJNIBridge.getTableSchema(uri);
        // Table::new succeeds for a non-existent path and returns an empty
        // schema ("[]"), so treat an empty schema as "table not found".
        if (json == null || json.isEmpty() || json.equals("[]")) {
            return null;
        }
        return new BenoStreamDBTableHandle(tableName.getSchemaName(), tableName.getTableName());
    }

    @Override
    public ConnectorTableMetadata getTableMetadata(ConnectorSession session, ConnectorTableHandle table) {
        if (table instanceof RegionalDriftFunction.Handle driftHandle) {
            List<ColumnMetadata> columns = new ArrayList<>();
            for (Descriptor.Field field : RegionalDriftFunction.descriptor().getFields()) {
                columns.add(new ColumnMetadata(
                        field.getName().orElseThrow(),
                        field.getType().orElseThrow()));
            }
            return new ConnectorTableMetadata(
                    new SchemaTableName(driftHandle.getSchemaName(), driftHandle.getTableName()),
                    columns);
        }

        if (table instanceof ListIndexesTableHandle listHandle) {
            // Metadata-only function: columns come from the plan descriptor, not
            // from the table's Parquet schema.
            List<ColumnMetadata> columns = new ArrayList<>();
            for (Descriptor.Field field : ListIndexesFunction.descriptor().getFields()) {
                columns.add(new ColumnMetadata(
                        field.getName().orElseThrow(),
                        field.getType().orElseThrow()));
            }
            return new ConnectorTableMetadata(
                    new SchemaTableName(listHandle.getSchemaName(), listHandle.getTableName()),
                    columns);
        }
        BenoStreamDBTableHandle handle = (BenoStreamDBTableHandle) table;
        return new ConnectorTableMetadata(
                new SchemaTableName(handle.getSchemaName(), handle.getTableName()),
                resolveColumns(handle.getSchemaName(), handle.getTableName()));
    }

    @Override
    public List<SchemaTableName> listTables(ConnectorSession session, Optional<String> schemaName) {
        requireNative();
        List<SchemaTableName> result = new ArrayList<>();
        List<String> schemas = schemaName.isPresent()
                ? List.of(schemaName.get())
                : listSchemaNames(session);
        for (String schema : schemas) {
            for (String table : parseStringList(BenoStreamDBJNIBridge.listTables(warehouse, schema))) {
                result.add(new SchemaTableName(schema, table));
            }
        }
        return result;
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
            ConnectorInsertTableHandle insertHandle, List<ConnectorTableHandle> sourceTableHandles,
            Collection<Slice> fragments, Collection<ComputedStatistics> computedStatistics) {
        return Optional.empty();
    }

    // -------------------------------------------------------------------------
    // DDL path (CREATE TABLE AS SELECT)
    // -------------------------------------------------------------------------

    @Override
    public ConnectorOutputTableHandle beginCreateTable(ConnectorSession session,
            ConnectorTableMetadata tableMetadata, Optional<ConnectorTableLayout> layout, RetryMode retryMode,
            boolean replace) {
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
        return mergeRowIdColumn(handle.getSchemaName(), handle.getTableName(), columns);
    }

    /**
     * The hidden merge row-id column: not one of the data columns, sourced from
     * the primary key (mirrors Iceberg's {@code $row_id}). It carries the key
     * column's own type so non-numeric keys round-trip correctly.
     */
    private BenoStreamDBColumnHandle mergeRowIdColumn(String schemaName, String tableName,
            List<ColumnMetadata> columns) {
        String keyName = resolvePrimaryKeyColumn(schemaName, tableName, columns);
        Type keyType = columns.stream()
                .filter(c -> c.getName().equals(keyName))
                .map(ColumnMetadata::getType)
                .findFirst()
                .orElse(BigintType.BIGINT);
        return BenoStreamDBColumnHandle.rowId(keyName, keyType);
    }

    /** The first primary-key column, or the first column when no PK is declared. */
    private String resolvePrimaryKeyColumn(String schemaName, String tableName, List<ColumnMetadata> columns) {
        requireNative();
        String uri = BenoStreamDBTableUri.of(warehouse, schemaName, tableName);
        String json = BenoStreamDBJNIBridge.getPrimaryKey(uri);
        if (json != null && !json.isEmpty()) {
            try {
                List<String> pk = MAPPER.readValue(json, new TypeReference<>() {
                });
                if (!pk.isEmpty()) {
                    return pk.get(0);
                }
            } catch (Exception e) {
                throw new RuntimeException(
                        "Failed to parse BenoStreamDB primary key for " + uri + ": " + e.getMessage(), e);
            }
        }
        // No declared primary key: fall back to the first column (a real column,
        // not mock data).
        return columns.isEmpty() ? "_row_id" : columns.get(0).getName();
    }

    /**
     * Trino requires this to enable MERGE; the default throws
     * "This connector does not support modifying table rows". The merge sink
     * reads full data columns per row, so use delete-then-insert (the row id is
     * a separate hidden column, so the data columns are the complete new row).
     */
    @Override
    public RowChangeParadigm getRowChangeParadigm(ConnectorSession session, ConnectorTableHandle tableHandle) {
        return RowChangeParadigm.DELETE_ROW_AND_INSERT_ROW;
    }

    /**
     * The 4-argument form is the one every supported band declares (482-483 and
     * 484+ both carry it; the 3-argument overload was dropped before 482).
     * Overriding it, like the Iceberg connector does, is what makes the planner
     * emit the post-update row in the insert slot rather than the pre-update
     * target row. The per-update-case assignments are accepted and ignored: the
     * native engine upserts by key, so the full new row is already in the
     * fragment.
     */
    @Override
    public ConnectorMergeTableHandle beginMerge(ConnectorSession session, ConnectorTableHandle tableHandle,
            Map<Integer, Collection<ColumnHandle>> updateCaseColumns, RetryMode retryMode) {
        BenoStreamDBTableHandle handle = (BenoStreamDBTableHandle) tableHandle;
        List<ColumnMetadata> columns = resolveColumns(handle.getSchemaName(), handle.getTableName());
        List<BenoStreamDBColumnHandle> cols = new ArrayList<>();
        for (ColumnMetadata col : columns) {
            cols.add(new BenoStreamDBColumnHandle(col.getName(), col.getType()));
        }
        BenoStreamDBInsertTableHandle insert = new BenoStreamDBInsertTableHandle(
                handle.getSchemaName(), handle.getTableName(), cols);
        BenoStreamDBColumnHandle rowId = mergeRowIdColumn(handle.getSchemaName(), handle.getTableName(), columns);
        return new BenoStreamDBMergeTableHandle(handle, insert, rowId);
    }

    // finishMerge intentionally lives in the band-specific subclass: its return
    // type differs between Trino 482-483 (void) and Trino 484+ (Optional).

    // -------------------------------------------------------------------------
    // Scalar functions
    // -------------------------------------------------------------------------

    // The hooks below are intentionally NOT overridden. Instrumenting
    // listFunctions showed Trino never calls it (probe count 0) for a catalog
    // created through DistributedQueryRunner.createCatalog, so connector
    // functions are never enumerated and no id or naming scheme can make them
    // resolve. Trino 484 also already ships cosine_distance, euclidean_distance,
    // hamming_distance and dot_product as built-ins, which is why those two
    // names appeared to work. See plans/trino_version_bands.md
    // ("Open: scalar functions") for the evidence trail.

    // -------------------------------------------------------------------------
    // Filter pushdown
    // -------------------------------------------------------------------------

    @Override
    public Optional<TableFunctionApplicationResult<ConnectorTableHandle>> applyTableFunction(
            ConnectorSession session,
            ConnectorTableFunctionHandle functionHandle) {
        if (functionHandle instanceof GraphRagSearchFunction.Handle ragHandle) {
            String docsUri = BenoStreamDBTableUri.of(
                    warehouse, ragHandle.getSchemaName(), ragHandle.getTableName());
            String edgeUri = BenoStreamDBTableUri.of(
                    warehouse, ragHandle.getEdgeSchemaName(), ragHandle.getEdgeTableName());
            List<ColumnHandle> columnHandles = new ArrayList<>();
            for (ColumnMetadata column : resolveColumns(
                    ragHandle.getSchemaName(), ragHandle.getTableName())) {
                columnHandles.add(new BenoStreamDBColumnHandle(column.getName(), column.getType()));
            }
            columnHandles.add(new BenoStreamDBColumnHandle("hop", IntegerType.INTEGER));
            BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle(
                    ragHandle.getSchemaName(),
                    ragHandle.getTableName(),
                    Optional.empty(),
                    Optional.of(GraphRagSql.build(ragHandle)),
                    Optional.empty(),
                    Optional.of(GraphRagSearchFunction.tablesJson(docsUri, edgeUri)));
            return Optional.of(new TableFunctionApplicationResult<>(tableHandle, columnHandles));
        }

        if (functionHandle instanceof RegionalDriftFunction.Handle driftHandle) {
            return Optional.of(new TableFunctionApplicationResult<>(driftHandle, List.of(
                    new BenoStreamDBColumnHandle(RegionalDriftPageSource.NODE_ID, BigintType.BIGINT))));
        }

        if (functionHandle instanceof ListIndexesTableHandle) {
            // Metadata-only function: the handle itself is the table handle, and
            // the columns come from the same descriptor the analysis declared.
            ListIndexesTableHandle handle = (ListIndexesTableHandle) functionHandle;
            return Optional.of(new TableFunctionApplicationResult<>(handle, List.of(
                    new BenoStreamDBColumnHandle(ListIndexesPageSource.COLUMN_NAME, VarcharType.VARCHAR),
                    new BenoStreamDBColumnHandle(ListIndexesPageSource.INDEX_CATEGORY, VarcharType.VARCHAR),
                    new BenoStreamDBColumnHandle(ListIndexesPageSource.ALGORITHM, VarcharType.VARCHAR),
                    new BenoStreamDBColumnHandle(ListIndexesPageSource.INDEX_ID, VarcharType.VARCHAR),
                    new BenoStreamDBColumnHandle(ListIndexesPageSource.FORMAT_VERSION, BigintType.BIGINT))));
        }

        if (!(functionHandle instanceof VectorSearchHandle)
                && !(functionHandle instanceof GraphTableFunctionHandle)
                && !(functionHandle instanceof HybridSearchHandle)) {
            return Optional.empty();
        }

        if (functionHandle instanceof HybridSearchHandle) {
            HybridSearchHandle handle = (HybridSearchHandle) functionHandle;
            String[] schemaTable = BenoStreamDBTableUri.splitTable(handle.getTable());
            List<ColumnMetadata> tableColumns = resolveColumns(schemaTable[0], schemaTable[1]);
            List<ColumnHandle> columnHandles = new ArrayList<>();
            for (ColumnMetadata column : tableColumns) {
                columnHandles.add(new BenoStreamDBColumnHandle(column.getName(), column.getType()));
            }
            columnHandles.add(new BenoStreamDBColumnHandle("rrf_score", RealType.REAL));
            BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle(
                    schemaTable[0],
                    schemaTable[1],
                    Optional.empty(),
                    Optional.empty(),
                    Optional.of(handle));
            return Optional.of(new TableFunctionApplicationResult<>(tableHandle, columnHandles));
        }

        if (functionHandle instanceof GraphTableFunctionHandle) {
            GraphTableFunctionHandle handle = (GraphTableFunctionHandle) functionHandle;
            String[] schemaTable = BenoStreamDBTableUri.splitTable(handle.getTable());
            List<ColumnHandle> columnHandles = new ArrayList<>(
                    GraphSql.columns(GraphSql.Kind.fromName(handle.getFunction())));
            BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle(
                    schemaTable[0],
                    schemaTable[1],
                    Optional.empty(),
                    Optional.of(GraphSql.build(handle)));
            return Optional.of(new TableFunctionApplicationResult<>(tableHandle, columnHandles));
        }

        VectorSearchHandle handle = (VectorSearchHandle) functionHandle;
        String[] schemaTable = BenoStreamDBTableUri.splitTable(handle.getTable());
        List<ColumnMetadata> tableColumns = resolveColumns(schemaTable[0], schemaTable[1]);

        List<ColumnHandle> columnHandles = new ArrayList<>();
        for (ColumnMetadata column : tableColumns) {
            columnHandles.add(new BenoStreamDBColumnHandle(column.getName(), column.getType()));
        }
        columnHandles.add(new BenoStreamDBColumnHandle("distance", DoubleType.DOUBLE));

        BenoStreamDBTableHandle tableHandle = new BenoStreamDBTableHandle(
                schemaTable[0],
                schemaTable[1],
                Optional.empty(),
                Optional.of(VectorSearchSql.build(handle)));
        return Optional.of(new TableFunctionApplicationResult<>(tableHandle, columnHandles));
    }

    @Override
    public Optional<ConstraintApplicationResult<ConnectorTableHandle>> applyFilter(
            ConnectorSession session,
            ConnectorTableHandle table,
            Constraint constraint) {

        BenoStreamDBTableHandle handle = (BenoStreamDBTableHandle) table;
        if (handle.getPushdownSql().isPresent()) {
            // The table function already produced the complete KNN statement;
            // leave any residual predicate to the Trino engine.
            return Optional.empty();
        }
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

        // Trino 468's ConstraintApplicationResult carries a remaining
        // ConnectorExpression in addition to the remaining TupleDomain. The
        // public constructor wraps it in `Optional.of(...)`, so it must be
        // non-null; `Constant.TRUE` means "no residual expression".
        return Optional.of(new ConstraintApplicationResult<ConnectorTableHandle>(
                newHandle,
                summary,
                io.trino.spi.expression.Constant.TRUE,
                false));
    }

    // -------------------------------------------------------------------------
    // Schema resolution
    // -------------------------------------------------------------------------

    List<ColumnMetadata> resolveColumns(String schemaName, String tableName) {
        requireNative();
        String uri = BenoStreamDBTableUri.of(warehouse, schemaName, tableName);
        String json = BenoStreamDBJNIBridge.getTableSchema(uri);
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("BenoStreamDB table not found: " + uri);
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

    /** Fail loudly when the native engine is unavailable — never return mock data. */
    private static void requireNative() {
        if (!BenoStreamDBJNIBridge.isLoaded()) {
            throw new IllegalStateException(
                    "BenoStreamDB native library (libbenostreamdb) is not loaded; "
                            + "add it to java.library.path");
        }
    }

    /** Parse a JSON array of strings, tolerating null/empty input. */
    private static List<String> parseStringList(String json) {
        if (json == null || json.isEmpty()) {
            return List.of();
        }
        try {
            return MAPPER.readValue(json, new TypeReference<List<String>>() {
            });
        } catch (Exception e) {
            throw new RuntimeException("Failed to parse BenoStreamDB metadata JSON: " + e.getMessage(), e);
        }
    }

    private static String arrowTypeName(Type type) {
        if (type instanceof BigintType) {
            return "Int64";
        }
        if (type instanceof IntegerType) {
            return "Int32";
        }
        if (type instanceof RealType) {
            return "Float32";
        }
        if (type instanceof DoubleType) {
            return "Float64";
        }
        if (type instanceof BooleanType) {
            return "Boolean";
        }
        if (type instanceof DateType) {
            return "Date32";
        }
        if (type instanceof VarcharType) {
            return "Utf8";
        }
        if (type instanceof ArrayType) {
            return "List(" + arrowTypeName(((ArrayType) type).getElementType()) + ")";
        }
        if (type instanceof io.trino.spi.type.RowType) {
            io.trino.spi.type.RowType row = (io.trino.spi.type.RowType) type;
            StringBuilder builder = new StringBuilder("Struct(");
            List<io.trino.spi.type.RowType.Field> fields = row.getFields();
            for (int i = 0; i < fields.size(); i++) {
                if (i > 0) {
                    builder.append(", ");
                }
                builder.append(fields.get(i).getName().orElse("field" + i))
                        .append(": ")
                        .append(arrowTypeName(fields.get(i).getType()));
            }
            return builder.append(")").toString();
        }
        return "Utf8";
    }

    private static Type trinoType(String arrowType) {
        if (arrowType.startsWith("List(") && arrowType.endsWith(")")) {
            String inner = arrowType.substring(5, arrowType.length() - 1);
            return new io.trino.spi.type.ArrayType(trinoType(inner));
        }
        if (arrowType.startsWith("FixedSizeList(") && arrowType.endsWith(")")) {
            String inner = arrowType.substring("FixedSizeList(".length(), arrowType.length() - 1);
            int comma = inner.lastIndexOf(", ");
            String element = comma > 0 ? inner.substring(0, comma) : inner;
            return new io.trino.spi.type.ArrayType(trinoType(element));
        }
        if (arrowType.startsWith("Struct(") && arrowType.endsWith(")")) {
            String inner = arrowType.substring(7, arrowType.length() - 1);
            List<io.trino.spi.type.RowType.Field> fields = new ArrayList<>();
            for (String part : inner.split(", ")) {
                int colonIdx = part.indexOf(": ");
                if (colonIdx > 0) {
                    String name = part.substring(0, colonIdx);
                    String ty = part.substring(colonIdx + 2);
                    fields.add(io.trino.spi.type.RowType.field(name, trinoType(ty)));
                }
            }
            return io.trino.spi.type.RowType.from(fields);
        }

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
