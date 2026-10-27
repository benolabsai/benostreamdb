package com.benostreamdb.trino;

import io.airlift.slice.Slices;
import io.trino.spi.connector.Connector;
import io.trino.spi.connector.ConnectorContext;
import io.trino.spi.connector.ConnectorFactory;
import io.trino.spi.connector.ConnectorMetadata;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorSplitManager;
import io.trino.spi.connector.ConnectorTransactionHandle;
import io.trino.spi.function.table.ConnectorTableFunction;
import io.trino.spi.procedure.Procedure;
import io.trino.spi.transaction.IsolationLevel;
import io.trino.spi.connector.ConnectorPageSourceProvider; // Correct interface
import io.trino.spi.connector.ConnectorPageSinkProvider;
import io.trino.spi.type.VarcharType;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class BenoStreamDBConnectorFactory implements ConnectorFactory {
    @Override
    public String getName() {
        return "benostreamdb";
    }

    @Override
    public Connector create(String catalogName, Map<String, String> config, ConnectorContext context) {
        String gpuDevice = config.getOrDefault("benostream.gpu-device", "auto");
        String warehouse = config.getOrDefault("benostream.warehouse", BenoStreamDBTableUri.DEFAULT_WAREHOUSE);
        return new BenoStreamDBConnector(warehouse, gpuDevice, new BenoStreamProcedures(warehouse));
    }

    private static class BenoStreamDBConnector implements Connector {
        private final String warehouse;
        private final String gpuDevice;
        private final BenoStreamProcedures procedures;

        public BenoStreamDBConnector(String warehouse, String gpuDevice, BenoStreamProcedures procedures) {
            this.warehouse = warehouse;
            this.gpuDevice = gpuDevice;
            this.procedures = procedures;
        }

        @Override
        public ConnectorMetadata getMetadata(ConnectorSession session, ConnectorTransactionHandle transactionHandle) {
            return new BenoStreamDBMetadata(warehouse);
        }

        @Override
        public void shutdown() {
            // No shared process state to release.
        }

        @Override
        public Set<ConnectorTableFunction> getTableFunctions() {
            return Set.of(
                    new VectorSearchFunction(warehouse),
                    new HybridSearchFunction(warehouse),
                    new GraphTableFunction(warehouse, GraphSql.Kind.NEIGHBORS),
                    new GraphTableFunction(warehouse, GraphSql.Kind.SHORTEST_PATH),
                    new GraphTableFunction(warehouse, GraphSql.Kind.SUBGRAPH),
                    new ListIndexesFunction(warehouse),
                    new RegionalDriftFunction(warehouse),
                    new GraphRagSearchFunction(warehouse));
        }

        /**
         * The full maintenance surface, matching the Spark connector's procedure
         * catalog name for name so a migration needs no rewrite. Spark exposes
         * {@code compact} and {@code optimize} as aliases; both are registered
         * here for the same reason.
         */
        @Override
        public Set<Procedure> getProcedures() {
            return Set.copyOf(List.of(
                    procedure("add_index",
                            List.of(arg("TABLE"), arg("COLUMN"), optionalArg("ALGORITHM", "vector")),
                            "addIndex", String.class, String.class, String.class),
                    procedure("build_index",
                            List.of(arg("TABLE"), optionalArg("SEGMENT_ID", "all")),
                            "buildIndex", String.class, String.class),
                    procedure("rebuild_index",
                            List.of(arg("TABLE"), arg("COLUMN")),
                            "rebuildIndex", String.class, String.class),
                    procedure("drop_index",
                            List.of(arg("TABLE"), arg("COLUMN")),
                            "dropIndex", String.class, String.class),
                    procedure("compact",
                            List.of(arg("TABLE")),
                            "compact", String.class),
                    procedure("optimize",
                            List.of(arg("TABLE")),
                            "compact", String.class),
                    procedure("set_primary_key",
                            List.of(arg("TABLE"), arg("COLUMNS")),
                            "setPrimaryKey", String.class, String.class),
                    procedure("drop_primary_key",
                            List.of(arg("TABLE"), arg("COLUMNS")),
                            "dropPrimaryKey", String.class, String.class)));
        }

        /**
         * Binds one {@code system} procedure to a {@link BenoStreamProcedures}
         * instance.
         *
         * <p>Handle parameters are {@code String} because that is what the CALL
         * execution path passes, while any declared default value must be a
         * {@link Slice} because {@code Procedure.Argument} validates it against
         * {@code VarcharType.getJavaType()}. Both rules hold at once; see
         * {@link BenoStreamProcedures}.</p>
         */
        private Procedure procedure(
                String name,
                List<Procedure.Argument> arguments,
                String method,
                Class<?>... argumentTypes) {
            Class<?>[] parameterTypes = new Class<?>[argumentTypes.length + 1];
            parameterTypes[0] = ConnectorSession.class;
            System.arraycopy(argumentTypes, 0, parameterTypes, 1, argumentTypes.length);
            try {
                MethodHandle handle = MethodHandles.lookup()
                        .findVirtual(
                                BenoStreamProcedures.class,
                                method,
                                MethodType.methodType(void.class, parameterTypes))
                        .bindTo(procedures);
                return new Procedure("system", name, arguments, handle);
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("Failed to register BenoStreamDB procedure " + name, e);
            }
        }

        private static Procedure.Argument arg(String name) {
            return new Procedure.Argument(name, VarcharType.VARCHAR);
        }

        private static Procedure.Argument optionalArg(String name, String defaultValue) {
            return new Procedure.Argument(name, VarcharType.VARCHAR, false, Slices.utf8Slice(defaultValue));
        }

        // Scalar functions are contributed by BenoStreamDBPlugin.getFunctions(),
        // not through the connector: Trino never enumerates connector functions
        // for a catalog created at runtime, so getFunctionProvider alone leaves
        // them unresolvable.

        @Override
        public ConnectorSplitManager getSplitManager() {
            // Opt-in: one split per Parquet file for plain, unfiltered scans.
            boolean fileGranularity =
                    Boolean.parseBoolean(config.getOrDefault("benostream.split.file-granularity", "false"));
            return new BenoStreamDBSplitManager(warehouse, gpuDevice, fileGranularity);
        }

        @Override
        public ConnectorPageSourceProvider getPageSourceProvider() {
            return new BenoStreamDBPageSourceProvider(gpuDevice);
        }

        @Override
        public ConnectorPageSinkProvider getPageSinkProvider() {
            return new BenoStreamDBPageSinkProvider(warehouse, gpuDevice);
        }

        @Override
        public ConnectorTransactionHandle beginTransaction(IsolationLevel isolationLevel, boolean readOnly,
                boolean autoCommit) {
            return BenoStreamDBTransactionHandle.INSTANCE;
        }
    }

    /**
     * The transaction handle is shipped to workers inside the serialized
     * {@code TaskUpdateRequest}, so it must be JSON-serializable. An enum is the
     * idiomatic Trino choice: Jackson serializes it by name, and there is no
     * state to carry.
     */
    public enum BenoStreamDBTransactionHandle implements ConnectorTransactionHandle {
        INSTANCE
    }
}
