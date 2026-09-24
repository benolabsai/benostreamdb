package com.benostreamdb.trino;

import io.trino.spi.connector.Connector;
import io.trino.spi.connector.ConnectorContext;
import io.trino.spi.connector.ConnectorFactory;
import io.trino.spi.connector.ConnectorMetadata;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorSplitManager;
import io.trino.spi.connector.ConnectorTransactionHandle;
import io.trino.spi.transaction.IsolationLevel;
import io.trino.spi.connector.ConnectorPageSourceProvider; // Correct interface

import java.util.Map;

public class BenoStreamDBConnectorFactory implements ConnectorFactory {
    @Override
    public String getName() {
        return "benostreamdb";
    }

    @Override
    public Connector create(String catalogName, Map<String, String> config, ConnectorContext context) {
        String gpuDevice = config.getOrDefault("benostream.gpu-device", "auto");
        return new BenoStreamDBConnector(gpuDevice);
    }

    private static class BenoStreamDBConnector implements Connector {
        private final String gpuDevice;

        public BenoStreamDBConnector(String gpuDevice) {
            this.gpuDevice = gpuDevice;
        }
        @Override
        public ConnectorMetadata getMetadata(ConnectorSession session, ConnectorTransactionHandle transactionHandle) {
            return new BenoStreamDBMetadata();
        }

        @Override
        public ConnectorSplitManager getSplitManager() {
            return new BenoStreamDBSplitManager(gpuDevice);
        }

        @Override
        public ConnectorPageSourceProvider getPageSourceProvider() {
            return new BenoStreamDBPageSourceProvider(gpuDevice);
        }

        @Override
        public ConnectorTransactionHandle beginTransaction(IsolationLevel isolationLevel, boolean readOnly,
                boolean autoCommit) {
            return new BenoStreamDBTransactionHandle();
        }
    }

    public static class BenoStreamDBTransactionHandle implements ConnectorTransactionHandle {
    }
}
