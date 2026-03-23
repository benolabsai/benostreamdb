package com.hyperstreamdb.trino;

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

public class HyperStreamDBConnectorFactory implements ConnectorFactory {
    @Override
    public String getName() {
        return "hyperstreamdb";
    }

    @Override
    public Connector create(String catalogName, Map<String, String> config, ConnectorContext context) {
        return new HyperStreamDBConnector();
    }

    private static class HyperStreamDBConnector implements Connector {
        @Override
        public ConnectorMetadata getMetadata(ConnectorSession session, ConnectorTransactionHandle transactionHandle) {
            return new HyperStreamDBMetadata();
        }

        @Override
        public ConnectorSplitManager getSplitManager() {
            return new HyperStreamDBSplitManager();
        }

        @Override
        public ConnectorPageSourceProvider getPageSourceProvider() {
            return new HyperStreamDBPageSourceProvider();
        }

        @Override
        public ConnectorTransactionHandle beginTransaction(IsolationLevel isolationLevel, boolean readOnly,
                boolean autoCommit) {
            return new HyperStreamDBTransactionHandle();
        }
    }

    public static class HyperStreamDBTransactionHandle implements ConnectorTransactionHandle {
    }
}
