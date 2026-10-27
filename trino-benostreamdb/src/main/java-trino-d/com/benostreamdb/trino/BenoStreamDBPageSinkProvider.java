package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorMergeSink;
import io.trino.spi.connector.ConnectorMergeTableHandle;
import io.trino.spi.connector.ConnectorPageSinkId;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorTableCredentials;
import io.trino.spi.connector.ConnectorTransactionHandle;

import java.util.Optional;

/**
 * Sink provider shell for the {@code d} band (Trino 482-483).
 *
 * <p>This band's {@code createMergeSink} has no {@code MemoryContext}
 * parameter; 484 added it. The overloads that are identical across bands live
 * in {@link AbstractBenoStreamDBPageSinkProvider}.</p>
 */
public class BenoStreamDBPageSinkProvider extends AbstractBenoStreamDBPageSinkProvider {

    public BenoStreamDBPageSinkProvider(String warehouse, String gpuDevice) {
        super(warehouse, gpuDevice);
    }

    @Override
    public ConnectorMergeSink createMergeSink(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorMergeTableHandle mergeHandle,
            Optional<ConnectorTableCredentials> credentials,
            ConnectorPageSinkId pageSinkId) {
        return createMergeSinkFor(mergeHandle);
    }
}
