package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorMergeSink;
import io.trino.spi.connector.ConnectorMergeTableHandle;
import io.trino.spi.connector.ConnectorPageSinkId;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorTableCredentials;
import io.trino.spi.connector.ConnectorTransactionHandle;
import io.trino.spi.connector.MemoryContext;

import java.util.Optional;

/**
 * Sink provider shell for the {@code e} band (Trino 484 and newer).
 *
 * <p>484 added a trailing {@code MemoryContext} parameter to
 * {@code createMergeSink}. The two create-page-sink overloads are identical in
 * both bands and live in {@link AbstractBenoStreamDBPageSinkProvider}.</p>
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
            ConnectorPageSinkId pageSinkId,
            MemoryContext memoryContext) {
        return createMergeSinkFor(mergeHandle);
    }
}
