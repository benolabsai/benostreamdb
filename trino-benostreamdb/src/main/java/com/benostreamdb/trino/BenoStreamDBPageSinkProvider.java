package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorInsertTableHandle;
import io.trino.spi.connector.ConnectorMergeSink;
import io.trino.spi.connector.ConnectorMergeTableHandle;
import io.trino.spi.connector.ConnectorOutputTableHandle;
import io.trino.spi.connector.ConnectorPageSink;
import io.trino.spi.connector.ConnectorPageSinkId;
import io.trino.spi.connector.ConnectorPageSinkProvider;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorTransactionHandle;

public class BenoStreamDBPageSinkProvider implements ConnectorPageSinkProvider {
    private final String warehouse;
    private final String gpuDevice;

    public BenoStreamDBPageSinkProvider(String warehouse, String gpuDevice) {
        this.warehouse = warehouse;
        this.gpuDevice = gpuDevice;
    }

    @Override
    public ConnectorPageSink createPageSink(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorOutputTableHandle outputTableHandle,
            ConnectorPageSinkId pageSinkId) {
        BenoStreamDBOutputTableHandle out = (BenoStreamDBOutputTableHandle) outputTableHandle;
        BenoStreamDBInsertTableHandle insert = new BenoStreamDBInsertTableHandle(
                out.getSchemaName(), out.getTableName(), out.getColumns());
        return new BenoStreamDBPageSink(insert, warehouse, gpuDevice, true);
    }

    @Override
    public ConnectorPageSink createPageSink(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorInsertTableHandle insertTableHandle,
            ConnectorPageSinkId pageSinkId) {
        return new BenoStreamDBPageSink((BenoStreamDBInsertTableHandle) insertTableHandle, warehouse, gpuDevice);
    }

    @Override
    public ConnectorMergeSink createMergeSink(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorMergeTableHandle mergeHandle,
            ConnectorPageSinkId pageSinkId) {
        return new BenoStreamDBMergeSink((BenoStreamDBMergeTableHandle) mergeHandle, warehouse, gpuDevice);
    }
}
