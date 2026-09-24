package com.benostreamdb.trino;

import io.trino.spi.connector.*;
import java.util.List;

public class BenoStreamDBPageSourceProvider implements ConnectorPageSourceProvider {
    private final String gpuDevice;

    public BenoStreamDBPageSourceProvider(String gpuDevice) {
        this.gpuDevice = gpuDevice;
    }

    @Override
    public ConnectorPageSource createPageSource(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorSplit split,
            ConnectorTableHandle table,
            List<ColumnHandle> columns,
            DynamicFilter dynamicFilter) {

        BenoStreamDBSplit hSplit = (BenoStreamDBSplit) split;
        return new BenoStreamDBPageSource(hSplit, columns, gpuDevice);
    }
}
