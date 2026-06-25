package com.hyperstreamdb.trino;

import io.trino.spi.connector.*;
import java.util.List;

public class HyperStreamDBPageSourceProvider implements ConnectorPageSourceProvider {
    private final String gpuDevice;

    public HyperStreamDBPageSourceProvider(String gpuDevice) {
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

        HyperStreamDBSplit hSplit = (HyperStreamDBSplit) split;
        return new HyperStreamDBPageSource(hSplit, columns, gpuDevice);
    }
}
