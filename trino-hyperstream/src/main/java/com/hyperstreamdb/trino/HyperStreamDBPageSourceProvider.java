package com.hyperstreamdb.trino;

import io.trino.spi.connector.*;
import java.util.List;

public class HyperStreamDBPageSourceProvider implements ConnectorPageSourceProvider {

    @Override
    public ConnectorPageSource createPageSource(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorSplit split,
            ConnectorTableHandle table,
            List<ColumnHandle> columns,
            DynamicFilter dynamicFilter) {

        HyperStreamDBSplit hSplit = (HyperStreamDBSplit) split;
        return new HyperStreamDBPageSource(hSplit, columns);
    }
}
