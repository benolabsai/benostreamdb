package com.benostreamdb.trino;

import io.airlift.slice.Slice;
import io.trino.spi.connector.ConnectorMergeTableHandle;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorTableHandle;
import io.trino.spi.statistics.ComputedStatistics;

import java.util.Collection;
import java.util.List;

/**
 * Metadata shell for the {@code d} band (Trino 482-483).
 *
 * <p>This band declares {@code finishMerge} as returning {@code void}; 484
 * changed it to {@code Optional<ConnectorOutputMetadata>}. Because Java cannot
 * overload on return type, the two bands cannot share one implementation of
 * this method, which is the entire reason this file exists. Everything else is
 * inherited from {@link AbstractBenoStreamDBMetadata}.</p>
 */
public class BenoStreamDBMetadata extends AbstractBenoStreamDBMetadata {

    public BenoStreamDBMetadata() {
        super();
    }

    public BenoStreamDBMetadata(String warehouse) {
        super(warehouse);
    }

    @Override
    public void finishMerge(
            ConnectorSession session,
            ConnectorMergeTableHandle mergeHandle,
            List<ConnectorTableHandle> sourceTableHandles,
            Collection<Slice> fragments,
            Collection<ComputedStatistics> computedStatistics) {
        // The merge sink already applied the appends and deletes.
    }
}
