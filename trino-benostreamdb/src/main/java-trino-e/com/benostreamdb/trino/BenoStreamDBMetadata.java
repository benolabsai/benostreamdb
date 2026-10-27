package com.benostreamdb.trino;

import io.airlift.slice.Slice;
import io.trino.spi.connector.ConnectorMergeTableHandle;
import io.trino.spi.connector.ConnectorOutputMetadata;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorTableHandle;
import io.trino.spi.statistics.ComputedStatistics;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

/**
 * Metadata shell for the {@code e} band (Trino 484 and newer).
 *
 * <p>This band declares {@code finishMerge} as returning
 * {@code Optional<ConnectorOutputMetadata>}. Everything else is inherited from
 * {@link AbstractBenoStreamDBMetadata}, so this class stays a one-method shell
 * and never needs to be kept in sync with the connector's real logic.</p>
 */
public class BenoStreamDBMetadata extends AbstractBenoStreamDBMetadata {

    public BenoStreamDBMetadata() {
        super();
    }

    public BenoStreamDBMetadata(String warehouse) {
        super(warehouse);
    }

    @Override
    public Optional<ConnectorOutputMetadata> finishMerge(
            ConnectorSession session,
            ConnectorMergeTableHandle mergeHandle,
            List<ConnectorTableHandle> sourceTableHandles,
            Collection<Slice> fragments,
            Collection<ComputedStatistics> computedStatistics) {
        // The merge sink already applied the appends and deletes.
        return Optional.empty();
    }
}
