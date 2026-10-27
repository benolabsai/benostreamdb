package com.benostreamdb.trino;

import com.google.common.collect.ImmutableList;
import io.trino.spi.Plugin;
import io.trino.spi.connector.ConnectorFactory;

import java.util.Set;

public class BenoStreamDBPlugin implements Plugin {
    @Override
    public Iterable<ConnectorFactory> getConnectorFactories() {
        return ImmutableList.of(new BenoStreamDBConnectorFactory());
    }

    /**
     * Scalar functions are contributed here, not through connector metadata.
     * Trino derives their signatures from the annotations on the class, so they
     * are visible to every session without any catalog configuration.
     */
    @Override
    public Set<Class<?>> getFunctions() {
        return Set.of(BenoStreamScalarFunctions.class);
    }
}
