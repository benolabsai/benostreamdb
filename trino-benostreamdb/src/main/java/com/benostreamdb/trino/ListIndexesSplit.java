package com.benostreamdb.trino;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.trino.spi.connector.ConnectorSplit;

import java.util.Objects;

/**
 * Split carrying the engine's index listing as JSON.
 *
 * <p>The listing is produced on the coordinator by {@code listIndexes}, so the
 * payload travels with the split instead of being re-read on a worker; it is
 * tiny (a handful of index records per table) and the split stays remotely
 * accessible.</p>
 */
public class ListIndexesSplit implements ConnectorSplit {

    private final String json;

    @JsonCreator
    public ListIndexesSplit(@JsonProperty("json") String json) {
        this.json = Objects.requireNonNull(json, "json is null");
    }

    @JsonProperty
    public String getJson() {
        return json;
    }

    @Override
    public boolean isRemotelyAccessible() {
        return true;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof ListIndexesSplit that && json.equals(that.json);
    }

    @Override
    public int hashCode() {
        return json.hashCode();
    }
}
