package com.benostreamdb.trino;

import io.trino.spi.connector.ConnectorSplit;
import io.trino.spi.HostAddress;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Collections;

/**
 * A single scan of a BenoStreamDB table.
 *
 * The split carries the table URI and the SQL query (with the pushed-down
 * predicate) that the page source runs through the engine's session. The engine
 * does the parallelism internally, so there is no file-range split fan-out.
 */
public class BenoStreamDBSplit implements ConnectorSplit {
    private final String tableUri;
    private final String sql;

    @JsonCreator
    public BenoStreamDBSplit(
        @JsonProperty("tableUri") String tableUri,
        @JsonProperty("sql") String sql) {
        this.tableUri = tableUri;
        this.sql = sql;
    }

    @Override
    public boolean isRemotelyAccessible() {
        return true;
    }

    @Override
    public List<HostAddress> getAddresses() {
        return Collections.emptyList();
    }

    @Override
    public Object getInfo() {
        return this;
    }

    @JsonProperty
    public String getTableUri() {
        return tableUri;
    }

    @JsonProperty
    public String getSql() {
        return sql;
    }
}
