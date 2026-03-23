package com.hyperstreamdb.trino;

import io.trino.spi.connector.ConnectorSplit;
import io.trino.spi.HostAddress;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Collections;

public class HyperStreamDBSplit implements ConnectorSplit {
    private final String segmentId;
    private final String path;
    private final String rowSelection;

    @JsonCreator
    public HyperStreamDBSplit(
        @JsonProperty("segmentId") String segmentId,
        @JsonProperty("path") String path,
        @JsonProperty("rowSelection") String rowSelection) {
        this.segmentId = segmentId;
        this.path = path;
        this.rowSelection = rowSelection;
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
    public String getSegmentId() { return segmentId; }

    @JsonProperty
    public String getPath() { return path; }

    @JsonProperty
    public String getRowSelection() { return rowSelection; }
}
