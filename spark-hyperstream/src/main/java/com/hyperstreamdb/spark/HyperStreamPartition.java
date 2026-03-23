package com.hyperstreamdb.spark;

import org.apache.spark.sql.connector.read.InputPartition;

public class HyperStreamPartition implements InputPartition {
    private final String segmentId;
    private final String path;

    public HyperStreamPartition(String segmentId, String path) {
        this.segmentId = segmentId;
        this.path = path;
    }

    public String getSegmentId() {
        return segmentId;
    }

    public String getPath() {
        return path;
    }
}
