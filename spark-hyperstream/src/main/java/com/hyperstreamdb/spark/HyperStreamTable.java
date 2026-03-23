```java
package com.hyperstreamdb.spark;

import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

import java.util.HashSet;
import java.util.Set;

public class HyperStreamTable implements SupportsRead {

    private final StructType schema;
    private final String path;

    public HyperStreamTable(StructType schema, java.util.Map<String, String> properties) {
        this.schema = schema;
        this.path = properties.get("path");
    }

    @Override
    public ScanBuilder newScanBuilder(CaseInsensitiveStringMap options) {
        return new HyperStreamScanBuilder(schema, options);
    }

    @Override
    public String name() {
        return "hyperstream_table";
    }

    @Override
    public StructType schema() {
        return schema;
    }

    @Override
    public Set<TableCapability> capabilities() {
        Set<TableCapability> caps = new HashSet<>();
        caps.add(TableCapability.BATCH_READ); // We strictly support BATCH_READ for now
        return caps;
    }
}
