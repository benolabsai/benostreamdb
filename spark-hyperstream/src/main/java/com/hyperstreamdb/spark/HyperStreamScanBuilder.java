package com.hyperstreamdb.spark;

import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import java.util.List;
import java.util.Map;
import java.util.List;
import java.util.Map;

public class HyperStreamScanBuilder implements ScanBuilder, Scan, Batch {

    private final StructType schema;
    private final CaseInsensitiveStringMap options;

    public HyperStreamScanBuilder(StructType schema, CaseInsensitiveStringMap options) {
        this.schema = schema;
        this.options = options;
    }

    @Override
    public Scan build() {
        return this;
    }

    // Scan Methods
    @Override
    public StructType readSchema() {
        return schema;
    }

    @Override
    public Batch toBatch() {
        return this;
    }

    // Batch Methods
    private static final ObjectMapper mapper = new ObjectMapper();

    static {
        try {
            System.loadLibrary("hyperstreamdb");
        } catch (UnsatisfiedLinkError e) {
            // Ignored for unit testing compilation if lib not present
            System.err.println("Native library hyperstreamdb not found: " + e.getMessage());
        }
    }

    private native String listDataFiles(String uri);

    @Override
    public InputPartition[] planInputPartitions() {
        String uri = options.get("path");
        if (uri == null) {
            // Fallback for testing or incomplete options
            uri = "s3://default/table";
        }

        System.out.println("HyperStreamScanBuilder: Planning partitions for " + uri);

        try {
            String json = listDataFiles(uri);
            if (json == null || json.isEmpty()) {
                System.out.println("HyperStreamScanBuilder: No files returned from JNI");
                return new InputPartition[0];
            }

            List<Map<String, Object>> files = mapper.readValue(json, new TypeReference<List<Map<String, Object>>>() {
            });

            return files.stream().map(f -> {
                String path = (String) f.get("file_path");
                // Simplified segment ID derivation
                String segmentId = path.substring(path.lastIndexOf('/') + 1);
                return new HyperStreamPartition(segmentId, path);
            }).toArray(InputPartition[]::new);

        } catch (UnsatisfiedLinkError e) {
            System.err.println("JNI not linked, using mock partition");
            return new InputPartition[] {
                    new HyperStreamPartition("mock_segment", "/tmp/mock.parquet")
            };
        } catch (Exception e) {
            throw new RuntimeException("Failed to plan partitions", e);
        }
    }

    @Override
    public PartitionReaderFactory createReaderFactory() {
        return new HyperStreamPartitionReaderFactory(schema);
    }
}
