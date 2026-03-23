package com.hyperstreamdb.trino;

import io.trino.spi.connector.*;
import java.util.List;
import java.util.Map;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;

public class HyperStreamDBSplitManager implements ConnectorSplitManager {

    // Load the generic library.
    // In production, use JNA or proper OS-specific loading.
    static {
        try {
            System.loadLibrary("hyperstreamdb");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load. \n" + e);
            // Fallback or exit? For PoC allow continuation but JNI calls will fail
        }
    }

    private static final ObjectMapper mapper = new ObjectMapper();

    // Check ffi.rs for signature: getSplits(String uri, long maxSplitSize) ->
    // String json
    // Java_com_hyperstreamdb_trino_HyperStreamDBSplitManager_getSplits
    private native String getSplits(String uri, long maxSplitSize);

    @Override
    public ConnectorSplitSource getSplits(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorTableHandle table,
            DynamicFilter dynamicFilter,
            Constraint constraint) {

        HyperStreamDBTableHandle tableHandle = (HyperStreamDBTableHandle) table;
        String tableName = tableHandle.getTableName();
        // Assuming simplistic URI mapping for PoC
        String uri = "s3://default/" + tableHandle.getSchemaName() + "/" + tableName;

        System.out.println("HyperStreamDBSplitManager: Computing splits for " + uri);

        try {
            // Default 64MB split size
            String jsonResult = getSplits(uri, 64 * 1024 * 1024);

            if (jsonResult == null || jsonResult.isEmpty() || jsonResult.equals("[]")) {
                return new FixedSplitSource(List.of());
            }

            // Using HyperStreamDBSplit for deserialization
            // Rust struct `Split` fields: file_path, start_offset, length, ...
            // Java `HyperStreamDBSplit`: segmentId, path, rowSelection
            // We need to map `Split` to `HyperStreamDBSplit`.
            // Rust `Split` -> JSON: { "file_path": "...", "start_offset": 0, "length": 100,
            // ... }

            // HyperStreamDBSplit expects "segmentId", "path", "rowSelection"
            // We might need a custom mapping or update HyperStreamDBSplit to match Rust
            // Split.
            // For now, let's map generic JSON to Java Split manually

            List<Map<String, Object>> rawSplits = mapper.readValue(jsonResult,
                    new TypeReference<List<Map<String, Object>>>() {
                    });

            List<ConnectorSplit> splits = rawSplits.stream().map(m -> {
                String path = (String) m.get("file_path");
                long start = ((Number) m.get("start_offset")).longValue();
                long len = ((Number) m.get("length")).longValue();
                String segmentId = path.substring(path.lastIndexOf('/') + 1);

                // Encode range in "rowSelection" or add fields to Split
                // For PoC reusing rowSelection as range "start-length"
                String selection = start + "-" + len;

                return new HyperStreamDBSplit(segmentId, path, selection);
            }).collect(java.util.stream.Collectors.toList());

            return new FixedSplitSource(splits);

        } catch (UnsatisfiedLinkError e) {
            System.err.println("JNI method not found: " + e.getMessage());
            // Mock response
            return new FixedSplitSource(List.of(
                    new HyperStreamDBSplit("mock_seg", "/tmp/mock.parquet", "all")));
        } catch (Exception e) {
            throw new RuntimeException("Failed to get splits", e);
        }
    }
}
