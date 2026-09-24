package com.benostreamdb.trino;

import io.trino.spi.connector.*;
import java.util.List;
import java.util.Map;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import java.util.Optional;

public class BenoStreamDBSplitManager implements ConnectorSplitManager {
    private final String gpuDevice;

    public BenoStreamDBSplitManager(String gpuDevice) {
        this.gpuDevice = gpuDevice;
    }

    // Load the generic library.
    // In production, use JNA or proper OS-specific loading.
    static {
        try {
            System.loadLibrary("benostreamdb");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load. \n" + e);
            // Fallback or exit? For PoC allow continuation but JNI calls will fail
        }
    }

    private static final ObjectMapper mapper = new ObjectMapper();

    // Check ffi.rs for signature: getSplits(String uri, long maxSplitSize, String filter) -> String json
    // Java_com_benostreamdb_trino_BenoStreamDBSplitManager_getSplits
    private native String getSplits(String uri, long maxSplitSize, String filter);

    @Override
    public ConnectorSplitSource getSplits(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorTableHandle table,
            DynamicFilter dynamicFilter,
            Constraint constraint) {

        BenoStreamDBTableHandle tableHandle = (BenoStreamDBTableHandle) table;
        String tableName = tableHandle.getTableName();
        // Assuming simplistic URI mapping for PoC
        String uri = "s3://default/" + tableHandle.getSchemaName() + "/" + tableName;
        
        String filter = tableHandle.getFilterString().orElse("");

        System.out.println("BenoStreamDBSplitManager: Computing splits for " + uri + " with filter: " + filter);
        
        // Configure GPU backend before computing splits
        if (BenoStreamDBJNIBridge.isLoaded()) {
            BenoStreamDBJNIBridge.setGpuContext(gpuDevice);
        }

        try {
            // Default 64MB split size
            String jsonResult = getSplits(uri, 64 * 1024 * 1024, filter);

            if (jsonResult == null || jsonResult.isEmpty() || jsonResult.equals("[]")) {
                return new FixedSplitSource(List.of());
            }

            // Using BenoStreamDBSplit for deserialization
            // Rust struct `Split` fields: file_path, start_offset, length, ...
            // Java `BenoStreamDBSplit`: segmentId, path, rowSelection
            // We need to map `Split` to `BenoStreamDBSplit`.
            // Rust `Split` -> JSON: { "file_path": "...", "start_offset": 0, "length": 100,
            // ... }

            // BenoStreamDBSplit expects "segmentId", "path", "rowSelection"
            // We might need a custom mapping or update BenoStreamDBSplit to match Rust
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

                return new BenoStreamDBSplit(segmentId, path, selection);
            }).collect(java.util.stream.Collectors.toList());

            return new FixedSplitSource(splits);

        } catch (UnsatisfiedLinkError e) {
            System.err.println("JNI method not found: " + e.getMessage());
            // Mock response
            return new FixedSplitSource(List.of(
                    new BenoStreamDBSplit("mock_seg", "/tmp/mock.parquet", "all")));
        } catch (Exception e) {
            throw new RuntimeException("Failed to get splits", e);
        }
    }
}
