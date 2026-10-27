package com.benostreamdb.trino;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.airlift.slice.Slices;
import io.trino.spi.Page;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.connector.ConnectorPageSource;
import io.trino.spi.connector.SourcePage;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.VarcharType;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Turns the engine's index listing (a JSON array of index-file records) into a
 * single Trino page.
 *
 * <p>No Arrow and no scan: this path only reads manifest metadata, so building
 * the page directly is both simpler and cheaper than going through the batch
 * reader used for table scans.</p>
 */
public class ListIndexesPageSource implements ConnectorPageSource {

    static final String COLUMN_NAME = "column_name";
    static final String INDEX_CATEGORY = "index_category";
    static final String ALGORITHM = "algorithm";
    static final String INDEX_ID = "index_id";
    static final String FORMAT_VERSION = "format_version";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final Page page;
    private boolean finished;

    public ListIndexesPageSource(String json) {
        this.page = buildPage(json);
    }

    private static Page buildPage(String json) {
        List<JsonNode> records = new ArrayList<>();
        try {
            JsonNode root = MAPPER.readTree(json == null || json.isBlank() ? "[]" : json);
            if (root != null && root.isArray()) {
                root.forEach(records::add);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to parse BenoStreamDB index listing", e);
        }

        BlockBuilder columnName = VarcharType.VARCHAR.createBlockBuilder(null, records.size());
        BlockBuilder indexCategory = VarcharType.VARCHAR.createBlockBuilder(null, records.size());
        BlockBuilder algorithm = VarcharType.VARCHAR.createBlockBuilder(null, records.size());
        BlockBuilder indexId = VarcharType.VARCHAR.createBlockBuilder(null, records.size());
        BlockBuilder formatVersion = BigintType.BIGINT.createBlockBuilder(null, records.size());

        for (JsonNode record : records) {
            VarcharType.VARCHAR.writeSlice(columnName, Slices.utf8Slice(text(record, COLUMN_NAME)));
            VarcharType.VARCHAR.writeSlice(indexCategory, Slices.utf8Slice(text(record, INDEX_CATEGORY)));
            VarcharType.VARCHAR.writeSlice(algorithm, Slices.utf8Slice(text(record, ALGORITHM)));
            VarcharType.VARCHAR.writeSlice(indexId, Slices.utf8Slice(text(record, INDEX_ID)));
            BigintType.BIGINT.writeLong(formatVersion, record.path(FORMAT_VERSION).asLong(0));
        }

        return new Page(
                columnName.build(),
                indexCategory.build(),
                algorithm.build(),
                indexId.build(),
                formatVersion.build());
    }

    private static String text(JsonNode record, String field) {
        JsonNode value = record.get(field);
        return value == null || value.isNull() ? "" : value.asText();
    }

    @Override
    public SourcePage getNextSourcePage() {
        if (finished) {
            return null;
        }
        finished = true;
        return SourcePage.create(page);
    }

    @Override
    public boolean isFinished() {
        return finished;
    }

    @Override
    public long getCompletedBytes() {
        return 0;
    }

    @Override
    public long getReadTimeNanos() {
        return 0;
    }

    @Override
    public long getMemoryUsage() {
        return 0;
    }

    @Override
    public void close() {
    }
}
