package com.benostreamdb.trino;

import io.trino.spi.function.table.Descriptor;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Unit coverage for the URI helpers and graph schemas. */
public class TestGraphSqlAndTableUri {

    @Test
    void tableUriComposesWarehouseSchemaAndTable() {
        String uri = BenoStreamDBTableUri.of("s3://wh", "default", "events");
        assertTrue(uri.startsWith("s3://wh"), "uri should start with the warehouse: " + uri);
        assertTrue(uri.endsWith("/default/events"), "uri should end with schema/table: " + uri);
    }

    @Test
    void splitTableSplitsSchemaQualifiedNames() {
        String[] parts = BenoStreamDBTableUri.splitTable("default.events");
        assertEquals(2, parts.length);
        assertEquals("default", parts[0]);
        assertEquals("events", parts[1]);
    }

    @Test
    void resolveAcceptsQualifiedAndQualifiesBareNames() {
        String qualified = BenoStreamDBTableUri.resolve("s3://wh", "default.events");
        assertTrue(qualified.endsWith("/default/events"), "unexpected uri: " + qualified);

        // A bare table name is qualified with the default schema rather than
        // producing "...//events".
        String bare = BenoStreamDBTableUri.resolve("s3://wh", "events");
        assertEquals(qualified, bare);
    }

    @Test
    void graphDescriptorsMatchDocumentedColumns() {
        assertEquals(List.of("node", "hop", "seed"), names(GraphSql.descriptor(GraphSql.Kind.NEIGHBORS)));
        assertEquals(List.of("node", "hop"), names(GraphSql.descriptor(GraphSql.Kind.SHORTEST_PATH)));
        assertEquals(List.of("source", "target"), names(GraphSql.descriptor(GraphSql.Kind.SUBGRAPH)));
    }

    private static List<String> names(Descriptor descriptor) {
        return descriptor.getFields().stream()
                .map(field -> field.getName().orElseThrow())
                .toList();
    }
}
