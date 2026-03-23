package com.hyperstreamdb.spark;

import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableProvider;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

import java.util.Map;

public class DefaultSource implements TableProvider {

    @Override
    public StructType inferSchema(CaseInsensitiveStringMap options) {
        // In reality, call JNI or ObjectStore to get schema
        return new StructType()
                .add("id", "integer")
                .add("name", "string");
    }

    @Override
    public Table getTable(StructType schema, Transform[] partitioning, Map<String, String> properties) {
        return new HyperStreamTable(schema, properties);
    }

    @Override
    public boolean supportsExternalMetadata() {
        return true;
    }
}
