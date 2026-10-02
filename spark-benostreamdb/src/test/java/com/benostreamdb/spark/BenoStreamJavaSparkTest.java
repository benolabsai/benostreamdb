package com.benostreamdb.spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.benostreamdb.spark.functions.BenoStreamFunctions;
import com.benostreamdb.spark.vector.SparseVector;
import com.benostreamdb.spark.vector.VectorDistance;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Java unit and integration tests verifying that Java-based Spark applications
 * can seamlessly use BenoStreamDB vector search, hybrid scoring, and sparse vectors.
 */
public class BenoStreamJavaSparkTest {

    private SparkSession spark;

    @Before
    public void setup() {
        spark = SparkSession.builder()
                .master("local[2]")
                .appName("BenoStreamJavaSparkTest")
                .config("spark.sql.catalogImplementation", "in-memory")
                .getOrCreate();

        BenoStreamFunctions.register(spark);
    }

    @After
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    public void testVectorDistanceJavaInterop() {
        float[] v1 = new float[]{1.0f, 0.0f, 0.0f};
        float[] v2 = new float[]{0.0f, 1.0f, 0.0f};
        float[] v3 = new float[]{1.0f, 0.0f, 0.0f};

        // Cosine distance
        assertEquals(1.0f, VectorDistance.cosineDistance(v1, v2), 0.0001f);
        assertEquals(0.0f, VectorDistance.cosineDistance(v1, v3), 0.0001f);

        // L2 distance
        assertEquals(0.0f, VectorDistance.l2Distance(v1, v3), 0.0001f);
        assertEquals(1.41421356f, VectorDistance.l2Distance(v1, v2), 0.0001f);

        // Dot product
        assertEquals(1.0f, VectorDistance.dotProduct(v1, v3), 0.0001f);
        assertEquals(0.0f, VectorDistance.dotProduct(v1, v2), 0.0001f);

        // Hybrid score
        assertEquals(0.78, VectorDistance.hybridScore(0.9, 0.5, 0.7), 0.0001);
    }

    @Test
    public void testSparseVectorJavaInterop() {
        long[] indices1 = new long[]{1L, 5L, 10L};
        float[] values1 = new float[]{0.5f, 1.2f, 3.0f};

        long[] indices2 = new long[]{2L, 5L, 10L};
        float[] values2 = new float[]{0.8f, 2.0f, 1.5f};

        SparseVector sv1 = new SparseVector(indices1, values1);
        SparseVector sv2 = new SparseVector(indices2, values2);

        // Matching at 5 (1.2*2.0=2.4) and 10 (3.0*1.5=4.5) -> sum = 6.9
        assertEquals(6.9f, sv1.dot(sv2), 0.0001f);
        assertNotNull(SparseVector.schema());
    }

    @Test
    public void testJavaSparkSqlExecution() {
        List<Row> rows = Arrays.asList(
                RowFactory.create(1, Arrays.asList(1.0, 0.0, 0.0)),
                RowFactory.create(2, Arrays.asList(0.0, 1.0, 0.0))
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("vec", DataTypes.createArrayType(DataTypes.DoubleType), false, Metadata.empty())
        });

        Dataset<Row> df = spark.createDataFrame(rows, schema);
        df.createOrReplaceTempView("documents");

        Dataset<Row> result = spark.sql(
                "SELECT id, cosine_distance(vec, array(1.0, 0.0, 0.0)) as dist FROM documents ORDER BY dist ASC"
        );

        List<Row> collected = result.collectAsList();
        assertEquals(2, collected.size());
        assertEquals(1, collected.get(0).getInt(0));
        assertEquals(0.0, collected.get(0).getDouble(1), 0.0001);
    }
}
