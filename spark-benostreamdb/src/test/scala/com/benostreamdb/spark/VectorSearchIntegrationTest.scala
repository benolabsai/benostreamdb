package com.benostreamdb.spark

import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.junit.Assert._
import com.benostreamdb.spark.vector.VectorDistance
import com.benostreamdb.spark.functions.BenoStreamFunctions
import com.benostreamdb.spark.implicits._

class VectorSearchIntegrationTest {

  var spark: SparkSession = _

  @Before
  def setup(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("VectorSearchTest")
      .config("spark.sql.catalogImplementation", "in-memory")
      .getOrCreate()

    BenoStreamFunctions.register(spark)
  }

  @After
  def tearDown(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  @Test
  def testVectorMathPure(): Unit = {
    val v1 = Array(1.0f, 0.0f, 0.0f)
    val v2 = Array(0.0f, 1.0f, 0.0f)
    val v3 = Array(1.0f, 0.0f, 0.0f)

    // Orthogonal: cosine distance = 1.0
    assertEquals(1.0f, VectorDistance.cosineDistance(v1, v2), 0.0001f)
    // Identical: cosine distance = 0.0
    assertEquals(0.0f, VectorDistance.cosineDistance(v1, v3), 0.0001f)

    // L2 distance
    assertEquals(0.0f, VectorDistance.l2Distance(v1, v3), 0.0001f)
    assertEquals(1.41421356f, VectorDistance.l2Distance(v1, v2), 0.0001f)

    // Dot product
    assertEquals(1.0f, VectorDistance.dotProduct(v1, v3), 0.0001f)
    assertEquals(0.0f, VectorDistance.dotProduct(v1, v2), 0.0001f)

    // Sparse dot product
    val idx1 = Array(0L, 2L, 5L)
    val val1 = Array(1.0f, 2.0f, 3.0f)
    val idx2 = Array(1L, 2L, 5L)
    val val2 = Array(4.0f, 0.5f, 2.0f)
    // matching indices: 2 (2.0*0.5=1.0) and 5 (3.0*2.0=6.0) -> total 7.0
    assertEquals(7.0f, VectorDistance.sparseDotProduct(idx1, val1, idx2, val2), 0.0001f)
  }

  @Test
  def testSparkSqlVectorFunctions(): Unit = {
    val s = spark
    import s.implicits._
    val data = Seq(
      (1, Array(1.0, 0.0, 0.0)),
      (2, Array(0.0, 1.0, 0.0)),
      (3, Array(0.7071, 0.7071, 0.0))
    ).toDF("id", "vec")
    data.createOrReplaceTempView("items")

    // Test SQL UDF cosine_distance
    val sqlDf = spark.sql("SELECT id, cosine_distance(vec, array(1.0, 0.0, 0.0)) as dist FROM items ORDER BY dist ASC")
    val rows = sqlDf.collect()
    assertEquals(1, rows(0).getInt(0)) // id 1 should be closest (dist ~ 0)
    assertTrue(rows(0).getDouble(1) < 0.0001)

    // Test DataFrame fluent vectorSearch extension
    val topK = data.vectorSearch("vec", Array(1.0, 0.0, 0.0), k = 2, metric = "cosine")
    val topKRows = topK.collect()
    assertEquals(2, topKRows.length)
    assertEquals(1, topKRows(0).getInt(0))
  }

  @Test
  def testSparseVectorAndHybridSearch(): Unit = {
    import com.benostreamdb.spark.vector.SparseVector
    val sv1 = SparseVector(Seq((10L, 0.8f), (20L, 0.5f)))
    val sv2 = SparseVector(Seq((10L, 0.5f), (30L, 0.9f)))
    // Dot product between sv1 and sv2: index 10: 0.8 * 0.5 = 0.4
    assertEquals(0.4f, sv1.dot(sv2), 0.0001f)

    // Test SQL hybrid_score
    val s = spark
    val df = s.sql("SELECT hybrid_score(0.9, 0.5, 0.7) as score")
    val res = df.collect()(0).getDouble(0)
    // 0.7 * 0.9 + 0.3 * 0.5 = 0.63 + 0.15 = 0.78
    assertEquals(0.78, res, 0.0001)

    // Test RRF
    val rrfDf = s.sql("SELECT reciprocal_rank_fusion(array(1, 3), 60) as rrf")
    val rrfVal = rrfDf.collect()(0).getDouble(0)
    // 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    assertEquals(1.0/61.0 + 1.0/63.0, rrfVal, 0.0001)
  }
}
