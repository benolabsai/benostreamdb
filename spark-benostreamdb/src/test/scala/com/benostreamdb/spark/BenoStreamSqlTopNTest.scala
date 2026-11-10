package com.benostreamdb.spark

import org.apache.spark.sql.sources.{EqualTo, Filter}
import org.apache.spark.sql.types._
import org.junit.Assert._
import org.junit.Test

/**
 * SQL rendering for the pushed-down TopN.
 *
 * The interesting shape is a KNN query: Spark's `SortOrder` carries a full
 * expression, so `ORDER BY cosine_distance(embedding, '[…]')` must survive into
 * the engine SQL. These are pure string assertions over the same helpers the
 * scan builder calls — no Spark session and no warehouse — so they run in the
 * shared Scala 2.12/2.13 test tree under surefire.
 */
class BenoStreamSqlTopNTest {

  private val schema = StructType(Seq(
    StructField("id", IntegerType),
    StructField("title", StringType),
    StructField("embedding", ArrayType(FloatType))
  ))

  @Test
  def noOrderingAndNoLimitProducesPlainScan(): Unit = {
    val sql = BenoStreamSql.buildScanSql(schema, schema, Array.empty)
    assertEquals("SELECT * FROM t", sql)
  }

  @Test
  def topNRendersOrderByAndLimit(): Unit = {
    val sql = BenoStreamSql.buildScanSql(
      schema, schema, Array.empty, Some("\"id\" DESC"), Some(10))
    assertEquals("SELECT * FROM t ORDER BY \"id\" DESC LIMIT 10", sql)
  }

  @Test
  def filterOrderingAndLimitComposeInOrder(): Unit = {
    val filters = Array[Filter](EqualTo("title", "graph"))
    val sql = BenoStreamSql.buildScanSql(
      schema, schema, filters, Some("cosine_distance(embedding) ASC"), Some(5))
    assertEquals(
      "SELECT * FROM t WHERE \"title\" = 'graph' ORDER BY cosine_distance(embedding) ASC LIMIT 5",
      sql)
  }

  @Test
  def projectionIsAppliedBeforeOrdering(): Unit = {
    val required = StructType(Seq(StructField("id", IntegerType)))
    val sql = BenoStreamSql.buildScanSql(
      schema, required, Array.empty, Some("\"id\" ASC"), Some(3))
    assertEquals("SELECT \"id\" FROM t ORDER BY \"id\" ASC LIMIT 3", sql)
  }

  @Test
  def limitWithoutOrderingIsStillPushed(): Unit = {
    val sql = BenoStreamSql.buildScanSql(schema, schema, Array.empty, None, Some(2))
    assertEquals("SELECT * FROM t LIMIT 2", sql)
  }
}
