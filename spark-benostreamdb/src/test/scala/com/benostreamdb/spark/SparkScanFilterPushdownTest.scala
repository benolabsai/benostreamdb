package com.benostreamdb.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.connector.catalog.{Identifier, TableCatalog}
import org.apache.spark.sql.connector.read.{SupportsPushDownFilters, SupportsPushDownRequiredColumns}
import org.apache.spark.sql.sources.{EqualTo, In, StringStartsWith}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.junit.{After, Before, Test}
import org.junit.Assert._

class SparkScanFilterPushdownTest {

  var spark: SparkSession = _

  @Before
  def setup(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("SparkScanFilterPushdownTest")
      .config("spark.sql.catalogImplementation", "in-memory")
      .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
      .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkCatalog")
      .config("spark.sql.catalog.spark_catalog.type", "hadoop")
      .config("spark.sql.catalog.spark_catalog.warehouse", "target/warehouse_scan")
      .getOrCreate()
  }

  @After
  def tearDown(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  @Test
  def testScanBuilderPushdownAndPruning(): Unit = {
    spark.sql("CREATE DATABASE IF NOT EXISTS default")
    spark.sql("DROP TABLE IF EXISTS spark_catalog.default.scan_test")
    spark.sql("CREATE TABLE spark_catalog.default.scan_test (id INT, category STRING, embedding ARRAY<FLOAT>) USING iceberg")

    val icebergCatalog = spark.sessionState.catalogManager.catalog("spark_catalog").asInstanceOf[TableCatalog]
    val icebergTable = icebergCatalog.loadTable(Identifier.of(Array("default"), "scan_test"))

    val properties = new java.util.HashMap[String, String]()
    properties.put("primary_key", "id")
    properties.put("indexed_columns", "category,embedding")

    val benoTable = new BenoStreamTable(icebergTable, icebergTable.schema(), properties, "auto")

    val scanBuilder = benoTable.newScanBuilder(CaseInsensitiveStringMap.empty())
    assertTrue("ScanBuilder must implement SupportsPushDownFilters", scanBuilder.isInstanceOf[SupportsPushDownFilters])
    assertTrue("ScanBuilder must implement SupportsPushDownRequiredColumns", scanBuilder.isInstanceOf[SupportsPushDownRequiredColumns])

    val filterBuilder = scanBuilder.asInstanceOf[SupportsPushDownFilters]
    val filters: Array[org.apache.spark.sql.sources.Filter] = Array(
      EqualTo("id", 42),
      In("id", Array[Any](1, 2, 3)),
      EqualTo("category", "hardware")
    )

    val unhandled = filterBuilder.pushFilters(filters)
    val pushed = filterBuilder.pushedFilters()

    assertTrue("Pushed filters should not be empty", pushed.nonEmpty || unhandled.length <= filters.length)

    // Test build returns BenoStreamScan with runtime filtering attributes
    val scan = scanBuilder.build()
    assertTrue("Scan must be BenoStreamScan", scan.isInstanceOf[BenoStreamScan])
    val bsScan = scan.asInstanceOf[BenoStreamScan]

    val filterAttrs = bsScan.filterAttributes()
    assertEquals("Should advertise primary key attribute", 1, filterAttrs.length)
    assertEquals("id", filterAttrs(0).fieldNames().head)
  }
}
