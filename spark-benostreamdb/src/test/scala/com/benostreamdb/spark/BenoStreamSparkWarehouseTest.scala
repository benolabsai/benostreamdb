package com.benostreamdb.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.junit.{After, Assert, Before, Test}
import org.junit.Assume

import java.nio.file.Files
import scala.util.Try

/**
 * End-to-end Spark tests against a real warehouse directory.
 *
 * Every other Spark test in this module runs with an `in-memory` catalog, so the
 * V2 scan path (splits, per-file partitions, procedure scans) is never planned.
 * This class points the catalog at a temp directory and drives real SQL:
 *
 *  * two commits produce two Parquet files, and a scan must return every row
 *    exactly once;
 *  * `spark.benostream.split.maxSize` decides whether the read fans out to one
 *    partition per file or stays a single SQL scan;
 *  * `CALL ... show_indexes` must yield the five connector columns with rows
 *    (Spark 4 only — the class is probed at runtime so this file still compiles
 *    against Spark 3.5).
 */
class BenoStreamSparkWarehouseTest {

  private val procedureCatalogClass = "com.benostreamdb.spark.BenoStreamProcedureCatalog"

  private def procedureCatalogAvailable: Boolean =
    Try(Class.forName(procedureCatalogClass)).isSuccess

  // `BenoStreamProcedureCatalog` (Spark 4) adds `system.*` procedures to the same
  // table catalog; Spark 3.5 uses the base catalog.
  private def catalogClass: String =
    if (procedureCatalogAvailable) procedureCatalogClass
    else "com.benostreamdb.spark.BenoStreamCatalog"

  private var spark: SparkSession = _
  private var warehouse: java.nio.file.Path = _

  @Before
  def setUp(): Unit = {
    warehouse = Files.createTempDirectory("bsdb-spark-warehouse-")
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("BenoStreamSparkWarehouseTest")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.catalog.benostream", catalogClass)
      .config("spark.sql.catalog.benostream.warehouse", warehouse.toUri.toString)
      .config("spark.sql.warehouse.dir", warehouse.resolve("spark").toUri.toString)
      .getOrCreate()
    // No UDF registration needed: the catalog implements `FunctionCatalog`.
  }

  @After
  def tearDown(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  private def createTwoFileTable(name: String): Unit = {
    spark.sql(s"CREATE TABLE benostream.default.$name (id bigint, label string)")
    spark.sql(s"INSERT INTO benostream.default.$name VALUES (1, 'a'), (2, 'b')")
    spark.sql(s"INSERT INTO benostream.default.$name VALUES (3, 'c'), (4, 'd')")
  }

  private def idsOf(table: String): Seq[Long] =
    spark
      .table(s"benostream.default.$table")
      .select("id")
      .orderBy("id")
      .collect()
      .map(_.getLong(0))
      .toSeq

  @Test
  def scanReturnsEveryRowExactlyOnce(): Unit = {
    createTwoFileTable("split_t")

    val df = spark.table("benostream.default.split_t")
    Assert.assertEquals(
      s"rows must not duplicate or drop across files, got ${idsOf("split_t")}",
      Seq(1L, 2L, 3L, 4L),
      idsOf("split_t")
    )
    // Each data file becomes its own partition, so the partition count follows
    // the number of files the writes produced (>= 2 here) rather than the row
    // count.
    Assert.assertTrue(
      s"expected one partition per data file, got ${df.rdd.getNumPartitions}",
      df.rdd.getNumPartitions >= 2
    )
  }

  @Test
  def splitFanOutCanBeDisabled(): Unit = {
    spark.conf.set(BenoStreamScanBuilder.SplitMaxSizeConf, "0")
    try {
      createTwoFileTable("single_split_t")
      Assert.assertEquals(
        s"maxSize=0 must read the same rows through one scan, got ${idsOf("single_split_t")}",
        Seq(1L, 2L, 3L, 4L),
        idsOf("single_split_t")
      )
      Assert.assertEquals(
        "maxSize=0 must keep the single SQL scan",
        1,
        spark.table("benostream.default.single_split_t").rdd.getNumPartitions
      )
    } finally {
      spark.conf.unset(BenoStreamScanBuilder.SplitMaxSizeConf)
    }
  }

  @Test
  def filtersStillApplyWhenFanOutIsDisabled(): Unit = {
    spark.conf.set(BenoStreamScanBuilder.SplitMaxSizeConf, "0")
    try {
      createTwoFileTable("filtered_t")
      // A pushed filter disables fan-out regardless of the split setting; the
      // scan has to return the filtered subset, not the whole table.
      val ids = spark
        .table("benostream.default.filtered_t")
        .filter("id > 2")
        .select("id")
        .orderBy("id")
        .collect()
        .map(_.getLong(0))
        .toSeq
      Assert.assertEquals(Seq(3L, 4L), ids)
    } finally {
      spark.conf.unset(BenoStreamScanBuilder.SplitMaxSizeConf)
    }
  }

  /**
   * `show_indexes` must hand Spark a scan over the engine's index listing.
   *
   * `CALL` is used via reflection because the Spark 4 procedure class does not
   * exist in the Spark 3.5 build, and the index is created through the same JNI
   * bridge the procedure uses (Spark 4.0.1's `CALL` rejects parameters carrying
   * default values, which is a Spark-side limitation, not a connector one).
   */
  @Test
  def showIndexesScanCarriesTheConnectorColumns(): Unit = {
    Assume.assumeTrue(
      "show_indexes result scans exist only on the Spark 4 procedure catalog",
      procedureCatalogAvailable
    )
    spark.sql("CREATE TABLE benostream.default.idx_t (id bigint, label string)")
    spark.sql("INSERT INTO benostream.default.idx_t VALUES (1, 'graph rag'), (2, 'vector search')")

    val uri = warehouse.toUri.toString.stripSuffix("/") + "/default/idx_t"
    val bridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    Assert.assertTrue("add_index should succeed", bridge.addIndex(uri, "label", "bm25"))
    Assert.assertTrue("build_index should succeed", bridge.buildIndex(uri, "all"))

    val procClass = Class.forName("com.benostreamdb.spark.procedures.ShowIndexesProcedure")
    val procedure = procClass.getDeclaredConstructor().newInstance()
    val args = new org.apache.spark.sql.catalyst.expressions.GenericInternalRow(
      Array[Any](org.apache.spark.unsafe.types.UTF8String.fromString("default.idx_t"))
    )
    val scans = procClass
      .getMethod("call", classOf[org.apache.spark.sql.catalyst.InternalRow])
      .invoke(procedure, args)
      .asInstanceOf[java.util.Iterator[org.apache.spark.sql.connector.read.Scan]]

    Assert.assertTrue("show_indexes must return a scan", scans.hasNext)
    val scan = scans.next()

    // The five-column shape is the connector's contract with Trino's
    // `system.list_indexes`; a shape change must fail here.
    Assert.assertEquals(
      Seq("column_name", "index_category", "algorithm", "index_id", "format_version"),
      scan.readSchema().fieldNames.toSeq
    )

    // The index build is detached, so poll briefly for the listing to catch up.
    var columnNames = Seq.empty[String]
    var attempts = 0
    while (columnNames.isEmpty && attempts < 40) {
      val batch = scan.toBatch()
      val factory = batch.createReaderFactory()
      columnNames = batch
        .planInputPartitions()
        .flatMap { partition =>
          val reader = factory.createReader(partition)
          val buffer = scala.collection.mutable.ArrayBuffer.empty[InternalRow]
          while (reader.next()) {
            buffer += reader.get().copy()
          }
          reader.close()
          buffer
        }
        .map(row => row.getUTF8String(0).toString)
        .toSeq
      if (columnNames.isEmpty) {
        Thread.sleep(250)
        attempts += 1
      }
    }

    Assert.assertTrue(
      s"show_indexes must surface the built index, got $columnNames",
      columnNames.contains("label")
    )
  }
}
