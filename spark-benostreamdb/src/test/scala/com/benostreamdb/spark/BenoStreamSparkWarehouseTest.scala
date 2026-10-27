package com.benostreamdb.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.types.{LongType, StringType, StructType}
import org.apache.spark.unsafe.types.UTF8String
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
  private val tableCatalogClass = "com.benostreamdb.spark.BenoStreamCatalog"

  /**
   * True only where the procedure catalog is *also* the table catalog.
   *
   * Spark 4's `BenoStreamProcedureCatalog` extends `BenoStreamCatalog` (tables +
   * `system.*` procedures). Spark 3.5's class of the same name is an Iceberg
   * `ProcedureCatalog` shim with no table support, so procedures and tables cannot
   * share one catalog there.
   */
  private def procedureCatalogHasTables: Boolean =
    Try {
      val procedureCatalog = Class.forName(procedureCatalogClass)
      Class.forName(tableCatalogClass).isAssignableFrom(procedureCatalog)
    }.getOrElse(false)

  private def catalogClass: String =
    if (procedureCatalogHasTables) procedureCatalogClass else tableCatalogClass

  private var spark: SparkSession = _
  private var warehouse: java.nio.file.Path = _

  @Before
  def setUp(): Unit = {
    warehouse = Files.createTempDirectory("bsdb-spark-warehouse-")
    // `SparkSession.getOrCreate` returns whatever session another test class left
    // behind, ignoring the catalog config below. Start from a clean session so the
    // warehouse catalog is actually installed.
    SparkSession.getActiveSession.foreach(_.stop())
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
   * `CALL add_index(...)` with no ALGORITHM, then `CALL show_indexes(...)`.
   *
   * Covers two user-visible contracts at once:
   *
   *  * a VARCHAR procedure parameter default is a *quoted SQL literal* (Spark
   *    parses `defaultValue` as an expression, so unquoted `vector` fails with
   *    `INVALID_DEFAULT_VALUE.UNRESOLVED_EXPRESSION`);
   *  * procedure output reaches the user as rows — the connector's scan has to be
   *    a `LocalScan`, otherwise Spark aborts with "Only local scans are
   *    temporarily supported as procedure output".
   */
  @Test
  def callAcceptsDefaultedArgument(): Unit = {
    Assume.assumeTrue(
      "CALL against the table catalog exists only on Spark 4",
      procedureCatalogHasTables
    )
    spark.sql("CREATE TABLE benostream.default.call_t (id bigint, label string)")
    spark.sql("INSERT INTO benostream.default.call_t VALUES (1, 'graph rag')")

    // ALGORITHM is omitted on purpose: this is the default-value path.
    spark.sql("CALL benostream.system.add_index('default.call_t', 'label')")

    // The five-column shape is the connector's contract with Trino's
    // `system.list_indexes`; a shape change must fail here.
    val listing = spark.sql("CALL benostream.system.show_indexes('default.call_t')")
    Assert.assertEquals(
      Seq("column_name", "index_category", "algorithm", "index_id", "format_version"),
      listing.columns.toSeq
    )

    var indexedColumns = Seq.empty[String]
    var attempts = 0
    while (indexedColumns.isEmpty && attempts < 40) {
      indexedColumns = spark
        .sql("CALL benostream.system.show_indexes('default.call_t')")
        .select("column_name")
        .collect()
        .map(_.getString(0))
        .toSeq
        .filter(_ == "label")
      if (indexedColumns.isEmpty) {
        Thread.sleep(250)
        attempts += 1
      }
    }
    Assert.assertTrue(
      s"add_index with the defaulted ALGORITHM must register the index, got $indexedColumns",
      indexedColumns.contains("label")
    )
  }

  /**
   * MERGE exercises the row-level write path, which now also buffers in the task
   * and applies deletes + inserts in the driver-side commit.
   */
  @Test
  def mergeAppliesDeletesAndInsertsExactlyOnce(): Unit = {
    Assume.assumeTrue(
      "row-level operations need a primary key and the Spark 4 catalog",
      procedureCatalogHasTables
    )
    spark.sql("CREATE TABLE benostream.default.merge_t (id bigint, label string)")
    spark.sql("INSERT INTO benostream.default.merge_t VALUES (1, 'a'), (2, 'b')")
    spark.sql("CALL benostream.system.set_primary_key('default.merge_t', 'id')")
    spark.sql(
      "CREATE OR REPLACE TEMP VIEW merge_src AS "
        + "SELECT * FROM VALUES (2, 'B'), (3, 'c') AS v(id, label)")

    spark.sql(
      "MERGE INTO benostream.default.merge_t t USING merge_src s ON t.id = s.id "
        + "WHEN MATCHED THEN UPDATE SET label = s.label "
        + "WHEN NOT MATCHED THEN INSERT (id, label) VALUES (s.id, s.label)")

    val rows = spark
      .table("benostream.default.merge_t")
      .orderBy("id")
      .collect()
      .map(row => (row.getLong(0), row.getString(1)))
      .toSeq
    // The updated row replaced the old one (no duplicate) and the new row landed.
    Assert.assertEquals(Seq((1L, "a"), (2L, "B"), (3L, "c")), rows)
  }

  /**
   * The driver-side commit contract, exercised directly.
   *
   * Two staged batches must land in **one** snapshot (Spark's `BatchWrite.commit`
   * is a job-wide commit, not a per-batch one), and committing the same token
   * twice — which Spark does when it retries `commit` — must not append again.
   */
  @Test
  def stagedAppendCommitsOnceAndIsRetrySafe(): Unit = {
    spark.sql("CREATE TABLE benostream.default.staged_t (id bigint, label string)")
    val uri = warehouse.toUri.toString + "/default/staged_t"
    val schema = new StructType().add("id", LongType).add("label", StringType)
    def rows(ids: Seq[Long]): Seq[InternalRow] =
      ids.map(id => InternalRow(id, UTF8String.fromString(s"v$id")))

    val token = java.util.UUID.randomUUID().toString
    BenoStreamArrowIpc.stage(uri, token, BenoStreamArrowIpc.encode(rows(Seq(1L, 2L)), schema))
    BenoStreamArrowIpc.stage(uri, token, BenoStreamArrowIpc.encode(rows(Seq(3L)), schema))
    BenoStreamArrowIpc.commitStaged(uri, token, "cpu")
    BenoStreamArrowIpc.commitStaged(uri, token, "cpu")

    Assert.assertEquals(
      "a retried commit must not append the staged rows twice",
      Seq(1L, 2L, 3L),
      idsOf("staged_t")
    )

    val tableDir = java.nio.file.Paths.get(java.net.URI.create(uri))
    val parquetFiles = {
      val stream = Files.list(tableDir)
      try {
        val out = new scala.collection.mutable.ArrayBuffer[String]()
        stream.forEach(p => if (p.toString.endsWith(".parquet")) out += p.toString)
        out.toSeq
      } finally {
        stream.close()
      }
    }
    Assert.assertEquals(
      s"both staged batches must share one commit (one data file), got $parquetFiles",
      1,
      parquetFiles.size
    )
  }
}
