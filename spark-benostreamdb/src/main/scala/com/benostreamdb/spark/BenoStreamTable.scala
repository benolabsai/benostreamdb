package com.benostreamdb.spark

import com.benostreamdb.spark.jni.BenoStreamJNIBridge
import org.apache.spark.sql.connector.catalog.{SupportsDelete, SupportsRead, SupportsRowLevelOperations, SupportsWrite, Table, TableCapability}
import org.apache.spark.sql.connector.read.ScanBuilder
import org.apache.spark.sql.connector.write.{LogicalWriteInfo, RowLevelOperationBuilder, RowLevelOperationInfo, WriteBuilder}
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

import java.io.IOException
import java.util.{Map => JMap, Set => JSet}
import scala.collection.JavaConverters._

/**
 * A native BenoStreamDB table: reads/writes/deletes go through the Rust engine
 * via JNI (like the Trino connector) — no `iceberg-spark-runtime` dependency, so
 * one Scala-2.13 build serves Spark 4.0/4.1/4.2 and a Scala-2.12 build 3.5.x.
 *
 * Row-level operations (DELETE/UPDATE/MERGE) use Spark's `SupportsDelta`
 * merge-on-read contract with the PRIMARY KEY as the row identifier (the engine
 * has no position deletes, but does support predicate deletes + appends).
 * `SupportsDelete` is also implemented as a fallback for plain DELETE.
 */
class BenoStreamTable(
    val tableUri: String,
    val structSchema: StructType,
    override val properties: JMap[String, String],
    val gpuDevice: String
) extends Table with SupportsRead with SupportsWrite with SupportsRowLevelOperations with SupportsDelete {

  // Declared primary key (or the first column) — used as the merge-on-read row
  // identifier, which Spark requires to be non-nullable.
  private val pkColumns: Seq[String] = {
    val declared = Option(properties.get("primary_key"))
      .map(_.split(",").map(_.trim).filter(_.nonEmpty).toSeq).getOrElse(Seq.empty)
    if (declared.nonEmpty) declared else structSchema.fields.headOption.map(_.name).toSeq
  }

  override def name(): String = tableUri

  // Report PK columns as non-nullable (a primary key is non-null by definition)
  // so Spark's SupportsDelta row-id contract is satisfied.
  override def schema(): StructType =
    if (pkColumns.isEmpty) structSchema
    else StructType(structSchema.fields.map(f => if (pkColumns.contains(f.name)) f.copy(nullable = false) else f))

  override def capabilities(): JSet[TableCapability] = Set(
    TableCapability.BATCH_READ,
    TableCapability.BATCH_WRITE
  ).asJava

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder =
    new BenoStreamScanBuilder(this)

  override def newWriteBuilder(info: LogicalWriteInfo): WriteBuilder =
    new BenoStreamWriteBuilder(this, info, gpuDevice)

  override def newRowLevelOperationBuilder(info: RowLevelOperationInfo): RowLevelOperationBuilder =
    new BenoStreamRowLevelOperationBuilder(this, info, gpuDevice)

  // ---- DELETE fallback (SupportsDelete) ----
  @throws(classOf[IOException])
  def deleteWhere(filters: Array[Filter]): Unit = deleteByFilters(filters)

  @throws(classOf[IOException])
  def deleteFrom(filters: Array[Filter]): Long = { deleteByFilters(filters); -1L }

  private def deleteByFilters(filters: Array[Filter]): Unit = {
    val preds = filters.map(BenoStreamSql.filterToSql(_).getOrElse {
      throw new IOException(s"Unsupported DELETE filter for BenoStreamDB: ${filters.mkString(", ")}")
    })
    val predicate = if (preds.isEmpty) "1 = 1" else preds.mkString(" AND ")
    com.benostreamdb.spark.gpu.GpuContextResolver.bindTaskGpuContext(gpuDevice)
    if (!BenoStreamJNIBridge.getInstance().deleteRows(tableUri, predicate))
      throw new IOException(s"BenoStreamDB deleteRows failed for $tableUri")
  }
}
