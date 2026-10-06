package com.benostreamdb.spark

import com.benostreamdb.spark.jni.BenoStreamJNIBridge
import org.apache.spark.sql.connector.catalog.{SupportsRead, SupportsRowLevelOperations, SupportsWrite, Table, TableCapability}
import org.apache.spark.sql.connector.read.ScanBuilder
import org.apache.spark.sql.connector.write.{LogicalWriteInfo, RowLevelOperationBuilder, RowLevelOperationInfo, WriteBuilder}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

import java.util.{Map => JMap, Set => JSet}
import scala.collection.JavaConverters._

/**
 * A native BenoStreamDB table. Talks to the Rust engine directly through JNI
 * (schema, scans, writes, deletes) exactly like the Trino connector — no
 * `iceberg-spark-runtime` dependency, so a single Scala-2.13 build serves Spark
 * 4.0/4.1/4.2 and a Scala-2.12 build serves 3.5.x.
 *
 * Row-level operations:
 *  - DELETE is served here via [[SupportsDelete]] (predicate -> engine deleteRows).
 *  - UPDATE / MERGE run through the engine's native SQL (pass-through), not
 *    Spark's merge-on-read framework, because the engine has no physical row-id
 *    (_file/_pos) to satisfy Spark's DeltaWriter identity contract.
 */
class BenoStreamTable(
    val tableUri: String,
    val structSchema: StructType,
    override val properties: JMap[String, String],
    val gpuDevice: String
) extends Table with SupportsRead with SupportsWrite with SupportsRowLevelOperations {

  override def name(): String = tableUri

  override def schema(): StructType = structSchema

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
}
