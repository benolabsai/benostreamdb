package com.benostreamdb.spark

import org.apache.spark.sql.connector.catalog.{SupportsRead, SupportsRowLevelOperations, Table, TableCapability}
import org.apache.spark.sql.connector.read.ScanBuilder
import org.apache.spark.sql.connector.write.{LogicalWriteInfo, RowLevelOperationBuilder, WriteBuilder}
import org.apache.spark.sql.connector.write.RowLevelOperationInfo
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import java.util.{Set => JSet}
import scala.collection.JavaConverters._

/**
 * BenoStreamTable wraps the official Iceberg SparkTable.
 * It intercepts operations where BenoStreamDB's indexes can be used (like MERGE),
 * and delegates everything else back to Iceberg.
 */
class BenoStreamTable(
    val delegate: Table,
    val tableSchema: StructType,
    override val properties: java.util.Map[String, String],
    val gpuDevice: String
) extends Table with SupportsRead with SupportsRowLevelOperations {

  override def name(): String = delegate.name()

  override def schema(): StructType = delegate.schema()

  override def capabilities(): JSet[TableCapability] = {
    // We add our row-level operation capabilities on top of the delegate's capabilities
    val caps = new java.util.HashSet[TableCapability](delegate.capabilities())
    caps.add(TableCapability.BATCH_READ)
    caps.add(TableCapability.BATCH_WRITE)
    caps.add(TableCapability.ACCEPT_ANY_SCHEMA)
    caps
  }

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder = {
    new BenoStreamScanBuilder(this, options)
  }

  override def newRowLevelOperationBuilder(info: RowLevelOperationInfo): RowLevelOperationBuilder = {
    // This is where the magic happens!
    // When Spark executes MERGE INTO, UPDATE, or DELETE, it calls this method.
    // We return our custom builder that enforces Merge-on-Read and uses our indexes.
    new BenoStreamMergeBuilder(this, info, gpuDevice)
  }
}
