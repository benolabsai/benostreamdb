package com.hyperstreamdb.spark

import org.apache.spark.sql.connector.read.ScanBuilder
import org.apache.spark.sql.connector.write.{RowLevelOperation, RowLevelOperationInfo, LogicalWriteInfo, WriteBuilder}
import org.apache.spark.sql.connector.expressions.{NamedReference, Expressions}
import org.apache.spark.sql.connector.expressions.filter.Predicate
import org.apache.spark.sql.util.CaseInsensitiveStringMap

class HyperStreamRowLevelOperation(
    table: HyperStreamTable,
    info: RowLevelOperationInfo,
    gpuDevice: String
) extends RowLevelOperation {

  override def command(): RowLevelOperation.Command = info.command()

  override def description(): String = s"HyperStream RowLevelOperation: ${info.command()}"

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder = {
    // Return a custom ScanBuilder that supports Runtime Filtering!
    // This allows us to intercept the dynamically broadcasted keys from the MERGE source.
    new HyperStreamScanBuilder(table, options)
  }

  override def newWriteBuilder(writeInfo: LogicalWriteInfo): WriteBuilder = {
    // Return a WriteBuilder that writes Position Deletes instead of Copy-on-Write
    new HyperStreamWriteBuilder(table, writeInfo, gpuDevice)
  }

  override def requiredMetadataAttributes(): Array[NamedReference] = {
    // Request Iceberg's implicit metadata columns for position deletes
    Array(
      Expressions.column("_file"),
      Expressions.column("_pos")
    )
  }
}
