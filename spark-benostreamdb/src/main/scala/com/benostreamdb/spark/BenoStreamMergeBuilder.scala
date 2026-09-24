package com.benostreamdb.spark

import org.apache.spark.sql.connector.write.{RowLevelOperation, RowLevelOperationBuilder, RowLevelOperationInfo, LogicalWriteInfo}

class BenoStreamMergeBuilder(
    table: BenoStreamTable,
    info: RowLevelOperationInfo,
    gpuDevice: String
) extends RowLevelOperationBuilder {

  override def build(): RowLevelOperation = {
    // We return our custom RowLevelOperation that handles Merge-on-Read
    new BenoStreamRowLevelOperation(table, info, gpuDevice)
  }
}


