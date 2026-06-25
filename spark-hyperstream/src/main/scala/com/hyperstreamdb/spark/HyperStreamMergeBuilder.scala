package com.hyperstreamdb.spark

import org.apache.spark.sql.connector.write.{RowLevelOperation, RowLevelOperationBuilder, RowLevelOperationInfo, LogicalWriteInfo}

class HyperStreamMergeBuilder(
    table: HyperStreamTable,
    info: RowLevelOperationInfo,
    gpuDevice: String
) extends RowLevelOperationBuilder {

  override def build(): RowLevelOperation = {
    // We return our custom RowLevelOperation that handles Merge-on-Read
    new HyperStreamRowLevelOperation(table, info, gpuDevice)
  }
}


