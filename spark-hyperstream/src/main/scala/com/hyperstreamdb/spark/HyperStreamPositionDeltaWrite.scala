package com.hyperstreamdb.spark

import org.apache.spark.sql.connector.write.{LogicalWriteInfo, RowLevelOperationInfo, DataWriterFactory, PhysicalWriteInfo, WriterCommitMessage, DeltaWrite, DeltaBatchWrite, DeltaWriterFactory, DeltaWriter}

class HyperStreamPositionDeltaWrite(
    table: HyperStreamTable,
    info: LogicalWriteInfo,
    gpuDevice: String
) extends DeltaWrite {

  override def toBatch(): DeltaBatchWrite = new HyperStreamDeltaBatchWrite(table, gpuDevice)
}

class HyperStreamDeltaBatchWrite(table: HyperStreamTable, gpuDevice: String) extends DeltaBatchWrite {
  override def createBatchWriterFactory(physicalInfo: PhysicalWriteInfo): DeltaWriterFactory = {
    // Return a factory that creates our custom JNI Position Delta Writers
    new HyperStreamPositionDeltaWriterFactory(table.name(), gpuDevice)
  }

  override def commit(messages: Array[WriterCommitMessage]): Unit = {
    // Commit manifest updates
  }

  override def abort(messages: Array[WriterCommitMessage]): Unit = {
    // Cleanup any orphaned delete files if the job aborts
  }
}
