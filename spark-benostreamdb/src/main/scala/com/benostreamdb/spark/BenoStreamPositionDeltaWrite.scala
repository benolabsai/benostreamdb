package com.benostreamdb.spark

import org.apache.spark.sql.connector.write.{LogicalWriteInfo, RowLevelOperationInfo, DataWriterFactory, PhysicalWriteInfo, WriterCommitMessage, DeltaWrite, DeltaBatchWrite, DeltaWriterFactory, DeltaWriter}

class BenoStreamPositionDeltaWrite(
    table: BenoStreamTable,
    info: LogicalWriteInfo,
    gpuDevice: String
) extends DeltaWrite {

  override def toBatch(): DeltaBatchWrite = new BenoStreamDeltaBatchWrite(table, gpuDevice)
}

class BenoStreamDeltaBatchWrite(table: BenoStreamTable, gpuDevice: String) extends DeltaBatchWrite {
  override def createBatchWriterFactory(physicalInfo: PhysicalWriteInfo): DeltaWriterFactory = {
    // Return a factory that creates our custom JNI Position Delta Writers
    new BenoStreamPositionDeltaWriterFactory(table.name(), gpuDevice)
  }

  override def commit(messages: Array[WriterCommitMessage]): Unit = {
    // Commit manifest updates
  }

  override def abort(messages: Array[WriterCommitMessage]): Unit = {
    // Cleanup any orphaned delete files if the job aborts
  }
}
