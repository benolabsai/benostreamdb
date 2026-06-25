package com.hyperstreamdb.spark

import org.apache.spark.sql.connector.write.{LogicalWriteInfo, Write, WriteBuilder}

class HyperStreamWriteBuilder(
    table: HyperStreamTable,
    info: LogicalWriteInfo,
    gpuDevice: String
) extends WriteBuilder {

  override def build(): Write = {
    // This Write implementation will provide factories for PositionDeltaWriter
    // to emit `.del` position deletes using HyperStreamDB's native writers.
    new HyperStreamPositionDeltaWrite(table, info, gpuDevice)
  }
}
