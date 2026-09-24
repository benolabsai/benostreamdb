package com.benostreamdb.spark

import org.apache.spark.sql.connector.write.{LogicalWriteInfo, Write, WriteBuilder}

class BenoStreamWriteBuilder(
    table: BenoStreamTable,
    info: LogicalWriteInfo,
    gpuDevice: String
) extends WriteBuilder {

  override def build(): Write = {
    // This Write implementation will provide factories for PositionDeltaWriter
    // to emit `.del` position deletes using BenoStreamDB's native writers.
    new BenoStreamPositionDeltaWrite(table, info, gpuDevice)
  }
}
