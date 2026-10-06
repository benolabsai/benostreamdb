package com.benostreamdb.spark

import com.benostreamdb.spark.jni.BenoStreamJNIBridge
import org.apache.arrow.c.{ArrowArray, ArrowSchema, Data}
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.expressions.{Expressions, NamedReference}
import org.apache.spark.sql.connector.read.ScanBuilder
import org.apache.spark.sql.connector.write._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/**
 * True merge-on-read row-level operations. The scan emits `_file`/`_pos`
 * (segment path + row offset) as required metadata; deletes/updates are
 * committed as position deletes via the engine (`commitPositionDeletes`), and
 * inserted/updated-new rows are appended via `appendBatch`. This mirrors
 * Iceberg's model without depending on Iceberg.
 */
class BenoStreamRowLevelOperationBuilder(table: BenoStreamTable, info: RowLevelOperationInfo, gpuDevice: String)
    extends RowLevelOperationBuilder {
  override def build(): RowLevelOperation = new BenoStreamRowLevelOperation(table, info, gpuDevice)
}

class BenoStreamRowLevelOperation(table: BenoStreamTable, info: RowLevelOperationInfo, gpuDevice: String)
    extends RowLevelOperation {
  override def command(): RowLevelOperation.Command = info.command()
  override def description(): String = s"BenoStreamDB native ${info.command()} on ${table.tableUri}"
  override def requiredMetadataAttributes(): Array[NamedReference] =
    Array(Expressions.column("_file"), Expressions.column("_pos"))
  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder =
    new BenoStreamScanBuilder(table)
  override def newWriteBuilder(info: LogicalWriteInfo): WriteBuilder =
    new BenoStreamDeltaWriteBuilder(table.tableUri, table.structSchema, gpuDevice)
}

class BenoStreamDeltaWriteBuilder(uri: String, dataSchema: StructType, gpuDevice: String) extends DeltaWriteBuilder {
  override def build(): DeltaWrite = new BenoStreamDeltaWrite(uri, dataSchema, gpuDevice)
}

class BenoStreamDeltaWrite(uri: String, dataSchema: StructType, gpuDevice: String) extends DeltaWrite {
  override def toBatch(): DeltaBatchWrite = new BenoStreamDeltaBatchWrite(uri, dataSchema, gpuDevice)
}

class BenoStreamDeltaBatchWrite(uri: String, dataSchema: StructType, gpuDevice: String) extends DeltaBatchWrite {
  override def createBatchWriterFactory(info: PhysicalWriteInfo): DeltaWriterFactory =
    new BenoStreamDeltaWriterFactory(uri, dataSchema, gpuDevice)
  override def commit(messages: Array[WriterCommitMessage]): Unit = ()
  override def abort(messages: Array[WriterCommitMessage]): Unit = ()
}

class BenoStreamDeltaWriterFactory(uri: String, dataSchema: StructType, gpuDevice: String) extends DeltaWriterFactory {
  override def createWriter(partitionId: Int, taskId: Long): DeltaWriter[InternalRow] =
    new BenoStreamDeltaWriter(uri, dataSchema, gpuDevice)
}

class BenoStreamDeltaWriter(uri: String, dataSchema: StructType, gpuDevice: String) extends DeltaWriter[InternalRow] {

  private val bridge = BenoStreamJNIBridge.getInstance()
  // file -> positions to delete
  private val deletes = scala.collection.mutable.LinkedHashMap.empty[String, scala.collection.mutable.ArrayBuffer[Long]]
  private val inserts = scala.collection.mutable.ArrayBuffer.empty[InternalRow]

  private def recordDelete(metadata: InternalRow): Unit = {
    // metadata row = [_file: UTF8String, _pos: Long]
    if (metadata == null || metadata.numFields < 2) return
    val file = metadata.getUTF8String(0).toString
    val pos = metadata.getLong(1)
    deletes.getOrElseUpdate(file, scala.collection.mutable.ArrayBuffer.empty) += pos
  }

  override def delete(metadata: InternalRow, id: InternalRow): Unit = recordDelete(metadata)

  override def update(metadata: InternalRow, id: InternalRow, row: InternalRow): Unit = {
    recordDelete(metadata)
    inserts += row.copy()
  }

  override def insert(row: InternalRow): Unit = inserts += row.copy()

  override def commit(): WriterCommitMessage = {
    if (deletes.nonEmpty) {
      val jmap = new java.util.LinkedHashMap[String, java.util.List[Long]]()
      deletes.foreach { case (f, ps) =>
        val l = new java.util.ArrayList[Long](); ps.foreach(l.add); jmap.put(f, l)
      }
      val json = new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(jmap)
      com.benostreamdb.spark.gpu.GpuContextResolver.bindTaskGpuContext(gpuDevice)
      if (!bridge.commitPositionDeletes(uri, json)) throw new RuntimeException(s"commitPositionDeletes failed for $uri")
    }
    if (inserts.nonEmpty) {
      val allocator = new RootAllocator()
      try {
        val root: VectorSchemaRoot = BenoStreamArrowWriter.toVectorSchemaRoot(inserts.toSeq, dataSchema, allocator)
        val array = ArrowArray.allocateNew(allocator)
        val schema = ArrowSchema.allocateNew(allocator)
        try {
          Data.exportVectorSchemaRoot(allocator, root, null, array, schema)
          com.benostreamdb.spark.gpu.GpuContextResolver.bindTaskGpuContext(gpuDevice)
          if (!bridge.appendBatch(uri, array.memoryAddress(), schema.memoryAddress()))
            throw new RuntimeException(s"appendBatch failed for $uri")
        } finally { array.close(); schema.close(); root.close() }
      } finally allocator.close()
    }
    BenoStreamCommitMessage
  }

  override def abort(): Unit = { deletes.clear(); inserts.clear() }
  override def close(): Unit = { deletes.clear(); inserts.clear() }
}
