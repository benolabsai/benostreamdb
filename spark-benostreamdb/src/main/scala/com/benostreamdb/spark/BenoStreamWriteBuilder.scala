package com.benostreamdb.spark

import com.benostreamdb.spark.jni.BenoStreamJNIBridge
import org.apache.arrow.c.{ArrowArray, ArrowSchema, Data}
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.write._
import org.apache.spark.sql.types.StructType

/** Native write builder: rows are converted to Arrow and appended via JNI. */
class BenoStreamWriteBuilder(table: BenoStreamTable, info: LogicalWriteInfo, gpuDevice: String)
    extends WriteBuilder {
  override def build(): Write = new BenoStreamWrite(table.tableUri, info.schema(), gpuDevice)
}

class BenoStreamWrite(uri: String, writeSchema: StructType, gpuDevice: String) extends Write {
  override def toBatch: BatchWrite = new BenoStreamBatchWrite(uri, writeSchema, gpuDevice)
}

class BenoStreamBatchWrite(uri: String, writeSchema: StructType, gpuDevice: String) extends BatchWrite {
  override def createBatchWriterFactory(info: PhysicalWriteInfo): DataWriterFactory =
    new BenoStreamDataWriterFactory(uri, writeSchema, gpuDevice)
  override def commit(messages: Array[WriterCommitMessage]): Unit = () // engine commits per batch
  override def abort(messages: Array[WriterCommitMessage]): Unit = ()
}

class BenoStreamDataWriterFactory(uri: String, writeSchema: StructType, gpuDevice: String) extends DataWriterFactory {
  override def createWriter(partitionId: Int, taskId: Long): DataWriter[InternalRow] =
    new BenoStreamDataWriter(uri, writeSchema, gpuDevice)
}

class BenoStreamDataWriter(uri: String, writeSchema: StructType, gpuDevice: String)
    extends DataWriter[InternalRow] {

  private val bridge = BenoStreamJNIBridge.getInstance()
  private val batch = scala.collection.mutable.ArrayBuffer.empty[InternalRow]
  private val flushSize = 10000

  private def copy(r: InternalRow): InternalRow = r.asInstanceOf[org.apache.spark.sql.catalyst.expressions.UnsafeRow].copy()

  override def write(record: InternalRow): Unit = {
    batch += copy(record)
    if (batch.size >= flushSize) flush()
  }

  private def flush(): Unit = {
    if (batch.isEmpty) return
    val allocator = new RootAllocator()
    try {
      val root: VectorSchemaRoot = BenoStreamArrowWriter.toVectorSchemaRoot(batch.toSeq, writeSchema, allocator)
      val array = ArrowArray.allocateNew(allocator)
      val schema = ArrowSchema.allocateNew(allocator)
      try {
        Data.exportVectorSchemaRoot(allocator, root, null, array, schema)
        com.benostreamdb.spark.gpu.GpuContextResolver.bindTaskGpuContext(gpuDevice)
        if (!bridge.appendBatch(uri, array.memoryAddress(), schema.memoryAddress())) {
          throw new RuntimeException(s"BenoStreamDB appendBatch failed for $uri")
        }
      } finally {
        array.close(); schema.close(); root.close()
      }
    } finally {
      batch.clear()
      allocator.close()
    }
  }

  override def commit(): WriterCommitMessage = { flush(); BenoStreamCommitMessage }
  override def abort(): Unit = batch.clear()
  override def close(): Unit = batch.clear()
}

case object BenoStreamCommitMessage extends WriterCommitMessage
