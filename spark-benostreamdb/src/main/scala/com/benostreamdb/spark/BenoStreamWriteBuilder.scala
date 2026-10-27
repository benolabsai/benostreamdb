package com.benostreamdb.spark

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.write._
import org.apache.spark.sql.types.StructType

/** Native write builder. */
class BenoStreamWriteBuilder(table: BenoStreamTable, info: LogicalWriteInfo, gpuDevice: String)
    extends WriteBuilder {
  override def build(): Write = new BenoStreamWrite(table.tableUri, info.schema(), gpuDevice)
}

class BenoStreamWrite(uri: String, writeSchema: StructType, gpuDevice: String) extends Write {
  override def toBatch: BatchWrite = new BenoStreamBatchWrite(uri, writeSchema, gpuDevice)
}

/**
 * Commit message carrying one task's rows as Arrow IPC bytes.
 *
 * The task never touches the engine: it encodes and hands the bytes to the
 * driver, which appends them exactly once. Replaying a task attempt therefore
 * cannot duplicate rows.
 *
 * `Vector`, not `Seq`: on Scala 2.12 `Iterator.toSeq` yields a lazy `Stream`
 * whose tail captures the iterator, which Spark cannot serialize (`not
 * serializable result: scala.collection.Iterator$$anon$10`). A `Vector` is
 * strict and serializable on both 2.12 and 2.13.
 */
case class BenoStreamCommitMessage(chunks: Vector[Array[Byte]]) extends WriterCommitMessage

class BenoStreamBatchWrite(uri: String, writeSchema: StructType, gpuDevice: String)
    extends BatchWrite {

  /**
   * Identifies *this* write attempt's staging buffer.
   *
   * Created once on the driver and carried to every task, so all commit
   * messages of the job land in the same buffer and are committed together.
   */
  private val token: String = java.util.UUID.randomUUID().toString

  override def createBatchWriterFactory(info: PhysicalWriteInfo): DataWriterFactory =
    new BenoStreamDataWriterFactory(uri, writeSchema, gpuDevice)

  /**
   * Driver-side commit: stage every accepted task's rows, then commit once.
   *
   * Two-phase on purpose — staging does not touch the table, and the native
   * commit drains its buffer before writing, so Spark retrying this method
   * cannot append the job's rows twice. The whole job becomes one snapshot.
   */
  override def commit(messages: Array[WriterCommitMessage]): Unit = {
    messages.foreach {
      case message: BenoStreamCommitMessage =>
        message.chunks.foreach(chunk => BenoStreamArrowIpc.stage(uri, token, chunk))
      case _ => ()
    }
    BenoStreamArrowIpc.commitStaged(uri, token, gpuDevice)
  }

  /** Nothing was written before the commit, so there is nothing to undo. */
  override def abort(messages: Array[WriterCommitMessage]): Unit =
    BenoStreamArrowIpc.discard(token)
}

class BenoStreamDataWriterFactory(uri: String, writeSchema: StructType, gpuDevice: String)
    extends DataWriterFactory {
  override def createWriter(partitionId: Int, taskId: Long): DataWriter[InternalRow] =
    new BenoStreamDataWriter(uri, writeSchema, gpuDevice)
}

class BenoStreamDataWriter(uri: String, writeSchema: StructType, gpuDevice: String)
    extends DataWriter[InternalRow] {

  private val batch = scala.collection.mutable.ArrayBuffer.empty[InternalRow]

  private def copy(row: InternalRow): InternalRow =
    row.asInstanceOf[org.apache.spark.sql.catalyst.expressions.UnsafeRow].copy()

  override def write(record: InternalRow): Unit = batch += copy(record)

  override def commit(): WriterCommitMessage = {
    val chunks = batch
      .grouped(BenoStreamDataWriter.ChunkRows)
      .map(chunk => BenoStreamArrowIpc.encode(chunk.toSeq, writeSchema))
      .toVector
    batch.clear()
    BenoStreamCommitMessage(chunks)
  }

  override def abort(): Unit = batch.clear()
  override def close(): Unit = ()
}

object BenoStreamDataWriter {
  /** Rows per encoded IPC chunk; bounds the transient encode buffer, not the task. */
  val ChunkRows: Int = 10000
}
