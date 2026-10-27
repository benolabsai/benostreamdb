package com.benostreamdb.spark

import com.benostreamdb.spark.jni.BenoStreamJNIBridge
import org.apache.arrow.c.{ArrowArray, ArrowSchema, Data}
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.types.StructType

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.nio.channels.Channels

/**
 * Task-side Arrow IPC encoding and driver-side replay for writes.
 *
 * Spark calls `DataWriter.commit()` inside the **task** and
 * `BatchWrite.commit(messages)` once on the **driver**. Writing to the engine
 * from the task is not idempotent: a task that appends successfully and is then
 * retried (the classic "commit acknowledged but lost" case) appends its rows
 * twice. So the task only *encodes* its rows — as Arrow IPC bytes carried in the
 * commit message — and the driver replays exactly the messages Spark accepted.
 *
 * Replay is two-phase, which keeps the Spark contract:
 *
 *  1. [[stage]] hands each decoded batch to the native staging buffer, keyed by
 *     a per-write token (no engine mutation);
 *  2. [[commitStaged]] writes and commits the whole buffer in **one** snapshot.
 *
 * The native side drains the buffer *before* writing, so a repeated commit (Spark
 * retries `BatchWrite.commit` on failure) is a no-op rather than a second append,
 * and a job with 20 partitions produces one manifest commit, not 20.
 *
 * Allocation: one process-wide `RootAllocator`, never closed. Buffers exported
 * through the C Data interface are owned by the native engine from that point on,
 * so a per-commit allocator would report them as leaked when it is closed
 * (`Memory was leaked by query`). A single long-lived allocator is what Spark's
 * own Arrow paths do as well.
 */
object BenoStreamArrowIpc {

  /** Tables whose tasks write data files and whose driver commits metadata. */
  val FilesMode: String = "files"

  /** Tables that need commit-time coordination (Iceberg v3 `_row_id`); rows are shipped instead. */
  val BufferedMode: String = "buffered"

  private val allocator = new RootAllocator()

  private def bridge: BenoStreamJNIBridge = BenoStreamJNIBridge.getInstance()

  /** Which write protocol this table supports (see `Table::write_mode` in the engine). */
  def writeMode(tableUri: String): String = {
    val mode = bridge.writeMode(tableUri)
    if (mode == null || mode.isEmpty) BufferedMode else mode
  }

  /**
   * Task side of the file protocol: write one task's rows as data files.
   *
   * Returns the manifest entries as JSON. The driver commits them; until it does,
   * the files are invisible to readers, so a retried task attempt can only leak
   * orphan files (reaped by `remove_orphan_files`), never duplicate rows.
   */
  def stageDataFiles(tableUri: String, chunks: Seq[Array[Byte]]): String = {
    val parts = chunks.map(chunk => stageChunkAsFiles(tableUri, chunk))
    mergeEntryArrays(parts)
  }

  private def stageChunkAsFiles(tableUri: String, ipc: Array[Byte]): String = {
    val reader =
      new ArrowStreamReader(Channels.newChannel(new ByteArrayInputStream(ipc)), allocator)
    try {
      val root = reader.getVectorSchemaRoot
      val parts = scala.collection.mutable.ArrayBuffer.empty[String]
      while (reader.loadNextBatch()) {
        val array = ArrowArray.allocateNew(allocator)
        val schema = ArrowSchema.allocateNew(allocator)
        try {
          Data.exportVectorSchemaRoot(allocator, root, null, array, schema)
          val json = bridge.stageDataFiles(
            tableUri,
            array.memoryAddress(),
            schema.memoryAddress()
          )
          if (json == null) {
            throw new RuntimeException(s"BenoStreamDB stageDataFiles failed for $tableUri")
          }
          parts += json
        } finally {
          array.close()
          schema.close()
        }
      }
      mergeEntryArrays(parts.toSeq)
    } finally {
      reader.close()
    }
  }

  /** Driver side of the file protocol: commit every accepted task's entries once. */
  def commitStagedEntries(tableUri: String, token: String, entries: Seq[String]): Unit = {
    val json = mergeEntryArrays(entries)
    if (!bridge.commitStagedEntries(tableUri, token, json)) {
      throw new RuntimeException(s"BenoStreamDB commitStagedEntries failed for $tableUri")
    }
  }

  /** Concatenate JSON arrays of manifest entries into one array. */
  private def mergeEntryArrays(parts: Seq[String]): String = {
    val bodies = parts.flatMap { part =>
      val trimmed = if (part == null) "" else part.trim
      if (trimmed.length <= 2) Seq.empty
      else if (trimmed.startsWith("[") && trimmed.endsWith("]"))
        Seq(trimmed.substring(1, trimmed.length - 1))
      else Seq(trimmed)
    }
    bodies.mkString("[", ",", "]")
  }

  /** Encode rows as an Arrow IPC stream (self-describing: carries its schema). */
  def encode(rows: Seq[InternalRow], schema: StructType): Array[Byte] = {
    val root = BenoStreamArrowWriter.toVectorSchemaRoot(rows, schema, allocator)
    val out = new ByteArrayOutputStream()
    val writer = new ArrowStreamWriter(root, null, Channels.newChannel(out))
    try {
      writer.start()
      writer.writeBatch()
      writer.end()
    } finally {
      writer.close()
      root.close()
    }
    out.toByteArray
  }

  /** Decode one IPC chunk and buffer it natively under `token` (no commit). */
  def stage(tableUri: String, token: String, ipc: Array[Byte]): Unit = {
    val reader =
      new ArrowStreamReader(Channels.newChannel(new ByteArrayInputStream(ipc)), allocator)
    try {
      val root = reader.getVectorSchemaRoot
      while (reader.loadNextBatch()) {
        val array = ArrowArray.allocateNew(allocator)
        val schema = ArrowSchema.allocateNew(allocator)
        try {
          Data.exportVectorSchemaRoot(allocator, root, null, array, schema)
          if (!bridge.stageBatch(
                tableUri,
                token,
                array.memoryAddress(),
                schema.memoryAddress()
              )) {
            throw new RuntimeException(s"BenoStreamDB stageBatch failed for $tableUri")
          }
        } finally {
          array.close()
          schema.close()
        }
      }
    } finally {
      reader.close()
    }
  }

  /**
   * Write and commit everything staged for `token` as a single snapshot.
   *
   * Idempotent: a token whose buffer was already drained commits nothing.
   */
  def commitStaged(tableUri: String, token: String, gpuDevice: String): Unit = {
    com.benostreamdb.spark.gpu.GpuContextResolver.bindTaskGpuContext(gpuDevice)
    if (!bridge.commitStaged(tableUri, token)) {
      throw new RuntimeException(s"BenoStreamDB commitStaged failed for $tableUri")
    }
  }

  /** Drop everything staged for `token` (aborted or failed write). */
  def discard(token: String): Unit = {
    if (!bridge.discardStaged(token)) {
      throw new RuntimeException(s"BenoStreamDB discardStaged failed for token $token")
    }
  }

  /**
   * Apply delete predicates in as few commits as possible.
   *
   * A MERGE renders up to one `pk IN (...)` predicate per key chunk; the engine
   * OR-combines them into one predicate (and one delete commit) where the
   * resulting AST stays within its size budget.
   */
  def deleteRows(tableUri: String, predicates: Seq[String]): Unit = {
    if (predicates.isEmpty) return
    val json = predicates.map(p => "\"" + escapeJson(p) + "\"").mkString("[", ",", "]")
    if (!bridge.deleteRowsBatch(tableUri, json)) {
      throw new RuntimeException(s"BenoStreamDB deleteRowsBatch failed for $tableUri")
    }
  }

  /** Minimal JSON string escaping: only the characters JSON requires. */
  private def escapeJson(value: String): String = {
    val sb = new StringBuilder(value.length + 8)
    value.foreach {
      case '"' => sb.append("\\\"")
      case '\\' => sb.append("\\\\")
      case '\n' => sb.append("\\n")
      case '\r' => sb.append("\\r")
      case '\t' => sb.append("\\t")
      case c if c < ' ' => sb.append(f"\\u${c.toInt}%04x")
      case c => sb.append(c)
    }
    sb.toString
  }
}
