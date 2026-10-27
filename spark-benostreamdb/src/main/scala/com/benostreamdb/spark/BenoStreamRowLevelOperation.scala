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
 * Row-level operations (DELETE/UPDATE/MERGE) implemented over the engine's real
 * primitives: predicate deletes (`deleteRows`) and appends (`appendBatch`).
 *
 * Spark's merge-on-read needs a row identifier; the engine has no position
 * deletes, so we use the PRIMARY KEY columns as both `rowId` and the required
 * metadata. The PK is a normal data column, so it resolves on the target
 * relation and the scan already emits it. A delete/update therefore becomes a
 * `deleteRows(pk = ...)`; inserted/updated-new rows are appended.
 */
class BenoStreamRowLevelOperationBuilder(table: BenoStreamTable, info: RowLevelOperationInfo, gpuDevice: String)
    extends RowLevelOperationBuilder {
  override def build(): RowLevelOperation = new BenoStreamRowLevelOperation(table, info, gpuDevice)
}

class BenoStreamRowLevelOperation(table: BenoStreamTable, info: RowLevelOperationInfo, gpuDevice: String)
    extends SupportsDelta {

  private val pkColumns: Seq[String] = BenoStreamRowLevelOperation.primaryKey(table)

  override def command(): RowLevelOperation.Command = info.command()
  override def description(): String = s"BenoStreamDB native ${info.command()} on ${table.tableUri}"
  override def rowId(): Array[NamedReference] = pkColumns.map(Expressions.column).toArray
  override def requiredMetadataAttributes(): Array[NamedReference] = pkColumns.map(Expressions.column).toArray
  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder = new BenoStreamScanBuilder(table)
  override def newWriteBuilder(info: LogicalWriteInfo): DeltaWriteBuilder =
    new BenoStreamDeltaWriteBuilder(
      table.tableUri,
      table.structSchema,
      pkColumns,
      gpuDevice,
      BenoStreamDeltaWriter.resolveFlushSize(info, table.properties))
}

object BenoStreamRowLevelOperation {
  def primaryKey(table: BenoStreamTable): Seq[String] = {
    val declared = Option(table.properties.get("primary_key"))
      .map(_.split(",").map(_.trim).filter(_.nonEmpty).toSeq)
      .getOrElse(Seq.empty)
    if (declared.nonEmpty) declared
    else if (table.structSchema.nonEmpty) Seq(table.structSchema.fields.head.name)
    else Seq.empty
  }
}

class BenoStreamDeltaWriteBuilder(
    uri: String,
    dataSchema: StructType,
    pkColumns: Seq[String],
    gpuDevice: String,
    flushSize: Int = BenoStreamDeltaWriter.DefaultFlushSize)
    extends DeltaWriteBuilder {
  override def build(): DeltaWrite = new BenoStreamDeltaWrite(uri, dataSchema, pkColumns, gpuDevice, flushSize)
}

class BenoStreamDeltaWrite(
    uri: String,
    dataSchema: StructType,
    pkColumns: Seq[String],
    gpuDevice: String,
    flushSize: Int)
    extends DeltaWrite {
  override def toBatch(): DeltaBatchWrite =
    new BenoStreamDeltaBatchWrite(uri, dataSchema, pkColumns, gpuDevice, flushSize)
}

class BenoStreamDeltaBatchWrite(
    uri: String,
    dataSchema: StructType,
    pkColumns: Seq[String],
    gpuDevice: String,
    flushSize: Int)
    extends DeltaBatchWrite {
  override def createBatchWriterFactory(info: PhysicalWriteInfo): DeltaWriterFactory =
    new BenoStreamDeltaWriterFactory(uri, dataSchema, pkColumns, gpuDevice, flushSize)
  override def commit(messages: Array[WriterCommitMessage]): Unit = ()
  override def abort(messages: Array[WriterCommitMessage]): Unit = ()
}

class BenoStreamDeltaWriterFactory(
    uri: String,
    dataSchema: StructType,
    pkColumns: Seq[String],
    gpuDevice: String,
    flushSize: Int)
    extends DeltaWriterFactory {
  override def createWriter(partitionId: Int, taskId: Long): DeltaWriter[InternalRow] =
    new BenoStreamDeltaWriter(uri, dataSchema, pkColumns, gpuDevice, flushSize)
}

object BenoStreamDeltaWriter {
  /** Rows/keys buffered before an intermediate delete or append is issued. */
  val DefaultFlushSize: Int = 10000

  /** Keys per `pk IN (...)` predicate for a single-column key. */
  val MaxKeysPerInList: Int = 1000

  /** Predicates per `(a = .. AND b = ..) OR ...` chunk for compound keys. */
  val MaxCompoundKeysPerChunk: Int = 500

  val BatchSizeKey: String = "write.batchSize"
  val BatchSizeConf: String = "spark.benostream.write.batchSize"

  /** Resolve the flush size: write option → table property → Spark conf → default. */
  def resolveFlushSize(info: LogicalWriteInfo, properties: java.util.Map[String, String]): Int = {
    def parse(raw: String): Option[Int] = scala.util.Try(raw.trim.toInt).toOption.filter(_ > 0)
    Option(info.options().get(BatchSizeKey))
      .flatMap(parse)
      .orElse(Option(properties.get(BatchSizeKey)).flatMap(parse))
      .orElse(
        org.apache.spark.sql.SparkSession.getActiveSession
          .flatMap(s => parse(s.conf.get(BatchSizeConf, ""))))
      .getOrElse(DefaultFlushSize)
  }
}

/**
 * Buffers row-level operations and applies them in bounded chunks.
 *
 * Deletes accumulate *rendered primary-key literals*, not one giant SQL string:
 * a 10k-row DELETE used to build a multi-megabyte `pk = .. OR pk = ..` predicate
 * that either hit the engine's AST recursion limit or exhausted memory. Keys are
 * now flushed as (a) `pk IN (lit, ..)` batches of
 * [[BenoStreamDeltaWriter.MaxKeysPerInList]] for single-column keys, or
 * (b) `(a = .. AND b = ..) OR ...` chunks of
 * [[BenoStreamDeltaWriter.MaxCompoundKeysPerChunk]] for compound keys. Rows whose
 * key contains a NULL cannot join an `IN` list and get their own
 * `pk IS NULL`-style predicate.
 *
 * Inserts are flushed through one `appendBatch` call per `flushSize` rows so the
 * task never holds an unbounded `ArrayBuffer[InternalRow]`.
 */
class BenoStreamDeltaWriter(
    uri: String,
    dataSchema: StructType,
    pkColumns: Seq[String],
    gpuDevice: String,
    flushSize: Int = BenoStreamDeltaWriter.DefaultFlushSize)
    extends DeltaWriter[InternalRow] {

  import BenoStreamDeltaWriter._

  private val bridge = BenoStreamJNIBridge.getInstance()
  private val pkTypes = pkColumns.map(c => dataSchema(c).dataType)

  /** Rendered PK literals per pending delete, in `pkColumns` order. */
  private val deleteKeys = scala.collection.mutable.ArrayBuffer.empty[Seq[String]]

  /** Pending deletes whose key has a NULL component (cannot go in an `IN` list). */
  private val nullKeyPredicates = scala.collection.mutable.ArrayBuffer.empty[String]

  private val inserts = scala.collection.mutable.ArrayBuffer.empty[InternalRow]

  private val effectiveFlushSize: Int = if (flushSize > 0) flushSize else DefaultFlushSize
  private val keysPerChunk: Int =
    if (pkColumns.length <= 1) MaxKeysPerInList else MaxCompoundKeysPerChunk

  private def quote(s: String): String = "\"" + s.replace("\"", "\"\"") + "\""
  private def literal(v: Any, isString: Boolean): String =
    if (v == null) "NULL" else if (isString) "'" + v.toString.replace("'", "''") + "'" else v.toString

  // metadata/id row = PK columns in `pkColumns` order
  private def recordDelete(row: InternalRow): Unit = {
    // No primary key means no predicate we can build; a table without a key
    // cannot service UPDATE/DELETE through the predicate-delete primitive, and
    // rendering an empty key list would index `pkColumns.head` on an empty Seq.
    if (pkColumns.isEmpty || row == null || row.numFields < pkColumns.length) return
    val keys = (0 until pkColumns.length).map { i =>
      if (row.isNullAt(i)) null
      else literal(row.get(i, pkTypes(i)), BenoStreamSql.isStringType(pkTypes(i)))
    }
    if (keys.contains(null)) {
      nullKeyPredicates += pkColumns.indices
        .map(i => if (keys(i) == null) s"${quote(pkColumns(i))} IS NULL" else s"${quote(pkColumns(i))} = ${keys(i)}")
        .mkString(" AND ")
    } else {
      deleteKeys += keys
    }
    if (deleteKeys.length + nullKeyPredicates.length >= effectiveFlushSize) flushDeletes()
  }

  /** Render the buffered keys as chunked predicates and apply them. */
  private def flushDeletes(): Unit = {
    if (nullKeyPredicates.isEmpty && deleteKeys.isEmpty) return

    val predicates = scala.collection.mutable.ArrayBuffer.empty[String]
    predicates ++= nullKeyPredicates
    deleteKeys.iterator
      .grouped(keysPerChunk)
      .foreach(chunk => predicates += renderPredicate(chunk.map(_.toSeq).toSeq))
    nullKeyPredicates.clear()
    deleteKeys.clear()

    com.benostreamdb.spark.gpu.GpuContextResolver.bindTaskGpuContext(gpuDevice)
    predicates.foreach { predicate =>
      if (!bridge.deleteRows(uri, predicate)) throw new RuntimeException(s"deleteRows failed for $uri")
    }
  }

  private def renderPredicate(chunk: Seq[Seq[String]]): String =
    if (pkColumns.length <= 1) {
      s"${quote(pkColumns.head)} IN (${chunk.map(_.head).mkString(", ")})"
    } else {
      chunk
        .map(keys => pkColumns.zip(keys).map { case (column, value) => s"${quote(column)} = $value" }.mkString("(", " AND ", ")"))
        .mkString(" OR ")
    }

  /** Append the buffered rows as one Arrow batch, then release them. */
  private def flushInserts(): Unit = {
    if (inserts.isEmpty) return
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
    inserts.clear()
  }

  override def delete(metadata: InternalRow, id: InternalRow): Unit =
    recordDelete(if (id != null && id.numFields >= pkColumns.length) id else metadata)

  override def update(metadata: InternalRow, id: InternalRow, row: InternalRow): Unit = {
    recordDelete(if (id != null && id.numFields >= pkColumns.length) id else metadata)
    inserts += row.copy()
    if (inserts.length >= effectiveFlushSize) flushInserts()
  }

  override def insert(row: InternalRow): Unit = {
    inserts += row.copy()
    if (inserts.length >= effectiveFlushSize) flushInserts()
  }

  override def commit(): WriterCommitMessage = {
    // DELETE before INSERT: an UPDATE reuses the key, so appending first would
    // let the delete reap the freshly inserted row.
    flushDeletes()
    flushInserts()
    BenoStreamCommitMessage
  }

  override def abort(): Unit = {
    deleteKeys.clear()
    nullKeyPredicates.clear()
    inserts.clear()
  }

  override def close(): Unit = {
    deleteKeys.clear()
    nullKeyPredicates.clear()
    inserts.clear()
  }
}
