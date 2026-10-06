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
    new BenoStreamDeltaWriteBuilder(table.tableUri, table.structSchema, pkColumns, gpuDevice)
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

class BenoStreamDeltaWriteBuilder(uri: String, dataSchema: StructType, pkColumns: Seq[String], gpuDevice: String)
    extends DeltaWriteBuilder {
  override def build(): DeltaWrite = new BenoStreamDeltaWrite(uri, dataSchema, pkColumns, gpuDevice)
}

class BenoStreamDeltaWrite(uri: String, dataSchema: StructType, pkColumns: Seq[String], gpuDevice: String)
    extends DeltaWrite {
  override def toBatch(): DeltaBatchWrite = new BenoStreamDeltaBatchWrite(uri, dataSchema, pkColumns, gpuDevice)
}

class BenoStreamDeltaBatchWrite(uri: String, dataSchema: StructType, pkColumns: Seq[String], gpuDevice: String)
    extends DeltaBatchWrite {
  override def createBatchWriterFactory(info: PhysicalWriteInfo): DeltaWriterFactory =
    new BenoStreamDeltaWriterFactory(uri, dataSchema, pkColumns, gpuDevice)
  override def commit(messages: Array[WriterCommitMessage]): Unit = ()
  override def abort(messages: Array[WriterCommitMessage]): Unit = ()
}

class BenoStreamDeltaWriterFactory(uri: String, dataSchema: StructType, pkColumns: Seq[String], gpuDevice: String)
    extends DeltaWriterFactory {
  override def createWriter(partitionId: Int, taskId: Long): DeltaWriter[InternalRow] =
    new BenoStreamDeltaWriter(uri, dataSchema, pkColumns, gpuDevice)
}

class BenoStreamDeltaWriter(uri: String, dataSchema: StructType, pkColumns: Seq[String], gpuDevice: String)
    extends DeltaWriter[InternalRow] {

  private val bridge = BenoStreamJNIBridge.getInstance()
  private val pkTypes = pkColumns.map(c => dataSchema(c).dataType)
  private val deletePredicates = scala.collection.mutable.ArrayBuffer.empty[String]
  private val inserts = scala.collection.mutable.ArrayBuffer.empty[InternalRow]

  private def quote(s: String): String = "\"" + s.replace("\"", "\"\"") + "\""
  private def literal(v: Any, isString: Boolean): String =
    if (v == null) "NULL" else if (isString) "'" + v.toString.replace("'", "''") + "'" else v.toString

  // metadata/id row = PK columns in `pkColumns` order
  private def recordDelete(row: InternalRow): Unit = {
    if (row == null || row.numFields < pkColumns.length) return
    val parts = (0 until pkColumns.length).map { i =>
      if (row.isNullAt(i)) s"${quote(pkColumns(i))} IS NULL"
      else s"${quote(pkColumns(i))} = ${literal(row.get(i, pkTypes(i)), BenoStreamSql.isStringType(pkTypes(i)))}"
    }
    deletePredicates += parts.mkString(" AND ")
  }

  override def delete(metadata: InternalRow, id: InternalRow): Unit =
    recordDelete(if (id != null && id.numFields >= pkColumns.length) id else metadata)

  override def update(metadata: InternalRow, id: InternalRow, row: InternalRow): Unit = {
    recordDelete(if (id != null && id.numFields >= pkColumns.length) id else metadata)
    inserts += row.copy()
  }

  override def insert(row: InternalRow): Unit = inserts += row.copy()

  override def commit(): WriterCommitMessage = {
    if (deletePredicates.nonEmpty) {
      val predicate = deletePredicates.mkString(" OR ")
      com.benostreamdb.spark.gpu.GpuContextResolver.bindTaskGpuContext(gpuDevice)
      if (!bridge.deleteRows(uri, predicate)) throw new RuntimeException(s"deleteRows failed for $uri")
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

  override def abort(): Unit = { deletePredicates.clear(); inserts.clear() }
  override def close(): Unit = { deletePredicates.clear(); inserts.clear() }
}
