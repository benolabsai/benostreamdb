package com.benostreamdb.spark.procedures

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.{Batch, InputPartition, PartitionReader, PartitionReaderFactory, Scan}
import org.apache.spark.sql.types.StructType

/** The single partition used by procedure result scans. */
case object BenoStreamProcedurePartition extends InputPartition

/**
 * A `Scan` that yields an already-materialised result set.
 *
 * Spark 4's `BoundProcedure.call` returns `Iterator[Scan]` rather than rows, so
 * a procedure that produces output (like `show_indexes`) has to hand back a
 * scan. The data comes from the JNI call performed in `call`, so this scan does
 * no planning and no I/O: one partition, one reader over the buffered rows.
 */
class BenoStreamProcedureScan(schema: StructType, rows: Seq[InternalRow])
    extends Scan
    with Batch
    with PartitionReaderFactory {

  override def readSchema(): StructType = schema

  override def toBatch(): Batch = this

  override def planInputPartitions(): Array[InputPartition] = Array(BenoStreamProcedurePartition)

  override def createReaderFactory(): PartitionReaderFactory = this

  override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
    new PartitionReader[InternalRow] {
      private val iterator: Iterator[InternalRow] = rows.iterator

      override def next(): Boolean = iterator.hasNext

      override def get(): InternalRow = iterator.next()

      override def close(): Unit = ()
    }
}
