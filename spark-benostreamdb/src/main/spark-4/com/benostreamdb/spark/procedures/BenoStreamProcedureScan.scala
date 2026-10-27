package com.benostreamdb.spark.procedures

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.LocalScan
import org.apache.spark.sql.types.StructType

/**
 * A scan over an already-materialised result set.
 *
 * Spark 4's `BoundProcedure.call` returns `Iterator[Scan]` rather than rows, so a
 * procedure that produces output (like `show_indexes`) has to hand back a scan.
 * The rows come from the JNI call performed in `call`, so there is no planning
 * and no I/O.
 *
 * It is a `LocalScan` because Spark executes procedure output in the driver and
 * rejects anything else with "Only local scans are temporarily supported as
 * procedure output".
 */
class BenoStreamProcedureScan(schema: StructType, materialisedRows: Seq[InternalRow])
    extends LocalScan {

  override def readSchema(): StructType = schema

  override def rows(): Array[InternalRow] = materialisedRows.toArray
}
