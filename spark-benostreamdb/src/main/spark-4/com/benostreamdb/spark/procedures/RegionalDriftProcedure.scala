package com.benostreamdb.spark.procedures

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.connector.read.Scan
import org.apache.spark.sql.types.{DataTypes, StructType}

/**
 * `regional_drift_search(...)`: runs the DRIFT graph search and returns the
 * discovered `node_id`s.
 *
 * The JNI call already returns the result set; the procedure used to drop it on
 * the floor and return an empty iterator. It now yields a one-column scan, which
 * is what the Spark 3 procedure (and the Trino `system.regional_drift_search`
 * table function) already returned.
 */
class RegionalDriftProcedure extends UnboundProcedure with BoundProcedure {

  override def name(): String = "regional_drift_search"
  override def description(): String = "Executes the regional DRIFT graph search algorithm"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this

  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build(),
    ProcedureParameter.in("query", DataTypes.StringType).build(),
    ProcedureParameter.in("seeds", DataTypes.createArrayType(DataTypes.LongType)).build(),
    ProcedureParameter.in("top_k", DataTypes.IntegerType).defaultValue("5").build(),
    ProcedureParameter.in("hops", DataTypes.IntegerType).defaultValue("2").build(),
    ProcedureParameter.in("n_depth", DataTypes.IntegerType).defaultValue("2").build(),
    ProcedureParameter.in("k_followups", DataTypes.IntegerType).defaultValue("3").build(),
    ProcedureParameter.in("mode", DataTypes.StringType).defaultValue("auto").build()
  )

  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
    val table = inputArgs.getString(0)
    val query = inputArgs.getString(1)
    val seedsArray = inputArgs.getArray(2).toLongArray()

    val topK = if (inputArgs.isNullAt(3)) 5 else inputArgs.getInt(3)
    val hops = if (inputArgs.isNullAt(4)) 2 else inputArgs.getInt(4)
    val nDepth = if (inputArgs.isNullAt(5)) 2 else inputArgs.getInt(5)
    val kFollowups = if (inputArgs.isNullAt(6)) 3 else inputArgs.getInt(6)
    val mode = if (inputArgs.isNullAt(7)) "auto" else inputArgs.getString(7)

    val tableUri = BenoStreamTableResolver.resolveFromSession(table)

    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val nodes = jniBridge.runRegionalDriftSearch(
      tableUri, query, seedsArray, topK, hops, nDepth, kFollowups, mode
    )

    val rows = nodes.map(node => new GenericInternalRow(Array[Any](node)): InternalRow)
    java.util.Collections.singletonList(
      new BenoStreamProcedureScan(RegionalDriftProcedure.OutputSchema, rows)
    ).iterator()
  }
}

object RegionalDriftProcedure {
  val OutputSchema: StructType = new StructType().add("node_id", DataTypes.LongType)
}
