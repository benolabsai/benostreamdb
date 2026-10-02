package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan

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
  
  // Spark 4's `BoundProcedure.call` returns an iterator of `Scan`s. Like the
  // other BenoStreamDB procedures (add_index, show_indexes, ...) this is a
  // side-effecting call: the JNI search runs, and no scan is produced.
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
    val table = inputArgs.getString(0)
    val query = inputArgs.getString(1)
    val seedsArray = inputArgs.getArray(2).toLongArray()
    
    val topK = if (inputArgs.isNullAt(3)) 5 else inputArgs.getInt(3)
    val hops = if (inputArgs.isNullAt(4)) 2 else inputArgs.getInt(4)
    val nDepth = if (inputArgs.isNullAt(5)) 2 else inputArgs.getInt(5)
    val kFollowups = if (inputArgs.isNullAt(6)) 3 else inputArgs.getInt(6)
    val mode = if (inputArgs.isNullAt(7)) "auto" else inputArgs.getString(7)

    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    jniBridge.runRegionalDriftSearch(
      table, query, seedsArray, topK, hops, nDepth, kFollowups, mode
    )

    java.util.Collections.emptyIterator()
  }
}
