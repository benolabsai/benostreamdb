package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.iceberg.catalog.{Procedure, ProcedureParameter}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow

class RegionalDriftProcedure extends Procedure {
  
  override def description(): String = "Executes the regional DRIFT graph search algorithm"
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.required("table", DataTypes.StringType),
    ProcedureParameter.required("query", DataTypes.StringType),
    ProcedureParameter.required("seeds", DataTypes.createArrayType(DataTypes.LongType)),
    ProcedureParameter.optional("top_k", DataTypes.IntegerType),
    ProcedureParameter.optional("hops", DataTypes.IntegerType),
    ProcedureParameter.optional("n_depth", DataTypes.IntegerType),
    ProcedureParameter.optional("k_followups", DataTypes.IntegerType),
    ProcedureParameter.optional("mode", DataTypes.StringType)
  )
  
  override def outputType(): StructType = new StructType()
    .add("node_id", DataTypes.LongType)
      
  override def call(inputArgs: InternalRow): Array[InternalRow] = {
    val table = inputArgs.getString(0)
    val query = inputArgs.getString(1)
    val seedsArray = inputArgs.getArray(2).toLongArray()
    
    val topK = if (inputArgs.isNullAt(3)) 5 else inputArgs.getInt(3)
    val hops = if (inputArgs.isNullAt(4)) 2 else inputArgs.getInt(4)
    val nDepth = if (inputArgs.isNullAt(5)) 2 else inputArgs.getInt(5)
    val kFollowups = if (inputArgs.isNullAt(6)) 3 else inputArgs.getInt(6)
    val mode = if (inputArgs.isNullAt(7)) "auto" else inputArgs.getString(7)

    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    
    val nodes = jniBridge.runRegionalDriftSearch(
      table, query, seedsArray, topK, hops, nDepth, kFollowups, mode
    )
    
    nodes.map { n =>
      val row = new GenericInternalRow(1)
      row.update(0, n)
      row
    }.toArray
  }
}
