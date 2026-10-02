package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan

class RebuildIndexProcedure extends UnboundProcedure with BoundProcedure {
  
  override def name(): String = "rebuild_index"
  override def description(): String = "Rebuilds an existing index for a table"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build(),
    ProcedureParameter.in("column", DataTypes.StringType).defaultValue("all").build()
  )
      
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
    val table = inputArgs.getString(0)
    val col = if (inputArgs.numFields > 1 && !inputArgs.isNullAt(1)) inputArgs.getString(1) else "all"
    
    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val gpuDevice = org.apache.spark.sql.SparkSession.active.conf.get("spark.benostream.gpu.device", "auto")
    jniBridge.setGpuContext(gpuDevice)
    val tableIdentifier = table.split("\\.").last
    jniBridge.buildIndex(tableIdentifier, col)
    
    java.util.Collections.emptyIterator()
  }
}
