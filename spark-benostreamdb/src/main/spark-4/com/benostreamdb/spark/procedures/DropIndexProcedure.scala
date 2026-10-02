package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan

class DropIndexProcedure extends UnboundProcedure with BoundProcedure {
  
  override def name(): String = "drop_index"
  override def description(): String = "Drops a BenoStreamDB index from a column"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build(),
    ProcedureParameter.in("column", DataTypes.StringType).build(),
    ProcedureParameter.in("index_type", DataTypes.StringType).defaultValue("vector").build()
  )
      
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
    val table = inputArgs.getString(0)
    val column = inputArgs.getString(1)
    val indexType = if (inputArgs.numFields > 2 && !inputArgs.isNullAt(2)) inputArgs.getString(2) else "vector"
    
    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val gpuDevice = org.apache.spark.sql.SparkSession.active.conf.get("spark.benostream.gpu.device", "auto")
    jniBridge.setGpuContext(gpuDevice)
    val tableIdentifier = table.split("\\.").last
    jniBridge.dropIndex(tableIdentifier, column, indexType)
    
    java.util.Collections.emptyIterator()
  }
}
