package com.hyperstreamdb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan

class AddIndexProcedure extends UnboundProcedure with BoundProcedure {
  
  override def name(): String = "add_index"
  override def description(): String = "Adds a HyperStreamDB index to a column"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build(),
    ProcedureParameter.in("column", DataTypes.StringType).build()
  )
      
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
    val table = inputArgs.getString(0)
    val column = inputArgs.getString(1)
    
    val jniBridge = com.hyperstreamdb.spark.jni.HyperStreamJNIBridge.getInstance()
    val gpuDevice = org.apache.spark.sql.SparkSession.active.conf.get("spark.hyperstream.gpu.device", "auto")
    jniBridge.setGpuContext(gpuDevice)
    // For standalone parsing
    val tableIdentifier = table.split("\\.").last
    jniBridge.addIndex(tableIdentifier, column, "vector")
    
    java.util.Collections.emptyIterator()
  }
}
