package com.hyperstreamdb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan

class SetPrimaryKeyProcedure extends UnboundProcedure with BoundProcedure {
  
  override def name(): String = "set_primary_key"
  override def description(): String = "Sets the primary key for a HyperStreamDB table"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build(),
    ProcedureParameter.in("columns", DataTypes.StringType).build()
  )
      
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
    val table = inputArgs.getString(0)
    val columns = inputArgs.getString(1)
    
    val jniBridge = com.hyperstreamdb.spark.jni.HyperStreamJNIBridge.getInstance()
    val gpuDevice = org.apache.spark.sql.SparkSession.active.conf.get("spark.hyperstream.gpu.device", "auto")
    jniBridge.setGpuContext(gpuDevice)
    jniBridge.setPrimaryKey(table.split("\\.").last, columns)
    
    java.util.Collections.emptyIterator()
  }
}
