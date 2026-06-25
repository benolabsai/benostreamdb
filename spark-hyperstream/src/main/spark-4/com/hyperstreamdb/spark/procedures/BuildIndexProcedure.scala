package com.hyperstreamdb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan
import org.apache.spark.sql.connector.catalog.DefaultValue

class BuildIndexProcedure extends UnboundProcedure with BoundProcedure {
  
  override def name(): String = "build_index"
  override def description(): String = "Builds offline batch indexes for a HyperStreamDB table"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build(),
    ProcedureParameter.in("segment_id", DataTypes.StringType).defaultValue("null").build()
  )
      
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
    val table = inputArgs.getString(0)
    
    val jniBridge = com.hyperstreamdb.spark.jni.HyperStreamJNIBridge.getInstance()
    val gpuDevice = org.apache.spark.sql.SparkSession.active.conf.get("spark.hyperstream.gpu.device", "auto")
    jniBridge.setGpuContext(gpuDevice)
    jniBridge.buildIndex(table.split("\\.").last, "all")
    
    java.util.Collections.emptyIterator()
  }
}
