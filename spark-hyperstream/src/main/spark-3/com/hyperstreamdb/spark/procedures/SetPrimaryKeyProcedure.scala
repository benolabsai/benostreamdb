package com.hyperstreamdb.spark.procedures

import org.apache.spark.sql.connector.iceberg.catalog.{Procedure, ProcedureParameter}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.unsafe.types.UTF8String

class SetPrimaryKeyProcedure extends Procedure {
  
  override def description(): String = "Sets the primary key for a HyperStreamDB table"
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.required("table", DataTypes.StringType),
    ProcedureParameter.required("columns", DataTypes.StringType)
  )
  
  override def outputType(): StructType = new StructType()
    .add("table", DataTypes.StringType)
    .add("columns", DataTypes.StringType)
    .add("status", DataTypes.StringType)
      
  override def call(inputArgs: InternalRow): Array[InternalRow] = {
    val table = inputArgs.getString(0)
    val columns = inputArgs.getString(1)
    
    val jniBridge = com.hyperstreamdb.spark.jni.HyperStreamJNIBridge.getInstance()
    val gpuDevice = org.apache.spark.sql.SparkSession.active.conf.get("spark.hyperstream.gpu.device", "auto")
    jniBridge.setGpuContext(gpuDevice)
    jniBridge.setPrimaryKey(table.split("\\.").last, columns)
    
    val row = new GenericInternalRow(3)
    row.update(0, UTF8String.fromString(table))
    row.update(1, UTF8String.fromString(columns))
    row.update(2, UTF8String.fromString("success"))
    
    Array(row)
  }
}
