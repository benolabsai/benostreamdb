package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.iceberg.catalog.{Procedure, ProcedureParameter}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.unsafe.types.UTF8String

class RebuildIndexProcedure extends Procedure {
  
  override def description(): String = "Rebuilds an existing index for a table"
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.required("table", DataTypes.StringType),
    ProcedureParameter.optional("column", DataTypes.StringType)
  )
  
  override def outputType(): StructType = new StructType()
    .add("table", DataTypes.StringType)
    .add("column", DataTypes.StringType)
    .add("status", DataTypes.StringType)
      
  override def call(inputArgs: InternalRow): Array[InternalRow] = {
    val table = inputArgs.getString(0)
    val col = if (inputArgs.numFields > 1 && !inputArgs.isNullAt(1)) inputArgs.getString(1) else "all"

    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val gpuDevice = org.apache.spark.sql.SparkSession.active.conf.get("spark.benostream.gpu.device", "auto")
    jniBridge.setGpuContext(gpuDevice)
    val tableIdentifier = table.split("\\.").last
    jniBridge.buildIndex(tableIdentifier, col)
    
    val row = new GenericInternalRow(3)
    row.update(0, UTF8String.fromString(table))
    row.update(1, UTF8String.fromString(col))
    row.update(2, UTF8String.fromString("rebuilt"))
    
    Array(row)
  }
}
