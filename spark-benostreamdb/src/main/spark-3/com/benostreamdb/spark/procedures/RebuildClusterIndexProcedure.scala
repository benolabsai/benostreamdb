package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.iceberg.catalog.{Procedure, ProcedureParameter}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.unsafe.types.UTF8String

class RebuildClusterIndexProcedure extends Procedure {
  
  override def description(): String = "Rebuilds table-level cluster index for a BenoStreamDB table"
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.required("table", DataTypes.StringType),
    ProcedureParameter.optional("column", DataTypes.StringType),
    ProcedureParameter.optional("k", DataTypes.IntegerType),
    ProcedureParameter.optional("m", DataTypes.IntegerType),
    ProcedureParameter.optional("ef_construction", DataTypes.IntegerType),
    ProcedureParameter.optional("target_shard_mb", DataTypes.LongType)
  )
  
  override def outputType(): StructType = new StructType()
    .add("table", DataTypes.StringType)
    .add("status", DataTypes.StringType)
      
  override def call(inputArgs: InternalRow): Array[InternalRow] = {
    val table = inputArgs.getString(0)
    val col = if (inputArgs.numFields > 1 && !inputArgs.isNullAt(1)) inputArgs.getString(1) else "embedding"
    val k = if (inputArgs.numFields > 2 && !inputArgs.isNullAt(2)) inputArgs.getInt(2) else 1024
    val m = if (inputArgs.numFields > 3 && !inputArgs.isNullAt(3)) inputArgs.getInt(3) else 16
    val efConstruction = if (inputArgs.numFields > 4 && !inputArgs.isNullAt(4)) inputArgs.getInt(4) else 200
    val targetMb = if (inputArgs.numFields > 5 && !inputArgs.isNullAt(5)) inputArgs.getLong(5) else 512L
    val shardBytes = targetMb * 1024L * 1024L
    
    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val gpuDevice = org.apache.spark.sql.SparkSession.active.conf.get("spark.benostream.gpu.device", "auto")
    jniBridge.setGpuContext(gpuDevice)
    val ok = jniBridge.rebuildClusterIndex(BenoStreamTableResolver.resolveFromSession(table), col, k, m, efConstruction, shardBytes)
    
    val row = new GenericInternalRow(2)
    row.update(0, UTF8String.fromString(table))
    row.update(1, UTF8String.fromString(if (ok) "success" else "failed"))
    
    Array(row)
  }
}
