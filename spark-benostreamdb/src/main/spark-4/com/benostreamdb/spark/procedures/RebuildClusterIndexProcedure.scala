package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan

class RebuildClusterIndexProcedure extends UnboundProcedure with BoundProcedure {
  
  override def name(): String = "rebuild_cluster_index"
  override def description(): String = "Rebuilds table-level cluster index into rolling Puffin shards"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build(),
    ProcedureParameter.in("column", DataTypes.StringType).defaultValue("'embedding'").build(),
    ProcedureParameter.in("k", DataTypes.IntegerType).defaultValue("1024").build(),
    ProcedureParameter.in("m", DataTypes.IntegerType).defaultValue("16").build(),
    ProcedureParameter.in("ef_construction", DataTypes.IntegerType).defaultValue("200").build(),
    ProcedureParameter.in("target_shard_mb", DataTypes.LongType).defaultValue("512").build()
  )
      
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
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
    val tableIdentifier = BenoStreamTableResolver.resolveFromSession(table)
    jniBridge.rebuildClusterIndex(tableIdentifier, col, k, m, efConstruction, shardBytes)
    
    java.util.Collections.emptyIterator()
  }
}

