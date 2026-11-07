package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan

class WarmupProcedure extends UnboundProcedure with BoundProcedure {
  
  override def name(): String = "warmup"
  override def description(): String = "Pre-warms table-level cluster index caches for a BenoStreamDB table"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build(),
    ProcedureParameter.in("n_probes", DataTypes.IntegerType).defaultValue("100").build(),
    ProcedureParameter.in("max_memory_gb", DataTypes.DoubleType).defaultValue("8.0").build()
  )
      
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
    val table = inputArgs.getString(0)
    val probes = if (inputArgs.numFields > 1 && !inputArgs.isNullAt(1)) inputArgs.getInt(1) else 100
    val maxMem = if (inputArgs.numFields > 2 && !inputArgs.isNullAt(2)) inputArgs.getDouble(2) else 8.0
    
    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val tableIdentifier = BenoStreamTableResolver.resolveFromSession(table)
    jniBridge.warmupTable(tableIdentifier, probes, maxMem)
    
    java.util.Collections.emptyIterator()
  }
}

