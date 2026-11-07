package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.iceberg.catalog.{Procedure, ProcedureParameter}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.unsafe.types.UTF8String

class WarmupProcedure extends Procedure {
  
  override def description(): String = "Pre-warms table-level cluster index caches for a BenoStreamDB table"
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.required("table", DataTypes.StringType),
    ProcedureParameter.optional("n_probes", DataTypes.IntegerType),
    ProcedureParameter.optional("max_memory_gb", DataTypes.DoubleType)
  )
  
  override def outputType(): StructType = new StructType()
    .add("table", DataTypes.StringType)
    .add("status", DataTypes.StringType)
      
  override def call(inputArgs: InternalRow): Array[InternalRow] = {
    val table = inputArgs.getString(0)
    val probes = if (inputArgs.numFields > 1 && !inputArgs.isNullAt(1)) inputArgs.getInt(1) else 100
    val maxMem = if (inputArgs.numFields > 2 && !inputArgs.isNullAt(2)) inputArgs.getDouble(2) else 8.0
    
    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val ok = jniBridge.warmupTable(BenoStreamTableResolver.resolveFromSession(table), probes, maxMem)
    
    val row = new GenericInternalRow(2)
    row.update(0, UTF8String.fromString(table))
    row.update(1, UTF8String.fromString(if (ok) "success" else "failed"))
    
    Array(row)
  }
}

