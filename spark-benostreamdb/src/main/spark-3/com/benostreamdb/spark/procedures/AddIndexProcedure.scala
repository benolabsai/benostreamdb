package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.iceberg.catalog.{Procedure, ProcedureParameter}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.unsafe.types.UTF8String

class AddIndexProcedure extends Procedure {
  
  override def description(): String = "Adds a BenoStreamDB index to a column"
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.required("table", DataTypes.StringType),
    ProcedureParameter.required("column", DataTypes.StringType),
    // hnsw | hnsw_pq | hnsw_tq4 | hnsw_tq8 | bm25 | bloom | bitmap |
    // composite_bitmap | csr_graph | json_path (see the SQL manual).
    ProcedureParameter.optional("algorithm", DataTypes.StringType)
  )
  
  override def outputType(): StructType = new StructType()
    .add("table", DataTypes.StringType)
    .add("column", DataTypes.StringType)
    .add("status", DataTypes.StringType)
      
  override def call(inputArgs: InternalRow): Array[InternalRow] = {
    val table = inputArgs.getString(0)
    val column = inputArgs.getString(1)
    val algorithm =
      if (inputArgs.numFields > 2 && !inputArgs.isNullAt(2)) inputArgs.getString(2) else "vector"
    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val gpuDevice = org.apache.spark.sql.SparkSession.active.conf.get("spark.benostream.gpu.device", "auto")
    jniBridge.setGpuContext(gpuDevice)
    // For standalone parsing
    val tableIdentifier = table.split("\\.").last
    jniBridge.addIndex(tableIdentifier, column, algorithm)
    
    val row = new GenericInternalRow(3)
    row.update(0, UTF8String.fromString(table))
    row.update(1, UTF8String.fromString(column))
    row.update(2, UTF8String.fromString("success"))
    
    Array(row)
  }
}
