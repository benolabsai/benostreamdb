package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan

class AddIndexProcedure extends UnboundProcedure with BoundProcedure {
  
  override def name(): String = "add_index"
  override def description(): String = "Adds a BenoStreamDB index to a column"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build(),
    ProcedureParameter.in("column", DataTypes.StringType).build(),
    // hnsw | hnsw_pq | hnsw_tq4 | hnsw_tq8 | bm25 | bloom | bitmap |
    // composite_bitmap | csr_graph | json_path (see the SQL manual).
    ProcedureParameter.in("algorithm", DataTypes.StringType).defaultValue("vector").build()
  )
      
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
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
    
    java.util.Collections.emptyIterator()
  }
}
