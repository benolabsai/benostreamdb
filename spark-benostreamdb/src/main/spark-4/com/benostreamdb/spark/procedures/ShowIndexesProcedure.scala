package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan

class ShowIndexesProcedure extends UnboundProcedure with BoundProcedure {
  
  override def name(): String = "show_indexes"
  override def description(): String = "Lists all indexes for a BenoStreamDB table"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build()
  )
      
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
    val table = inputArgs.getString(0)
    
    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val tableIdentifier = table.split("\\.").last
    jniBridge.listIndexes(tableIdentifier)
    
    java.util.Collections.emptyIterator()
  }
}
