package com.benostreamdb.spark.procedures

import org.apache.spark.sql.connector.iceberg.catalog.{Procedure, ProcedureParameter}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.unsafe.types.UTF8String

class ShowIndexesProcedure extends Procedure {
  
  override def description(): String = "Lists all indexes for a BenoStreamDB table"
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.required("table", DataTypes.StringType)
  )
  
  override def outputType(): StructType = new StructType()
    .add("table", DataTypes.StringType)
    .add("indexes", DataTypes.StringType)
      
  override def call(inputArgs: InternalRow): Array[InternalRow] = {
    val table = inputArgs.getString(0)

    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val tableIdentifier = table.split("\\.").last
    val indexesJson = jniBridge.listIndexes(tableIdentifier)
    
    val row = new GenericInternalRow(2)
    row.update(0, UTF8String.fromString(table))
    row.update(1, UTF8String.fromString(indexesJson))
    
    Array(row)
  }
}
