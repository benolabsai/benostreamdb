package com.benostreamdb.spark

import org.apache.spark.sql.connector.catalog.{Table, TableProvider}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.iceberg.spark.source.IcebergSource
import java.util.Map

import org.apache.spark.sql.sources.DataSourceRegister

/**
 * The entry point for the BenoStreamDB Spark Connector.
 * This wraps the official Iceberg source but injects our custom Table implementation
 * to intercept MERGE INTO and other operations.
 */
class DefaultSource extends TableProvider with DataSourceRegister {

  override def shortName(): String = "benostream"

  // We delegate basic metadata/table resolution to the official IcebergSource
  private lazy val icebergSource = new IcebergSource()

  override def inferSchema(options: CaseInsensitiveStringMap): StructType = {
    // Delegate to Iceberg to read the schema from the manifest
    icebergSource.inferSchema(options)
  }

  override def getTable(
      schema: StructType,
      partitioning: Array[Transform],
      properties: Map[String, String]
  ): Table = {
    // Get the underlying Iceberg Table
    val icebergTable = icebergSource.getTable(schema, partitioning, properties)
    
    val gpuDevice = properties.getOrDefault("benostream.gpu_device", "auto")
    
    // Wrap it in our BenoStreamTable to intercept operations
    new BenoStreamTable(icebergTable, schema, properties, gpuDevice)
  }

  override def supportsExternalMetadata(): Boolean = {
    true
  }
}
