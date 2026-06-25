package com.hyperstreamdb.spark

import org.apache.spark.sql.connector.read.{Scan, ScanBuilder, SupportsRuntimeFiltering}
import org.apache.spark.sql.connector.expressions.NamedReference
import org.apache.spark.sql.connector.expressions.filter.Predicate
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.util.CaseInsensitiveStringMap

class HyperStreamScanBuilder(
    table: HyperStreamTable,
    options: CaseInsensitiveStringMap
) extends ScanBuilder {

  override def build(): Scan = {
    val delegateBuilder = table.delegate.asInstanceOf[org.apache.spark.sql.connector.catalog.SupportsRead].newScanBuilder(options)
    new HyperStreamScan(delegateBuilder.build(), table)
  }
}

class HyperStreamScan(val delegate: Scan, val table: HyperStreamTable) extends Scan with SupportsRuntimeFiltering {

  private var dynamicFilters: Array[Filter] = Array.empty

  override def readSchema(): org.apache.spark.sql.types.StructType = delegate.readSchema()

  override def toBatch() = delegate.toBatch()
  // TODO: we should intercept toBatch to use our filters

  override def filterAttributes(): Array[NamedReference] = {
    // We advertise all columns for runtime filtering so Spark gives us the broadcast keys
    Array.empty // TODO: Return the primary key columns here to receive DynamicFilters
  }

  override def filter(filters: Array[Filter]): Unit = {
    this.dynamicFilters = filters
    // MAGIC HAPPENS HERE:
    // Extract the primary keys from 'filters' (usually an IN filter)
    // Send them to Rust Core via JNI:
    val jniBridge = com.hyperstreamdb.spark.jni.HyperStreamJNIBridge.getInstance()
    jniBridge.setGpuContext(table.gpuDevice)
    val fileBitmaps = jniBridge.queryIndexIn(delegate.asInstanceOf[org.apache.spark.sql.connector.read.Scan].getClass.getSimpleName, "pk", "[]") // Placeholder for JSON extraction
    // Then we can configure the Iceberg delegateBuilder with these exact file/bitmap constraints!
  }
}
