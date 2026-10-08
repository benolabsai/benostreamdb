package com.benostreamdb.spark

import com.benostreamdb.spark.jni.BenoStreamJNIBridge
import org.apache.arrow.c.{ArrowArray, ArrowSchema, Data}
import org.apache.arrow.memory.RootAllocator
import org.apache.spark.sql.connector.catalog.{SupportsRead, Table, TableCapability}
import org.apache.spark.sql.connector.read.{Scan, ScanBuilder}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.util.CaseInsensitiveStringMap

import java.util.{Set => JSet}
import scala.collection.JavaConverters._

/**
 * Pass-through reader: `spark.read.format("benostream")
 *   .option("path", "<table-uri>").option("query", "<engine SQL>").load()`.
 *
 * The SQL is executed by the engine's DataFusion session, so **every** engine
 * function is reachable from Spark — including the vector aggregates and the
 * graph UDAFs, which have no DSv2 scalar-function equivalent. The result
 * schema is derived from the engine's own result.
 */
class BenoStreamQueryTable(uri: String, sql: String, resultSchema: StructType)
    extends Table
    with SupportsRead {

  override def name(): String = "benostream-query"

  override def schema(): StructType = resultSchema

  override def capabilities(): JSet[TableCapability] =
    Set(TableCapability.BATCH_READ).asJava

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder =
    new ScanBuilder {
      override def build(): Scan = new BenoStreamScan(uri, resultSchema, sql)
    }
}

object BenoStreamQuerySchema {

  /**
   * Derive the result schema by running the query once and reading the first
   * batch's Arrow schema (the engine returns the schema even for zero rows).
   */
  def resolve(uri: String, sql: String): StructType = {
    val bridge = BenoStreamJNIBridge.getInstance()
    val allocator = new RootAllocator()
    val handle = bridge.openQuery(uri, sql)
    if (handle == 0L) return new StructType()

    val array = ArrowArray.allocateNew(allocator)
    val schema = ArrowSchema.allocateNew(allocator)
    try {
      if (bridge.readQueryBatch(handle, array.memoryAddress(), schema.memoryAddress()) == 0L) {
        new StructType()
      } else {
        val root = Data.importVectorSchemaRoot(allocator, array, schema, null)
        try {
          StructType(
            root.getSchema.getFields.asScala
              .map(f => StructField(f.getName, BenoStreamArrowUtils.arrowFieldToSparkType(f), nullable = true))
              .toArray
          )
        } finally root.close()
      }
    } finally {
      array.close()
      schema.close()
      bridge.closeQuery(handle)
      allocator.close()
    }
  }
}
