package com.benostreamdb.spark

import com.benostreamdb.spark.jni.BenoStreamJNIBridge
import org.apache.spark.sql.connector.catalog.{Table, TableProvider}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

import java.util.{Map => JMap}

/**
 * Entry point for `spark.read/write.format("benostream")`. Native: resolves the
 * table via the engine's JNI metadata (no Iceberg `IcebergSource`).
 */
class DefaultSource extends TableProvider with DataSourceRegister {

  override def shortName(): String = "benostream"

  private def uriOf(m: JMap[String, String]): String = {
    val u = Option(m.get("path")).orElse(Option(m.get("location"))).getOrElse("")
    u.stripSuffix("/")
  }

  private def queryOf(m: JMap[String, String]): Option[String] =
    Option(m.get("query")).map(_.trim).filter(_.nonEmpty)

  override def inferSchema(options: CaseInsensitiveStringMap): StructType = {
    val m = options.asCaseSensitiveMap()
    queryOf(m) match {
      case Some(sql) => BenoStreamQuerySchema.resolve(uriOf(m), sql)
      case None =>
        val json = BenoStreamJNIBridge.getInstance().getTableSchema(uriOf(m))
        if (json == null || json.isEmpty || json == "[]") new StructType()
        else BenoStreamArrowUtils.schemaJsonToStructType(json)
    }
  }

  override def getTable(
      schema: StructType,
      partitioning: Array[Transform],
      properties: JMap[String, String]
  ): Table = {
    val uri = uriOf(properties)
    queryOf(properties) match {
      case Some(sql) =>
        val resolved =
          if (schema != null && schema.fields.nonEmpty) schema
          else BenoStreamQuerySchema.resolve(uri, sql)
        new BenoStreamQueryTable(uri, sql, resolved)
      case None =>
        val resolved =
          if (schema != null && schema.fields.nonEmpty) schema
          else {
            val json = BenoStreamJNIBridge.getInstance().getTableSchema(uri)
            if (json == null || json.isEmpty || json == "[]") new StructType()
            else BenoStreamArrowUtils.schemaJsonToStructType(json)
          }
        val gpu = Option(properties.get("benostream.gpu_device"))
          .orElse(Option(properties.get("gpu_device")))
          .getOrElse("auto")
        new BenoStreamTable(uri, resolved, properties, gpu)
    }
  }

  override def supportsExternalMetadata(): Boolean = true
}
