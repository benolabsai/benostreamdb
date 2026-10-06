package com.benostreamdb.spark

import com.benostreamdb.spark.jni.BenoStreamJNIBridge
import org.apache.spark.sql.catalyst.analysis.{NoSuchNamespaceException, NoSuchTableException, TableAlreadyExistsException}
import org.apache.spark.sql.connector.catalog._
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

import java.util.{Map => JMap}
import scala.collection.JavaConverters._

/**
 * Native BenoStreamDB catalog. Resolves/creates/drops tables entirely through
 * the engine's JNI metadata surface (`listSchemas`/`listTables`/`getTableSchema`
 * /`createTable`/`dropTable`) — no Iceberg catalog, so one Scala-2.13 artifact
 * serves Spark 4.0/4.1/4.2.
 *
 * Configure like:
 * {{{
 *   spark.sql.catalog.benostream = com.benostreamdb.spark.BenoStreamCatalog
 *   spark.sql.catalog.benostream.warehouse = /path/to/warehouse   (or s3://bucket/prefix)
 * }}}
 */
class BenoStreamCatalog extends TableCatalog with SupportsNamespaces {

  private var catalogName: String = "benostream"
  private var warehouse: String = ""
  private var gpuDevice: String = "auto"

  private def bridge = BenoStreamJNIBridge.getInstance()

  override def initialize(name: String, options: CaseInsensitiveStringMap): Unit = {
    catalogName = name
    warehouse = Option(options.get("warehouse"))
      .orElse(Option(options.get("benostream.warehouse")))
      .getOrElse("")
      .stripSuffix("/")
    gpuDevice = options.getOrDefault("gpu_device", "auto")
  }

  override def name(): String = catalogName

  private def uriFor(namespace: Array[String], table: String): String =
    (warehouse +: namespace.toSeq :+ table).mkString("/")

  private def schemaOf(uri: String, ident: Identifier): StructType = {
    val json = bridge.getTableSchema(uri)
    if (json == null || json.isEmpty || json == "[]")
      throw new NoSuchTableException(ident)
    BenoStreamArrowUtils.schemaJsonToStructType(json)
  }

  // ---- TableCatalog ----

  override def loadTable(ident: Identifier): Table = {
    val uri = uriFor(ident.namespace(), ident.name())
    new BenoStreamTable(uri, schemaOf(uri, ident), new java.util.HashMap[String, String](), gpuDevice)
  }

  override def createTable(
      ident: Identifier,
      schema: StructType,
      partitions: Array[Transform],
      properties: JMap[String, String]
  ): Table = {
    val uri = uriFor(ident.namespace(), ident.name())
    if (!bridge.createTable(uri, BenoStreamArrowUtils.structTypeToSchemaJson(schema)))
      throw new TableAlreadyExistsException(uri)
    new BenoStreamTable(uri, schema, properties, gpuDevice)
  }

  override def alterTable(ident: Identifier, changes: TableChange*): Table =
    throw new UnsupportedOperationException("alterTable is not supported by the BenoStreamDB catalog")

  override def dropTable(ident: Identifier): Boolean =
    bridge.dropTable(uriFor(ident.namespace(), ident.name()))

  override def renameTable(oldIdent: Identifier, newIdent: Identifier): Unit =
    throw new UnsupportedOperationException("renameTable is not supported by the BenoStreamDB catalog")

  override def listTables(namespace: Array[String]): Array[Identifier] =
    BenoStreamArrowUtils
      .parseStringArray(bridge.listTables(warehouse, namespace.lastOption.getOrElse("default")))
      .map(t => Identifier.of(namespace, t))
      .toArray

  // ---- SupportsNamespaces ----

  override def listNamespaces(): Array[Array[String]] =
    BenoStreamArrowUtils.parseStringArray(bridge.listSchemas(warehouse)).map(s => Array(s)).toArray

  override def listNamespaces(namespace: Array[String]): Array[Array[String]] =
    if (namespace.isEmpty) listNamespaces() else Array.empty

  override def loadNamespaceMetadata(namespace: Array[String]): JMap[String, String] =
    new java.util.HashMap[String, String]()

  override def createNamespace(namespace: Array[String], metadata: JMap[String, String]): Unit =
    bridge.createSchema(warehouse, namespace.last)

  override def alterNamespace(namespace: Array[String], changes: NamespaceChange*): Unit =
    throw new UnsupportedOperationException("alterNamespace is not supported by the BenoStreamDB catalog")

  override def dropNamespace(namespace: Array[String], cascade: Boolean): Boolean =
    throw new UnsupportedOperationException("dropNamespace is not supported by the BenoStreamDB catalog")
}
