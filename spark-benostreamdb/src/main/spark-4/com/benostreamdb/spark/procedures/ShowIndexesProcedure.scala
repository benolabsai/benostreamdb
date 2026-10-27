package com.benostreamdb.spark.procedures

import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.connector.read.Scan
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.unsafe.types.UTF8String

import scala.jdk.CollectionConverters._

/**
 * `show_indexes(table)`: lists the indexes on a BenoStreamDB table.
 *
 * The procedure used to call `listIndexes` and then discard the JSON, returning
 * an empty iterator. It now returns the same five columns as the Trino
 * `system.list_indexes` table function (column_name, index_category, algorithm,
 * index_id, format_version).
 */
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
    val tableUri = BenoStreamTableResolver.resolveFromSession(table)

    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val indexesJson = jniBridge.listIndexes(tableUri)

    // `Iterator` is invariant, so the list must be typed as `Scan` rather than
    // as the concrete scan class.
    val scans: java.util.List[Scan] =
      java.util.Collections.singletonList[Scan](
        new BenoStreamProcedureScan(
          ShowIndexesProcedure.OutputSchema,
          ShowIndexesProcedure.rows(indexesJson)))
    scans.iterator()
  }
}

object ShowIndexesProcedure {

  /** Mirrors `ListIndexesPageSource` in the Trino connector. */
  val OutputSchema: StructType = new StructType()
    .add("column_name", DataTypes.StringType)
    .add("index_category", DataTypes.StringType)
    .add("algorithm", DataTypes.StringType)
    .add("index_id", DataTypes.StringType)
    .add("format_version", DataTypes.LongType)

  private val mapper = new ObjectMapper()

  private def text(record: JsonNode, field: String): String = {
    val value = record.get(field)
    if (value == null || value.isNull) "" else value.asText()
  }

  /** Parse the engine's JSON array of index records into result rows. */
  def rows(json: String): Seq[InternalRow] = {
    val root =
      try mapper.readTree(if (json == null || json.trim.isEmpty) "[]" else json)
      catch { case _: Throwable => null }

    if (root == null || !root.isArray) {
      Seq.empty
    } else {
      root.elements().asScala.toSeq.map { record =>
        new GenericInternalRow(Array[Any](
          UTF8String.fromString(text(record, "column_name")),
          UTF8String.fromString(text(record, "index_category")),
          UTF8String.fromString(text(record, "algorithm")),
          UTF8String.fromString(text(record, "index_id")),
          record.path("format_version").asLong(0L)
        ))
      }
    }
  }
}
