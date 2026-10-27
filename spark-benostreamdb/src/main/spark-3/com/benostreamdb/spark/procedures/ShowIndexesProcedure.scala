package com.benostreamdb.spark.procedures

import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}
import org.apache.spark.sql.connector.iceberg.catalog.{Procedure, ProcedureParameter}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.unsafe.types.UTF8String

import scala.jdk.CollectionConverters._

/**
 * `show_indexes(table)`: lists the indexes on a BenoStreamDB table.
 *
 * Returns the same five columns as the Trino `system.list_indexes` table
 * function instead of a single `(table, indexes_json)` blob, so both connectors
 * expose the same shape.
 */
class ShowIndexesProcedure extends Procedure {

  override def description(): String = "Lists all indexes for a BenoStreamDB table"

  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.required("table", DataTypes.StringType)
  )

  override def outputType(): StructType = ShowIndexesProcedure.OutputSchema

  override def call(inputArgs: InternalRow): Array[InternalRow] = {
    val table = inputArgs.getString(0)
    val tableUri = BenoStreamTableResolver.resolveFromSession(table)

    val jniBridge = com.benostreamdb.spark.jni.BenoStreamJNIBridge.getInstance()
    val indexesJson = jniBridge.listIndexes(tableUri)

    ShowIndexesProcedure.rows(indexesJson).toArray
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
