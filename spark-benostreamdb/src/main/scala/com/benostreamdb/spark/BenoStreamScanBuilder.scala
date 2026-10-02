package com.benostreamdb.spark

import org.apache.spark.sql.connector.read.{
  Scan,
  ScanBuilder,
  SupportsPushDownFilters,
  SupportsPushDownRequiredColumns,
  SupportsRuntimeFiltering
}
import org.apache.spark.sql.connector.catalog.SupportsRead
import org.apache.spark.sql.connector.expressions.NamedReference
import org.apache.spark.sql.sources.{EqualTo, Filter, In}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import com.benostreamdb.spark.jni.BenoStreamJNIBridge
import org.slf4j.LoggerFactory

class BenoStreamScanBuilder(
    val table: BenoStreamTable,
    val options: CaseInsensitiveStringMap
) extends ScanBuilder
    with SupportsPushDownFilters
    with SupportsPushDownRequiredColumns {

  private val logger = LoggerFactory.getLogger(classOf[BenoStreamScanBuilder])

  private val delegateBuilder: ScanBuilder = table.delegate match {
    case r: SupportsRead => r.newScanBuilder(options)
    case _ => throw new UnsupportedOperationException("Underlying table does not support read")
  }

  private var pushed: Array[Filter] = Array.empty

  override def pushFilters(filters: Array[Filter]): Array[Filter] = {
    val unhandled = delegateBuilder match {
      case f: SupportsPushDownFilters => f.pushFilters(filters)
      case _ => filters
    }

    this.pushed = filters.filter(isIndexedFilter)

    if (this.pushed.nonEmpty && BenoStreamJNIBridge.isLoaded) {
      try {
        val jni = BenoStreamJNIBridge.getInstance()
        com.benostreamdb.spark.gpu.GpuContextResolver.bindTaskGpuContext(table.gpuDevice)
        for (filter <- this.pushed) {
          extractFilterKeyValues(filter).foreach { case (colName, valuesJson) =>
            logger.info(s"Pushing down index filter on column $colName to BenoStreamDB native engine")
            jni.queryIndexIn(table.name(), colName, valuesJson)
          }
        }
      } catch {
        case e: UnsatisfiedLinkError =>
          logger.warn(s"Native queryIndexIn not linked: ${e.getMessage}")
      }
    }

    unhandled
  }

  override def pushedFilters(): Array[Filter] = {
    delegateBuilder match {
      case f: SupportsPushDownFilters => f.pushedFilters()
      case _ => pushed
    }
  }

  override def pruneColumns(requiredSchema: StructType): Unit = {
    delegateBuilder match {
      case c: SupportsPushDownRequiredColumns => c.pruneColumns(requiredSchema)
      case _ => // no-op
    }
  }

  override def build(): Scan = {
    new BenoStreamScan(delegateBuilder.build(), table, pushed)
  }

  private def isIndexedFilter(filter: Filter): Boolean = {
    val indexedCols = Option(table.properties.get("indexed_columns"))
      .map(_.split(",").map(_.trim.toLowerCase).toSet)
      .getOrElse(Set.empty)
    val pkCols = Option(table.properties.get("primary_key"))
      .map(_.split(",").map(_.trim.toLowerCase).toSet)
      .getOrElse(Set("id"))

    val targets = indexedCols ++ pkCols
    filter.references.exists(r => targets.contains(r.toLowerCase))
  }

  private def extractFilterKeyValues(filter: Filter): Option[(String, String)] = {
    filter match {
      case EqualTo(attribute, value) =>
        val safeVal = if (value == null) "null" else value.toString.replace("\"", "\\\"")
        Some((attribute, s"""["$safeVal"]"""))
      case In(attribute, values) =>
        val jsonArray = values.map { v =>
          if (v == null) "null" else s""""${v.toString.replace("\"", "\\\"")}""""
        }.mkString("[", ",", "]")
        Some((attribute, jsonArray))
      case _ => None
    }
  }
}

class BenoStreamScan(
    val delegate: Scan,
    val table: BenoStreamTable,
    val initialPushedFilters: Array[Filter] = Array.empty
) extends Scan with SupportsRuntimeFiltering {

  private var dynamicFilters: Array[Filter] = initialPushedFilters

  override def readSchema(): StructType = delegate.readSchema()

  override def toBatch() = delegate.toBatch()

  override def filterAttributes(): Array[NamedReference] = {
    val pkString = Option(table.properties.get("primary_key")).getOrElse("id")
    pkString.split(",").map(_.trim).map(col => 
      org.apache.spark.sql.connector.expressions.Expressions.column(col)
    ).toArray
  }

  override def filter(filters: Array[Filter]): Unit = {
    if (BenoStreamJNIBridge.isLoaded) {
      try {
        val jniBridge = BenoStreamJNIBridge.getInstance()
        com.benostreamdb.spark.gpu.GpuContextResolver.bindTaskGpuContext(table.gpuDevice)
        val pkString = Option(table.properties.get("primary_key")).getOrElse("id")
        jniBridge.queryIndexIn(table.name(), pkString, "[]")
      } catch {
        case e: UnsatisfiedLinkError =>
          // Graceful fallback for mock/test runs without native binary
      }
    }
  }
}
