package com.benostreamdb.spark

import org.apache.spark.sql.connector.catalog.{CatalogPlugin, Identifier, ProcedureCatalog}
import org.apache.spark.sql.connector.catalog.procedures.UnboundProcedure
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import com.benostreamdb.spark.procedures.{
  AddIndexProcedure,
  BuildIndexProcedure,
  CompactTableProcedure,
  DropIndexProcedure,
  RebuildIndexProcedure,
  SetPrimaryKeyProcedure,
  ShowIndexesProcedure,
  RegionalDriftProcedure
}

/**
 * BenoStreamProcedureCatalog provides our custom Stored Procedures.
 * Users configure: spark.sql.catalog.benostream=com.benostreamdb.spark.BenoStreamProcedureCatalog
 */
class BenoStreamProcedureCatalog extends ProcedureCatalog with CatalogPlugin {

  private var catalogName: String = "benostream"

  override def initialize(name: String, options: CaseInsensitiveStringMap): Unit = {
    this.catalogName = name
  }

  override def name(): String = catalogName

  override def loadProcedure(ident: Identifier): UnboundProcedure = {
    val namespace = ident.namespace()
    val procName = ident.name()

    if (namespace.length == 1 && namespace(0).equalsIgnoreCase("system")) {
      procName.toLowerCase match {
        case "add_index" => new AddIndexProcedure()
        case "drop_index" => new DropIndexProcedure()
        case "build_index" => new BuildIndexProcedure()
        case "rebuild_index" => new RebuildIndexProcedure()
        case "compact" | "optimize" => new CompactTableProcedure()
        case "show_indexes" | "list_indexes" => new ShowIndexesProcedure()
        case "set_primary_key" => new SetPrimaryKeyProcedure()
        case "regional_drift_search" => new RegionalDriftProcedure()
        case _ => throw new UnsupportedOperationException(s"Unknown procedure: $procName")
      }
    } else {
      throw new UnsupportedOperationException(s"Unknown procedure namespace: ${namespace.mkString(".")}")
    }
  }

  def listProcedures(namespace: Array[String]): Array[Identifier] = {
    if (namespace.length == 1 && namespace(0).equalsIgnoreCase("system")) {
      Array(
        Identifier.of(namespace, "add_index"),
        Identifier.of(namespace, "drop_index"),
        Identifier.of(namespace, "build_index"),
        Identifier.of(namespace, "rebuild_index"),
        Identifier.of(namespace, "compact"),
        Identifier.of(namespace, "show_indexes"),
        Identifier.of(namespace, "set_primary_key"),
        Identifier.of(namespace, "regional_drift_search")
      )
    } else {
      Array.empty[Identifier]
    }
  }
}
