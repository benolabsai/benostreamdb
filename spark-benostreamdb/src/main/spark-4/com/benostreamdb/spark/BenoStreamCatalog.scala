package com.benostreamdb.spark

import org.apache.spark.sql.connector.catalog.{Identifier, ProcedureCatalog}
import org.apache.spark.sql.connector.catalog.procedures.UnboundProcedure
import com.benostreamdb.spark.procedures.{
  AddIndexProcedure,
  BuildIndexProcedure,
  CompactTableProcedure,
  DropIndexProcedure,
  DropPrimaryKeyProcedure,
  RebuildIndexProcedure,
  SetPrimaryKeyProcedure,
  ShowIndexesProcedure,
  RegionalDriftProcedure
}

/**
 * The single BenoStreamDB catalog for Spark 4.x: native tables/namespaces
 * (inherited from [[BenoStreamCatalog]], which resolves everything through the
 * engine's JNI metadata surface) plus stored procedures under `system.*`.
 *
 * Spark 4.0 introduced the native DSv2 `ProcedureCatalog` API, so this build
 * needs no Iceberg runtime at all.
 *
 * Configure: spark.sql.catalog.benostream=com.benostreamdb.spark.BenoStreamProcedureCatalog
 */
class BenoStreamProcedureCatalog extends BenoStreamCatalog with ProcedureCatalog {

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
        case "drop_primary_key" => new DropPrimaryKeyProcedure()
        case "regional_drift_search" => new RegionalDriftProcedure()
        case _ => throw new UnsupportedOperationException(s"Unknown procedure: $procName")
      }
    } else {
      throw new UnsupportedOperationException(s"Unknown procedure namespace: ${namespace.mkString(".")}")
    }
  }

  // NOTE: `listProcedures` is a *new* method on Spark 4.0/4.1 (where
  // `ProcedureCatalog` only declares `loadProcedure`) and an *abstract* member
  // of `ProcedureCatalog` from Spark 4.2 onwards. Implementing an abstract
  // member does not require `override`, and defining a brand-new method must
  // not use it — so omitting the modifier is what lets this single source root
  // compile against every 4.x minor.
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
        Identifier.of(namespace, "drop_primary_key"),
        Identifier.of(namespace, "regional_drift_search")
      )
    } else {
      Array.empty[Identifier]
    }
  }
}
