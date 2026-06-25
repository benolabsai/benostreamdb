package com.hyperstreamdb.spark

import org.apache.spark.sql.connector.catalog.{CatalogPlugin, Identifier}
import org.apache.spark.sql.connector.iceberg.catalog.Procedure
import org.apache.spark.sql.connector.iceberg.catalog.ProcedureCatalog
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import com.hyperstreamdb.spark.procedures.{AddIndexProcedure, BuildIndexProcedure, SetPrimaryKeyProcedure}

/**
 * HyperStreamProcedureCatalog provides our custom Stored Procedures.
 * Users configure: spark.sql.catalog.hyperstream=com.hyperstreamdb.spark.HyperStreamProcedureCatalog
 */
class HyperStreamProcedureCatalog extends ProcedureCatalog with CatalogPlugin {

  private var catalogName: String = "hyperstream"

  override def initialize(name: String, options: CaseInsensitiveStringMap): Unit = {
    this.catalogName = name
  }

  override def name(): String = catalogName

  override def loadProcedure(ident: Identifier): Procedure = {
    val namespace = ident.namespace()
    val procName = ident.name()

    if (namespace.length == 1 && namespace(0).equalsIgnoreCase("system")) {
      procName.toLowerCase match {
        case "add_index" => new AddIndexProcedure()
        case "build_index" => new BuildIndexProcedure()
        case "set_primary_key" => new SetPrimaryKeyProcedure()
        case _ => throw new UnsupportedOperationException(s"Unknown procedure: $procName")
      }
    } else {
      throw new UnsupportedOperationException(s"Unknown procedure namespace: ${namespace.mkString(".")}")
    }
  }
}
