package com.benostreamdb.spark

import org.apache.spark.sql.connector.catalog.Identifier
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.junit.Test
import org.junit.Assert._

class SparkProcedureCatalogTest {

  @Test
  def testProcedureCatalogLoadAllProcedures(): Unit = {
    val catalog = new BenoStreamProcedureCatalog()
    catalog.initialize("benostream", CaseInsensitiveStringMap.empty())

    assertEquals("benostream", catalog.name())

    val systemNamespace = Array("system")
    val procedures = Seq(
      "add_index",
      "drop_index",
      "build_index",
      "rebuild_index",
      "compact",
      "optimize",
      "show_indexes",
      "list_indexes",
      "set_primary_key"
    )

    for (procName <- procedures) {
      val proc = catalog.loadProcedure(Identifier.of(systemNamespace, procName))
      assertNotNull(s"Procedure $procName should load successfully", proc)
      assertTrue(s"Description for $procName should be non-empty", proc.description().nonEmpty)
    }
  }

  @Test(expected = classOf[UnsupportedOperationException])
  def testUnknownProcedureThrows(): Unit = {
    val catalog = new BenoStreamProcedureCatalog()
    catalog.initialize("benostream", CaseInsensitiveStringMap.empty())
    catalog.loadProcedure(Identifier.of(Array("system"), "non_existent_procedure"))
  }

  @Test(expected = classOf[UnsupportedOperationException])
  def testInvalidNamespaceThrows(): Unit = {
    val catalog = new BenoStreamProcedureCatalog()
    catalog.initialize("benostream", CaseInsensitiveStringMap.empty())
    catalog.loadProcedure(Identifier.of(Array("invalid_namespace"), "add_index"))
  }
}
