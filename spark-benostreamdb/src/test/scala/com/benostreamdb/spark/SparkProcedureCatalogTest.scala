package com.benostreamdb.spark

import org.apache.spark.sql.connector.catalog.Identifier
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.junit.Test
import org.junit.Assert._

/**
 * Verifies the DSv2 stored-procedure surface of `BenoStreamProcedureCatalog`.
 *
 * The class is shared by the Spark 3.5 (Iceberg's `ProcedureCatalog` shim) and
 * Spark 4.x (native `ProcedureCatalog`) builds, so it must compile against both
 * catalog variants. `listProcedures` only exists on the Spark 4.x catalog, so it
 * is exercised reflectively to keep a single test source compiling everywhere.
 */
class SparkProcedureCatalogTest {

  // Canonical procedure names, plus the documented aliases.
  private val canonicalProcedures = Seq(
    "add_index",
    "drop_index",
    "build_index",
    "rebuild_index",
    "compact",
    "show_indexes",
    "set_primary_key",
    "drop_primary_key",
    "regional_drift_search"
  )

  private val aliases = Seq(
    "optimize",     // -> compact
    "list_indexes"  // -> show_indexes
  )

  private def newCatalog: BenoStreamProcedureCatalog = {
    val catalog = new BenoStreamProcedureCatalog()
    catalog.initialize("benostream", CaseInsensitiveStringMap.empty())
    catalog
  }

  @Test
  def testProcedureCatalogLoadAllProcedures(): Unit = {
    val catalog = newCatalog
    assertEquals("benostream", catalog.name())

    val systemNamespace = Array("system")
    for (procName <- canonicalProcedures ++ aliases) {
      val proc = catalog.loadProcedure(Identifier.of(systemNamespace, procName))
      assertNotNull(s"Procedure $procName should load successfully", proc)
      assertTrue(s"Description for $procName should be non-empty", proc.description().nonEmpty)
    }
  }

  @Test
  def testListProceduresWhenSupported(): Unit = {
    val catalog = newCatalog
    // `listProcedures` was added to Spark's `ProcedureCatalog` in 4.2; the Spark
    // 3.5 catalog does not have it. Invoke it reflectively so this one test file
    // compiles across both, and assert the listing when the method is present.
    val listing = try {
      Option(
        catalog.getClass
          .getMethod("listProcedures", classOf[Array[String]])
          .invoke(catalog, Array("system"))
      )
    } catch {
      case _: NoSuchMethodException => None
    }

    listing.foreach { result =>
      val idents = result.asInstanceOf[Array[Identifier]]
      val names = idents.map(_.name()).toSet
      val missing = canonicalProcedures.toSet.diff(names)
      assertTrue(s"listProcedures omitted: ${missing.mkString(", ")}", missing.isEmpty)
    }
  }

  @Test(expected = classOf[UnsupportedOperationException])
  def testUnknownProcedureThrows(): Unit = {
    newCatalog.loadProcedure(Identifier.of(Array("system"), "non_existent_procedure"))
  }

  @Test(expected = classOf[UnsupportedOperationException])
  def testInvalidNamespaceThrows(): Unit = {
    newCatalog.loadProcedure(Identifier.of(Array("invalid_namespace"), "add_index"))
  }
}
