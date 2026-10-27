package com.benostreamdb.spark

import com.benostreamdb.spark.procedures.BenoStreamTableResolver
import org.junit.Assert._
import org.junit.Test

/**
 * Unit coverage for `BenoStreamTableResolver`, the helper every stored procedure
 * uses to turn its `table` argument into an engine URI.
 *
 * The regression this guards against: `table.split(".").last` dropped the
 * namespace, so `benostream.default.users` and `users` were indistinguishable and
 * resolved against the wrong location.
 */
class BenoStreamTableResolverTest {

  private val warehouse = "s3://bucket/warehouse/"

  @Test
  def bareTableResolvesToDefaultSchema(): Unit = {
    assertEquals(
      "s3://bucket/warehouse/default/users",
      BenoStreamTableResolver.resolve(warehouse, "users")
    )
  }

  @Test
  def schemaQualifiedTableKeepsTheNamespace(): Unit = {
    assertEquals(
      "s3://bucket/warehouse/sales/users",
      BenoStreamTableResolver.resolve(warehouse, "sales.users")
    )
  }

  @Test
  def catalogSchemaTableDropsOnlyTheCatalog(): Unit = {
    assertEquals(
      "s3://bucket/warehouse/sales/users",
      BenoStreamTableResolver.resolve(warehouse, "benostream.sales.users")
    )
  }

  @Test
  def explicitUriPassesThrough(): Unit = {
    val uri = "file:///tmp/warehouse/default/users"
    assertEquals(uri, BenoStreamTableResolver.resolve(warehouse, uri))
  }

  @Test
  def whitespaceAndTrailingSlashesAreNormalized(): Unit = {
    assertEquals(
      "s3://bucket/warehouse/default/users",
      BenoStreamTableResolver.resolve("  s3://bucket/warehouse//  ", "  users  ")
    )
  }

  @Test(expected = classOf[IllegalArgumentException])
  def emptyTableArgumentIsRejected(): Unit = {
    BenoStreamTableResolver.resolve(warehouse, "   ")
  }

  @Test(expected = classOf[IllegalArgumentException])
  def emptyWarehouseIsRejected(): Unit = {
    BenoStreamTableResolver.resolve("", "users")
  }

  @Test(expected = classOf[IllegalArgumentException])
  def tooManySegmentsAreRejected(): Unit = {
    BenoStreamTableResolver.resolve(warehouse, "a.b.c.d")
  }
}
