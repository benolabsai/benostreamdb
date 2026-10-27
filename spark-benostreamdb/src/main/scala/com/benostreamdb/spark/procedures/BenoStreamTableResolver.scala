package com.benostreamdb.spark.procedures

import org.apache.spark.sql.SparkSession

/**
 * Resolves the `table` procedure argument into a full engine table URI.
 *
 * Stored procedures used to do `table.split("\\.").last`, which dropped the
 * namespace entirely: `benostream.default.orders`, `default.orders` and
 * `orders` all collapsed to `orders` and were handed to JNI with no namespace
 * context. This helper keeps the namespace and prefixes the catalog warehouse:
 *
 * {{{
 *   "orders"                   -> "<warehouse>/default/orders"
 *   "sales.orders"             -> "<warehouse>/sales/orders"
 *   "benostream.sales.orders"  -> "<warehouse>/sales/orders"
 *   "file:///tmp/orders"       -> "file:///tmp/orders"   (explicit URI, unchanged)
 * }}}
 */
object BenoStreamTableResolver {

  /** Explicit URIs are passed through untouched. */
  private val UriScheme = "://"

  /** Preferred catalog config key; other `spark.sql.catalog.*.warehouse` keys work too. */
  val DefaultWarehouseKey = "spark.sql.catalog.benostream.warehouse"

  /** Session-level fallback used by the Spark connector docs. */
  val WarehouseConfKey = "spark.benostream.warehouse"

  /** Resolve `tableArg` against an explicit warehouse location. */
  def resolve(warehouse: String, tableArg: String): String = {
    val table = Option(tableArg).map(_.trim).getOrElse("")
    if (table.isEmpty) {
      throw new IllegalArgumentException("table argument must not be empty")
    }
    if (table.contains(UriScheme)) {
      return table
    }

    val base = normalize(warehouse)
    if (base.isEmpty) {
      throw new IllegalArgumentException(
        s"cannot resolve '$tableArg' without a warehouse; configure $DefaultWarehouseKey")
    }

    val parts = table.split("\\.").map(_.trim).filter(_.nonEmpty)
    parts.length match {
      case 1 => s"$base/default/${parts(0)}"
      case 2 => s"$base/${parts(0)}/${parts(1)}"
      case 3 => s"$base/${parts(1)}/${parts(2)}"
      case _ =>
        throw new IllegalArgumentException(
          s"Invalid table argument '$tableArg'; expected [catalog.schema.]table or a URI")
    }
  }

  /** Resolve `tableArg` using the warehouse configured for this connector. */
  def resolveFromSession(tableArg: String): String = resolve(warehouseFromSession(), tableArg)

  /**
   * Discover the warehouse from the session: the `benostream` catalog first,
   * then any other `spark.sql.catalog.*.warehouse`, then the session-level
   * `spark.benostream.warehouse` fallback.
   */
  def warehouseFromSession(): String = {
    val session = SparkSession.getActiveSession
      .orElse(SparkSession.getDefaultSession)
      .getOrElse(throw new IllegalStateException(
        "no active SparkSession; cannot resolve the BenoStreamDB warehouse"))

    val entries = session.conf.getAll
    val catalogWarehouses = entries.keys
      .filter(key => key.startsWith("spark.sql.catalog.") && key.endsWith(".warehouse"))
      .toSeq
      .sortBy(key => (if (key == DefaultWarehouseKey) 0 else 1, key))

    catalogWarehouses.headOption
      .map(entries(_))
      .orElse(entries.get(WarehouseConfKey))
      .map(normalize)
      .filter(_.nonEmpty)
      .getOrElse(throw new IllegalStateException(
        s"BenoStreamDB warehouse is not configured; set $DefaultWarehouseKey or $WarehouseConfKey"))
  }

  /** Trim, and drop *all* trailing slashes so a joined path never doubles them. */
  private def normalize(warehouse: String): String =
    Option(warehouse).map(_.trim).getOrElse("").replaceAll("/+$", "")
}
