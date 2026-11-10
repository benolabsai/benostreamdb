package com.benostreamdb.spark

import com.benostreamdb.spark.jni.BenoStreamJNIBridge
// `scala.collection.JavaConverters` (not `scala.jdk.CollectionConverters`) because
// this file is shared by the Scala 2.12 (Spark 3.5) and 2.13 (Spark 4.x) builds.
// It matches BenoStreamJson, the other shared converter user.
import scala.collection.JavaConverters._
import org.apache.arrow.c.{ArrowArray, ArrowSchema, Data}
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.expressions.{Expression, GeneralScalarExpression, Literal, NamedReference, SortDirection, SortOrder}
import org.apache.spark.sql.connector.read._
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.unsafe.types.UTF8String
import org.slf4j.LoggerFactory

/**
 * Native scan builder. Two modes:
 *  - normal read: one partition running a pushed-down DataFusion SQL query.
 *  - merge-on-read (row-level ops): Spark requests `_file`/`_pos` metadata
 *    columns; we scan per data file and synthesize them (segment path + row
 *    offset) so the DeltaWriter can target exact rows via position deletes.
 */
class BenoStreamScanBuilder(table: BenoStreamTable)
    extends ScanBuilder
    with SupportsPushDownFilters
    with SupportsPushDownRequiredColumns
    with SupportsPushDownTopN {

  private val logger = LoggerFactory.getLogger(classOf[BenoStreamScanBuilder])
  private var requiredSchema: StructType = table.structSchema
  private var handled: Array[Filter] = Array.empty
  private var unhandled: Array[Filter] = Array.empty
  private var topNOrdering: Array[SortOrder] = Array.empty
  private var topNLimit: Option[Int] = None

  @native def listDataFiles(tableUri: String): String

  /** File-level scan splits (one entry per Parquet file) for parallel reads. */
  @native def listSplits(tableUri: String): String

  BenoStreamJNIBridge.getInstance() // ensure the library is loaded for natives

  override def pushFilters(filters: Array[Filter]): Array[Filter] = {
    val (ok, rest) = filters.partition(BenoStreamSql.filterToSql(_).isDefined)
    handled = ok
    unhandled = rest
    rest
  }

  override def pushedFilters(): Array[Filter] = handled

  override def pruneColumns(requiredSchema: StructType): Unit = {
    this.requiredSchema = requiredSchema
  }

  /**
   * Push `ORDER BY … LIMIT k` into the engine.
   *
   * Spark hands over the sort the same way it hands over filters, and unlike
   * Trino's `SortItem` it carries a full `Expression`. That matters here: a KNN
   * query sorts by `cosine_distance(embedding, …)`, which is an expression, not
   * a column. Rendering it back into the scan SQL is what lets the engine's KNN
   * optimizer rule see the query and use the index instead of scanning.
   *
   * Returning false is always safe — Spark then sorts and limits the rows
   * itself, which is slower but correct.
   */
  override def pushTopN(ordering: Array[SortOrder], limit: Int): Boolean = {
    if (limit <= 0 || ordering.isEmpty) {
      false
    } else {
      val rendered = ordering.map(BenoStreamSql.sortOrderToSql)
      if (rendered.exists(_.isEmpty)) {
        logger.debug(
          "benostream: not pushing TopN — could not render {} of {} sort expressions",
          rendered.count(_.isEmpty),
          rendered.length
        )
        false
      } else {
        topNOrdering = ordering
        topNLimit = Some(limit)
        true
      }
    }
  }

  override def build(): Scan = {
    val isDelta = requiredSchema.exists(f => f.name == "_file" || f.name == "_pos")
    if (isDelta) {
      val dataSchema = StructType(requiredSchema.fields.filterNot(f => f.name == "_file" || f.name == "_pos"))
      val files = parseFilePaths(BenoStreamScanBuilder.this.listDataFiles(table.tableUri))
      new BenoStreamDeltaScan(table.tableUri, requiredSchema, dataSchema, files)
    } else {
      // Multi-partition read: one task per data file, when the scan carries no
      // pushed-down predicate (so dropping the single SQL scan loses nothing) and
      // the table has more than one file. The per-file reader is the same one the
      // merge-on-read path already uses.
      // A pushed TopN must stay a single SQL scan: a per-file fan-out cannot
      // produce a global top-k.
      val splitFiles =
        if (handled.isEmpty && topNLimit.isEmpty && BenoStreamScanBuilder.splitFanoutEnabled()) {
          parseSplitFiles(BenoStreamScanBuilder.this.listSplits(table.tableUri))
        } else {
          Seq.empty
        }
      if (splitFiles.length > 1) {
        new BenoStreamSplitScan(table.tableUri, requiredSchema, splitFiles)
      } else {
        val orderBy =
          if (topNOrdering.isEmpty) {
            None
          } else {
            Some(topNOrdering.flatMap(BenoStreamSql.sortOrderToSql).mkString(", "))
          }
        val sql = BenoStreamSql.buildScanSql(
          table.structSchema,
          requiredSchema,
          handled,
          orderBy.filter(_.nonEmpty),
          topNLimit
        )
        new BenoStreamScan(table.tableUri, requiredSchema, sql)
      }
    }
  }

  /**
   * Split file paths, or nothing when a file carries merge-on-read deletes: a
   * raw per-file read would resurrect deleted rows, so such tables stay on the
   * single SQL scan.
   */
  private def parseSplitFiles(json: String): Seq[String] = {
    if (json == null || json.trim.isEmpty) return Seq.empty
    val root =
      try BenoStreamScanBuilder.mapper.readTree(json)
      catch { case _: Throwable => null }
    if (root == null || !root.isArray) return Seq.empty
    val nodes = root.elements().asScala.toSeq
    val hasDeletes = nodes.exists(node => node.path("has_deletes").asBoolean(false))
    if (hasDeletes) {
      Seq.empty
    } else {
      nodes.flatMap(node => Option(node.get("file_path")).map(_.asText())).filter(_.nonEmpty)
    }
  }

  /** Accept both `["path", ...]` and `[{"file_path": "..."}, ...]` shapes. */
  private def parseFilePaths(json: String): Seq[String] = {
    if (json == null || json.trim.isEmpty) return Seq.empty
    val root =
      try BenoStreamScanBuilder.mapper.readTree(json)
      catch { case _: Throwable => null }
    if (root == null || !root.isArray) {
      Seq.empty
    } else {
      root.elements().asScala.flatMap { node =>
        if (node.isTextual) Some(node.asText())
        else Option(node.get("file_path")).map(_.asText())
      }.filter(path => path != null && path.nonEmpty).toSeq
    }
  }
}

object BenoStreamScanBuilder {
  /** Advisory split size; splitting is file-granular (<= 0 disables fan-out). */
  val SplitMaxSizeConf = "spark.benostream.split.maxSize"
  val DefaultSplitMaxSizeBytes: Long = 128L * 1024 * 1024

  private[spark] val mapper = new com.fasterxml.jackson.databind.ObjectMapper()

  private[spark] def splitFanoutEnabled(): Boolean = {
    // `RuntimeConfig` has no typed getLong in every supported Spark version, and
    // a non-numeric value must not fail the scan.
    val raw = Option(org.apache.spark.sql.SparkSession.getActiveSession)
      .flatten
      .map(_.conf.get(SplitMaxSizeConf, DefaultSplitMaxSizeBytes.toString))
      .getOrElse(DefaultSplitMaxSizeBytes.toString)
    scala.util.Try(raw.trim.toLong).getOrElse(DefaultSplitMaxSizeBytes) > 0L
  }
}

// ---- Multi-partition scan: one partition per data file ----

class BenoStreamSplitScan(uri: String, readSchema: StructType, files: Seq[String]) extends Scan {
  override def toBatch: Batch = new BenoStreamSplitBatch(uri, readSchema, files)
  override def readSchema(): StructType = readSchema
}

class BenoStreamSplitBatch(uri: String, readSchema: StructType, files: Seq[String]) extends Batch {
  override def planInputPartitions(): Array[InputPartition] =
    files.map(file => BenoStreamSplitPartition(uri, file, readSchema)).toArray
  override def createReaderFactory(): PartitionReaderFactory = BenoStreamSplitReaderFactory
}

case class BenoStreamSplitPartition(uri: String, file: String, readSchema: StructType) extends InputPartition

object BenoStreamSplitReaderFactory extends PartitionReaderFactory {
  override def createReader(partition: InputPartition): PartitionReader[InternalRow] = {
    val p = partition.asInstanceOf[BenoStreamSplitPartition]
    // Same per-file reader as the merge-on-read path; `_file`/`_pos` are simply
    // absent from the read schema here.
    new BenoStreamPartitionReader(p.file, p.readSchema, p.readSchema)
  }
}

// ---- Normal SQL scan (unchanged behaviour) ----

class BenoStreamScan(uri: String, readSchemaType: StructType, sql: String) extends Scan {
  override def toBatch: Batch = new BenoStreamBatch(uri, readSchemaType, sql)
  override def readSchema(): StructType = readSchemaType
}

class BenoStreamBatch(uri: String, readSchema: StructType, sql: String) extends Batch {
  override def planInputPartitions(): Array[InputPartition] = Array(BenoStreamInputPartition(uri, readSchema, sql))
  override def createReaderFactory(): PartitionReaderFactory = BenoStreamPartitionReaderFactory
}

case class BenoStreamInputPartition(uri: String, readSchema: StructType, sql: String) extends InputPartition

object BenoStreamPartitionReaderFactory extends PartitionReaderFactory {
  override def createReader(partition: InputPartition): PartitionReader[InternalRow] = {
    val p = partition.asInstanceOf[BenoStreamInputPartition]
    new BenoStreamQueryReader(p.uri, p.readSchema, p.sql)
  }
}

class BenoStreamQueryReader(uri: String, readSchema: StructType, sql: String) extends PartitionReader[InternalRow] {
  private val bridge = BenoStreamJNIBridge.getInstance()
  private val allocator = new RootAllocator()
  private val handle: Long = bridge.openQuery(uri, sql)
  private var root: VectorSchemaRoot = null
  private var pos = -1
  private var current: InternalRow = null

  private def loadNext(): Boolean = {
    closeRoot()
    if (handle == 0L) return false
    val array = ArrowArray.allocateNew(allocator)
    val schema = ArrowSchema.allocateNew(allocator)
    try {
      if (bridge.readQueryBatch(handle, array.memoryAddress(), schema.memoryAddress()) == 0L) {
        array.close(); schema.close(); return false
      }
      root = Data.importVectorSchemaRoot(allocator, array, schema, null)
      pos = 0
      root.getRowCount > 0
    } catch { case e: Throwable => array.close(); schema.close(); throw e }
  }

  private def closeRoot(): Unit = if (root != null) { root.close(); root = null }

  override def next(): Boolean = {
    if (root == null || pos >= root.getRowCount) if (!loadNext()) return false
    val vectors = readSchema.fields.map(f => root.getVector(f.name))
    current = BenoStreamArrowUtils.toInternalRow(vectors.toSeq, pos, readSchema)
    pos += 1
    true
  }
  override def get(): InternalRow = current
  override def close(): Unit = { closeRoot(); if (handle != 0L) bridge.closeQuery(handle); allocator.close() }
}

// ---- Merge-on-read per-file scan emitting _file/_pos ----

class BenoStreamDeltaScan(uri: String, val fullReadSchema: StructType, val dataSchema: StructType, files: Seq[String]) extends Scan {
  override def toBatch: Batch = new BenoStreamDeltaBatch(uri, fullReadSchema, dataSchema, files)
  override def readSchema(): StructType = fullReadSchema
}

class BenoStreamDeltaBatch(uri: String, fullReadSchema: StructType, dataSchema: StructType, files: Seq[String]) extends Batch {
  override def planInputPartitions(): Array[InputPartition] = files.map(f => BenoStreamFilePartition(uri, f, fullReadSchema, dataSchema)).toArray
  override def createReaderFactory(): PartitionReaderFactory = BenoStreamFileReaderFactory
}

case class BenoStreamFilePartition(uri: String, file: String, fullReadSchema: StructType, dataSchema: StructType) extends InputPartition

object BenoStreamFileReaderFactory extends PartitionReaderFactory {
  override def createReader(partition: InputPartition): PartitionReader[InternalRow] = {
    val p = partition.asInstanceOf[BenoStreamFilePartition]
    new BenoStreamPartitionReader(p.file, p.fullReadSchema, p.dataSchema)
  }
}

/** Per-file reader: reads one segment via JNI and emits data columns + `_file`/`_pos`. */
class BenoStreamPartitionReader(file: String, fullReadSchema: StructType, dataSchema: StructType)
    extends PartitionReader[InternalRow] {

  @native def openSession(path: String): Long
  @native def readBatch(handle: Long, outArrayPtr: Long, outSchemaPtr: Long): Long
  @native def closeSession(handle: Long): Unit

  BenoStreamJNIBridge.getInstance()
  private val allocator = new RootAllocator()
  private val handle: Long = openSession(file)
  private var root: VectorSchemaRoot = null
  private var posInFile = 0L // running row offset within this file
  private var batchPos = -1
  private var current: InternalRow = null

  private def loadNext(): Boolean = {
    closeRoot()
    if (handle == 0L) return false
    val array = ArrowArray.allocateNew(allocator)
    val schema = ArrowSchema.allocateNew(allocator)
    try {
      if (readBatch(handle, array.memoryAddress(), schema.memoryAddress()) == 0L) {
        array.close(); schema.close(); return false
      }
      root = Data.importVectorSchemaRoot(allocator, array, schema, null)
      batchPos = 0
      root.getRowCount > 0
    } catch { case e: Throwable => array.close(); schema.close(); throw e }
  }

  private def closeRoot(): Unit = if (root != null) { root.close(); root = null }

  override def next(): Boolean = {
    if (root == null || batchPos >= root.getRowCount) if (!loadNext()) return false
    val values = new Array[Any](fullReadSchema.length)
    var i = 0
    while (i < fullReadSchema.length) {
      val f = fullReadSchema.fields(i)
      f.name match {
        case "_file" => values(i) = UTF8String.fromString(file)
        case "_pos" => values(i) = posInFile
        case _ =>
          val v = root.getVector(f.name)
          if (v == null || v.isNull(batchPos)) values(i) = null
          else {
            val ir = BenoStreamArrowUtils.toInternalRow(Seq(v), batchPos, StructType(Array(f)))
            values(i) = ir.get(0, f.dataType)
          }
      }
      i += 1
    }
    current = new org.apache.spark.sql.catalyst.expressions.GenericInternalRow(values)
    batchPos += 1
    posInFile += 1
    true
  }
  override def get(): InternalRow = current
  override def close(): Unit = { closeRoot(); if (handle != 0L) closeSession(handle); allocator.close() }
}

/** SQL construction + Spark-filter translation for the native scan. */
object BenoStreamSql {
  private def quote(id: String): String = "\"" + id.replace("\"", "\"\"") + "\""

  def isStringType(dt: org.apache.spark.sql.types.DataType): Boolean = dt match {
    case _: org.apache.spark.sql.types.StringType.type => true
    case _: org.apache.spark.sql.types.VarcharType => true
    case _: org.apache.spark.sql.types.CharType => true
    case _ => false
  }

  private def literal(v: Any): Option[String] = v match {
    case null => Some("NULL")
    case s: String => Some("'" + s.replace("'", "''") + "'")
    case b: java.lang.Boolean => Some(if (b.booleanValue()) "TRUE" else "FALSE")
    // `java.lang.Number` already covers Integer/Long/Double/Float/BigDecimal.
    case n: java.lang.Number => Some(n.toString)
    case _ => None
  }

  def filterToSql(f: Filter): Option[String] = f match {
    case EqualTo(attr, v) => literal(v).map(lit => s"${quote(attr)} = $lit")
    case GreaterThan(attr, v) => literal(v).map(lit => s"${quote(attr)} > $lit")
    case GreaterThanOrEqual(attr, v) => literal(v).map(lit => s"${quote(attr)} >= $lit")
    case LessThan(attr, v) => literal(v).map(lit => s"${quote(attr)} < $lit")
    case LessThanOrEqual(attr, v) => literal(v).map(lit => s"${quote(attr)} <= $lit")
    case IsNull(attr) => Some(s"${quote(attr)} IS NULL")
    case IsNotNull(attr) => Some(s"${quote(attr)} IS NOT NULL")
    case In(attr, vs) =>
      val lits = vs.map(literal)
      if (lits.forall(_.isDefined)) Some(s"${quote(attr)} IN (" + lits.flatten.mkString(", ") + ")") else None
    case StringStartsWith(attr, v) => Some(s"${quote(attr)} LIKE '" + v.replace("'", "''") + "%'")
    case And(l, r) => for (x <- filterToSql(l); y <- filterToSql(r)) yield s"($x AND $y)"
    case Or(l, r) => for (x <- filterToSql(l); y <- filterToSql(r)) yield s"($x OR $y)"
    case Not(c) => filterToSql(c).map(inner => s"NOT ($inner)")
    case _ => None
  }

  /**
   * Render a DSv2 sort order as engine SQL.
   *
   * `NamedReference` covers plain columns; `GeneralScalarExpression` covers the
   * function calls that make KNN work (`cosine_distance(embedding, …)`). Null
   * ordering is left to the engine, which only needs the direction to place
   * scored rows correctly.
   */
  def sortOrderToSql(order: SortOrder): Option[String] =
    expressionToSql(order.expression).map { expr =>
      // SortDirection is a Java enum in both Spark 3.5 and 4.x.
      val direction = if (order.direction == SortDirection.DESCENDING) "DESC" else "ASC"
      s"$expr $direction"
    }

  def expressionToSql(expr: Expression): Option[String] = expr match {
    case ref: NamedReference => Some(ref.fieldNames.map(quote).mkString("."))
    case lit: Literal[_] => literal(lit.value)
    case call: GeneralScalarExpression =>
      val rendered = call.children.map(expressionToSql)
      if (rendered.exists(_.isEmpty)) None
      else Some(s"${call.name}(${rendered.flatten.mkString(", ")})")
    case _ => None
  }

  def buildScanSql(
      full: StructType,
      required: StructType,
      filters: Array[Filter],
      orderBy: Option[String] = None,
      limit: Option[Int] = None
  ): String = {
    val cols =
      if (required.fields.length == full.fields.length && required.fieldNames.toSet == full.fieldNames.toSet) "*"
      else required.fields.map(f => quote(f.name)).mkString(", ")
    val preds = filters.flatMap(filterToSql)
    val where = if (preds.isEmpty) "" else " WHERE " + preds.mkString(" AND ")
    val order = orderBy.map(o => " ORDER BY " + o).getOrElse("")
    val lim = limit.map(k => " LIMIT " + k).getOrElse("")
    s"SELECT $cols FROM t$where$order$lim"
  }
}
