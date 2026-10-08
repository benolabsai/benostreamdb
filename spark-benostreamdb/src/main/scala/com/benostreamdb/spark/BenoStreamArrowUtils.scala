package com.benostreamdb.spark

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.core.`type`.TypeReference
import org.apache.arrow.vector.types.pojo.Field
import org.apache.arrow.vector.util.Text
import org.apache.arrow.vector.{BitVector, BigIntVector, DateDayVector, Float4Vector, Float8Vector, IntVector, SmallIntVector, TinyIntVector, ValueVector, VarCharVector}
import org.apache.arrow.vector.complex.{ListVector, StructVector}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.catalyst.util.{ArrayData, GenericArrayData}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

import scala.collection.JavaConverters._

/**
 * Helpers shared by the native Spark read/write paths: mapping the engine's
 * Arrow schema (JSON) to a Spark `StructType`, and decoding an Arrow batch into
 * catalyst `InternalRow`s. Deliberately Iceberg-free — the connector talks to
 * the native engine via JNI and the Arrow C Data Interface, exactly like the
 * Trino connector.
 */
object BenoStreamArrowUtils {

  private val mapper = new ObjectMapper()

  /** Map a Spark type to the engine's Arrow type-name (inverse of `arrowTypeToSparkType`). */
  def sparkTypeToEngineArrowName(dt: DataType): String = dt match {
    case IntegerType | ShortType | ByteType => "Int32"
    case LongType => "Int64"
    case FloatType => "Float32"
    case DoubleType => "Float64"
    case BooleanType => "Boolean"
    case DateType => "Date32"
    case _: StringType => "Utf8"
    case ArrayType(e, _) => s"List(${sparkTypeToEngineArrowName(e)})"
    case _ => "Utf8"
  }

  /** Serialize a Spark schema to the engine's `[{name,type,nullable}]` create-table JSON. */
  def structTypeToSchemaJson(schema: StructType): String = {
    val fields = schema.fields.map { f =>
      val m = new java.util.LinkedHashMap[String, AnyRef]()
      m.put("name", f.name)
      m.put("type", sparkTypeToEngineArrowName(f.dataType))
      m.put("nullable", java.lang.Boolean.valueOf(f.nullable))
      m
    }
    mapper.writeValueAsString(fields.toList.asJava)
  }

  /** Parse a JSON array of strings (listSchemas / listTables). */
  def parseStringArray(json: String): Seq[String] = {
    if (json == null || json.isEmpty) return Seq.empty
    val l = mapper.readValue(json, new TypeReference[java.util.List[String]] {})
    l.asScala.toSeq
  }

  /** Parse the engine's `[{name,type,nullable}]` schema JSON into a Spark schema. */
  def schemaJsonToStructType(json: String): StructType = {
    if (json == null || json.isEmpty || json == "[]") return new StructType()
    val fields: java.util.List[java.util.Map[String, AnyRef]] =
      mapper.readValue(json, new TypeReference[java.util.List[java.util.Map[String, AnyRef]]] {})
    val sparkFields = fields.asScala.map { f =>
      val name = String.valueOf(f.get("name"))
      val ty = String.valueOf(f.get("type"))
      val nullable = Option(f.get("nullable")).forall(_.toString.toBoolean)
      StructField(name, arrowTypeToSparkType(ty), nullable)
    }
    StructType(sparkFields.toSeq)
  }

  /** Map an Arrow type name (as emitted by `jni_util::format_datatype`) to Spark. */
  def arrowTypeToSparkType(t: String): DataType = {
    if (t.startsWith("List(") && t.endsWith(")")) {
      ArrayType(arrowTypeToSparkType(t.substring(5, t.length - 1)), containsNull = true)
    } else if (t.startsWith("Struct(") && t.endsWith(")")) {
      val inner = t.substring(7, t.length - 1)
      val sparkFields = inner.split(", ").map { part =>
        val idx = part.indexOf(": ")
        if (idx > 0) StructField(part.substring(0, idx), arrowTypeToSparkType(part.substring(idx + 2)), nullable = true)
        else StructField("_f", arrowTypeToSparkType(part), nullable = true)
      }
      StructType(sparkFields)
    } else t match {
      case "Int8" | "Int16" | "Int32" => IntegerType
      case "Int64" | "UInt8" | "UInt16" | "UInt32" | "UInt64" => LongType
      case "Float16" | "Float32" => FloatType
      case "Float64" => DoubleType
      case "Boolean" => BooleanType
      case "Date32" | "Date64" => DateType
      case "Utf8" | "LargeUtf8" | "Utf8View" => StringType
      case _ => StringType
    }
  }

  /**
   * Map an Arrow field type (from a query result) to a Spark type. Used by the
   * pass-through reader, whose schema comes from the engine's result rather
   * than from a table's stored schema.
   */
  def arrowFieldToSparkType(field: org.apache.arrow.vector.types.pojo.Field): DataType = {
    import org.apache.arrow.vector.types.pojo.ArrowType
    import org.apache.arrow.vector.types.FloatingPointPrecision
    field.getType match {
      case i: ArrowType.Int =>
        if (i.getBitWidth <= 32) IntegerType else LongType
      case f: ArrowType.FloatingPoint =>
        if (f.getPrecision == FloatingPointPrecision.SINGLE) FloatType else DoubleType
      case _: ArrowType.Bool => BooleanType
      case _: ArrowType.Utf8 => StringType
      case _: ArrowType.LargeUtf8 => StringType
      case _: ArrowType.Date => DateType
      case _: ArrowType.Timestamp => TimestampType
      case _: ArrowType.List =>
        // Children live on the Field, not on the ArrowType/FieldType.
        val children = field.getChildren
        if (children == null || children.isEmpty) ArrayType(StringType, containsNull = true)
        else ArrayType(arrowFieldToSparkType(children.get(0)), containsNull = true)
      case _: ArrowType.Struct =>
        val children = field.getChildren
        if (children == null) StructType(Seq.empty)
        else
          StructType(
            children.asScala
              .map(f => StructField(f.getName, arrowFieldToSparkType(f), nullable = true))
              .toArray
          )
      case _ => StringType
    }
  }

  /** Decode one Arrow row (across the root's vectors) into a catalyst row. */
  def toInternalRow(vectors: Seq[ValueVector], rowId: Int, schema: StructType): InternalRow = {
    val values = new Array[Any](schema.length)
    var i = 0
    while (i < schema.length) {
      val vec = vectors(i)
      values(i) = if (vec == null || vec.isNull(rowId)) null else readValue(vec, rowId, schema(i).dataType)
      i += 1
    }
    new GenericInternalRow(values)
  }

  private def readValue(vec: ValueVector, rowId: Int, dt: DataType): Any = dt match {
    case IntegerType =>
      vec match {
        case v: IntVector => v.get(rowId)
        case v: SmallIntVector => v.get(rowId).toInt
        case v: TinyIntVector => v.get(rowId).toInt
        case _ => vec.getObject(rowId).toString.toInt
      }
    case LongType =>
      vec match {
        case v: BigIntVector => v.get(rowId)
        case _ => vec.getObject(rowId).toString.toLong
      }
    case FloatType =>
      vec match {
        case v: Float4Vector => v.get(rowId)
        case _ => vec.getObject(rowId).toString.toFloat
      }
    case DoubleType =>
      vec match {
        case v: Float8Vector => v.get(rowId)
        case _ => vec.getObject(rowId).toString.toDouble
      }
    case BooleanType =>
      vec match {
        case v: BitVector => v.get(rowId) == 1
        case _ => java.lang.Boolean.parseBoolean(vec.getObject(rowId).toString)
      }
    case DateType =>
      vec match {
        case v: DateDayVector => v.get(rowId)
        case _ => vec.getObject(rowId).toString.toInt
      }
    case StringType =>
      vec.getObject(rowId) match {
        case t: Text => UTF8String.fromBytes(t.getBytes)
        case other => UTF8String.fromString(other.toString)
      }
    case ArrayType(elem, _) =>
      val list = vec.asInstanceOf[ListVector].getObject(rowId).asInstanceOf[java.util.List[Any]]
      new GenericArrayData(list.asScala.map(e => boxElement(e, elem)).toArray)
    case st: StructType =>
      val sv = vec.asInstanceOf[StructVector]
      val children = st.fields.map { f =>
        val child = sv.getChild(f.name)
        if (child == null || child.isNull(rowId)) null else readValue(child, rowId, f.dataType)
      }
      new GenericInternalRow(children.toArray)
    case _ => UTF8String.fromString(vec.getObject(rowId).toString)
  }

  private def boxElement(e: Any, dt: DataType): Any = (e, dt) match {
    case (null, _) => null
    case (x: java.lang.Number, IntegerType) => x.intValue()
    case (x: java.lang.Number, LongType) => x.longValue()
    case (x: java.lang.Number, FloatType) => x.floatValue()
    case (x: java.lang.Number, DoubleType) => x.doubleValue()
    case (x: Text, StringType) => UTF8String.fromBytes(x.getBytes)
    case (x, StringType) => UTF8String.fromString(x.toString)
    case (x, _) => x
  }
}
