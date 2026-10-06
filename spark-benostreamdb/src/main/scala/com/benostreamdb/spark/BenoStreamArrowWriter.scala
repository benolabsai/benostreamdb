package com.benostreamdb.spark

import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.ValueVector
import org.apache.arrow.vector.complex.ListVector
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema => ArrowSchema}
import org.apache.arrow.vector.types.{DateUnit, FloatingPointPrecision, TimeUnit}
import org.apache.arrow.vector.{BigIntVector, BitVector, DateDayVector, Float4Vector, Float8Vector, IntVector, VarCharVector, VectorSchemaRoot}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.types._

import scala.collection.JavaConverters._

/**
 * Converts a batch of catalyst `InternalRow`s into an Arrow `VectorSchemaRoot`
 * so it can be exported through the C Data Interface to the native engine
 * (`appendBatch`). Mirrors the reader in `BenoStreamArrowUtils`.
 */
object BenoStreamArrowWriter {

  def sparkToArrowField(f: StructField): Field = {
    val ft = new FieldType(f.nullable, sparkToArrowType(f.dataType), null)
    f.dataType match {
      case ArrayType(elem, _) =>
        new Field(f.name, ft, java.util.Collections.singletonList(sparkToArrowField(StructField("item", elem, true))))
      case _ => new Field(f.name, ft, java.util.Collections.emptyList[Field]())
    }
  }

  private def sparkToArrowType(dt: DataType): ArrowType = dt match {
    case IntegerType => new ArrowType.Int(32, true)
    case LongType => new ArrowType.Int(64, true)
    case ShortType => new ArrowType.Int(16, true)
    case ByteType => new ArrowType.Int(8, true)
    case FloatType => new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
    case DoubleType => new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
    case BooleanType => new ArrowType.Bool()
    case StringType | _: VarcharType | _: CharType => new ArrowType.Utf8()
    case DateType => new ArrowType.Date(DateUnit.DAY)
    case ArrayType(elem, _) => new ArrowType.List()
    case _ => new ArrowType.Utf8()
  }

  def structTypeToArrowSchema(schema: StructType): ArrowSchema =
    new ArrowSchema(schema.fields.map(sparkToArrowField).toList.asJava)

  /** Build a root and populate it from `rows` (which must match `schema`). */
  def toVectorSchemaRoot(rows: Seq[InternalRow], schema: StructType, allocator: BufferAllocator): VectorSchemaRoot = {
    val root = VectorSchemaRoot.create(structTypeToArrowSchema(schema), allocator)
    val vectors = schema.fields.map(f => root.getVector(f.name))
    rows.zipWithIndex.foreach { case (row, r) =>
      var i = 0
      while (i < schema.fields.length) {
        if (row.isNullAt(i)) vectors(i).setNull(r)
        else writeOne(vectors(i), schema.fields(i).dataType, row, i, r)
        i += 1
      }
    }
    root.setRowCount(rows.size)
    root
  }

  private def writeOne(v: ValueVector, dt: DataType, row: InternalRow, ordinal: Int, r: Int): Unit = dt match {
    case IntegerType => v.asInstanceOf[IntVector].setSafe(r, row.getInt(ordinal))
    case LongType => v.asInstanceOf[BigIntVector].setSafe(r, row.getLong(ordinal))
    case ShortType => v.asInstanceOf[IntVector].setSafe(r, row.getShort(ordinal).toInt)
    case FloatType => v.asInstanceOf[Float4Vector].setSafe(r, row.getFloat(ordinal))
    case DoubleType => v.asInstanceOf[Float8Vector].setSafe(r, row.getDouble(ordinal))
    case BooleanType => v.asInstanceOf[BitVector].setSafe(r, if (row.getBoolean(ordinal)) 1 else 0)
    case DateType => v.asInstanceOf[DateDayVector].setSafe(r, row.getInt(ordinal))
    case _: StringType =>
      val bytes = row.getUTF8String(ordinal).getBytes
      v.asInstanceOf[VarCharVector].setSafe(r, bytes, 0, bytes.length)
    case _: ArrayType =>
      throw new UnsupportedOperationException(
        "Native Arrow array/vector write is not implemented yet (tracked with the UDF milestone)")
    case _ =>
      val bytes = row.get(ordinal, dt).toString.getBytes("UTF-8")
      v.asInstanceOf[VarCharVector].setSafe(r, bytes, 0, bytes.length)
  }
}
