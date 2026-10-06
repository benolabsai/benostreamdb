package com.benostreamdb.spark

import com.benostreamdb.spark.vector.VectorDistance
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.catalog.Identifier
import org.apache.spark.sql.connector.catalog.functions.{BoundFunction, ScalarFunction, UnboundFunction}
import org.apache.spark.sql.types.{ArrayType, DataTypes, DataType, DoubleType, FloatType, IntegerType, LongType, StringType, StructType}

/**
 * BenoStreamDB vector functions exposed as Spark DSv2 catalog functions
 * (`FunctionCatalog`), so they resolve natively from SQL, DataFrames, and
 * PySpark with no session registration:
 *
 * {{{
 *   SELECT benostream.system.cosine_distance(embedding, array(1.0, 0.0, 0.0))
 *   FROM benostream.default.docs
 * }}}
 *
 * Implementations override `produceResult(InternalRow)` directly — the contract
 * required by Spark 4.x and accepted by Spark 3.4/3.5 — reading arguments
 * positionally from `inputTypes`.
 */
object BenoStreamCatalogFunctions {

  val names: Seq[String] = Seq(
    "vector_distance", "cosine_distance", "l2_distance", "dot_product",
    "sparse_dot_product", "hybrid_score", "reciprocal_rank_fusion"
  )

  private def dbl(v: Double): java.lang.Double = java.lang.Double.valueOf(v)
  private def flt(v: Float): java.lang.Float = java.lang.Float.valueOf(v)

  private def doubleArr(in: InternalRow, i: Int): Array[Double] =
    if (in.isNullAt(i)) null else in.getArray(i).toDoubleArray()
  private def floatArr(in: InternalRow, i: Int): Array[Float] =
    if (in.isNullAt(i)) null else in.getArray(i).toFloatArray()
  private def longArr(in: InternalRow, i: Int): Array[Long] =
    if (in.isNullAt(i)) null else in.getArray(i).toLongArray()
  private def intArr(in: InternalRow, i: Int): Array[Int] =
    if (in.isNullAt(i)) null else in.getArray(i).toIntArray()

  class VectorDistanceFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "vector_distance"
    override def inputTypes: Array[DataType] =
      Array(ArrayType(DataTypes.DoubleType), ArrayType(DataTypes.DoubleType), StringType)
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      val v1 = doubleArr(in, 0)
      val v2 = doubleArr(in, 1)
      if (v1 == null || v2 == null) dbl(Double.NaN)
      else {
        val metric = if (in.isNullAt(2)) "cosine" else in.getString(2).toString
        dbl(VectorDistance.vectorDistance(v1.toSeq, v2.toSeq, metric))
      }
    }
  }

  class CosineDistanceFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "cosine_distance"
    override def inputTypes: Array[DataType] =
      Array(ArrayType(DataTypes.DoubleType), ArrayType(DataTypes.DoubleType))
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      val v1 = doubleArr(in, 0)
      val v2 = doubleArr(in, 1)
      if (v1 == null || v2 == null) dbl(Double.NaN)
      else dbl(VectorDistance.cosineDistanceDouble(v1, v2))
    }
  }

  class L2DistanceFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "l2_distance"
    override def inputTypes: Array[DataType] =
      Array(ArrayType(DataTypes.DoubleType), ArrayType(DataTypes.DoubleType))
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      val v1 = doubleArr(in, 0)
      val v2 = doubleArr(in, 1)
      if (v1 == null || v2 == null) dbl(Double.NaN)
      else dbl(VectorDistance.l2DistanceDouble(v1, v2))
    }
  }

  class DotProductFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "dot_product"
    override def inputTypes: Array[DataType] =
      Array(ArrayType(DataTypes.DoubleType), ArrayType(DataTypes.DoubleType))
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      val v1 = doubleArr(in, 0)
      val v2 = doubleArr(in, 1)
      if (v1 == null || v2 == null) dbl(Double.NaN)
      else dbl(VectorDistance.dotProductDouble(v1, v2))
    }
  }

  class SparseDotProductFn extends ScalarFunction[java.lang.Float] {
    override def name(): String = "sparse_dot_product"
    override def inputTypes: Array[DataType] = Array(
      ArrayType(LongType), ArrayType(FloatType), ArrayType(LongType), ArrayType(FloatType)
    )
    override def resultType: DataType = FloatType
    override def produceResult(in: InternalRow): java.lang.Float = {
      val i1 = longArr(in, 0)
      val v1 = floatArr(in, 1)
      val i2 = longArr(in, 2)
      val v2 = floatArr(in, 3)
      if (i1 == null || v1 == null || i2 == null || v2 == null) flt(Float.NaN)
      else flt(VectorDistance.sparseDotProduct(i1, v1, i2, v2))
    }
  }

  class HybridScoreFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "hybrid_score"
    override def inputTypes: Array[DataType] = Array(DoubleType, DoubleType, DoubleType)
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      if (in.isNullAt(0) || in.isNullAt(1) || in.isNullAt(2)) dbl(Double.NaN)
      else dbl(VectorDistance.hybridScore(in.getDouble(0), in.getDouble(1), in.getDouble(2)))
    }
  }

  class ReciprocalRankFusionFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "reciprocal_rank_fusion"
    override def inputTypes: Array[DataType] = Array(ArrayType(IntegerType), IntegerType)
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      val ranks = intArr(in, 0)
      if (ranks == null || in.isNullAt(1)) dbl(Double.NaN)
      else dbl(VectorDistance.reciprocalRankFusion(ranks.toSeq, in.getInt(1)))
    }
  }

  private def build(name: String): BoundFunction = name.toLowerCase match {
    case "vector_distance" => new VectorDistanceFn
    case "cosine_distance" => new CosineDistanceFn
    case "l2_distance" => new L2DistanceFn
    case "dot_product" => new DotProductFn
    case "sparse_dot_product" => new SparseDotProductFn
    case "hybrid_score" => new HybridScoreFn
    case "reciprocal_rank_fusion" => new ReciprocalRankFusionFn
    case other => throw new IllegalArgumentException(s"Unknown BenoStreamDB function: $other")
  }

  private def describe(name: String): String = name.toLowerCase match {
    case "vector_distance" => "Distance between two vectors using the given metric ('cosine', 'l2', 'dot')"
    case "cosine_distance" => "Cosine distance between two vectors (0.0 = identical, 1.0 = orthogonal)"
    case "l2_distance" => "Euclidean (L2) distance between two vectors"
    case "dot_product" => "Dot product of two vectors (higher is more similar)"
    case "sparse_dot_product" => "Dot product between sparse vectors (indices, values)"
    case "hybrid_score" => "alpha * dense + (1 - alpha) * sparse score blending"
    case "reciprocal_rank_fusion" => "RRF score sum(1 / (k + rank_i)) over a list of ranks"
    case other => throw new IllegalArgumentException(s"Unknown BenoStreamDB function: $other")
  }

  def unbound(ident: Identifier): UnboundFunction = {
    val fnName = ident.name()
    new UnboundFunction {
      override def name(): String = fnName
      override def description(): String = describe(fnName)
      override def bind(inputType: StructType): BoundFunction = build(fnName)
    }
  }
}
