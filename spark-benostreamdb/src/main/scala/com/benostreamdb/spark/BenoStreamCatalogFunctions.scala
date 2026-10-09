package com.benostreamdb.spark

import com.benostreamdb.spark.vector.VectorDistance
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.util.{ArrayData, GenericArrayData}
import org.apache.spark.unsafe.types.UTF8String
import org.apache.spark.sql.connector.catalog.Identifier
import org.apache.spark.sql.connector.catalog.functions.{BoundFunction, ScalarFunction, UnboundFunction}
import org.apache.spark.sql.types.{ArrayType, BooleanType, DataTypes, DataType, DoubleType, FloatType, IntegerType, LongType, StringType, StructType}

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
    // distances (pgvector-compatible names; `dot_product` kept as an alias)
    "vector_distance", "cosine_distance", "l2_distance", "inner_product",
    "dot_product", "l1_distance", "hamming_distance", "jaccard_distance",
    "sparse_dot_product", "hybrid_score", "reciprocal_rank_fusion",
    // vector transforms
    "vector_add", "vector_sub", "vector_mul", "vector_concat", "vector_dims",
    "vector_norm", "l2_normalize", "binary_quantize", "subvector",
    "vector_to_binary",
    // lexical
    "bm25_score", "tf_idf",
    // json
    "json_extract_path", "json_extract_path_text", "json_contains",
    "json_exists", "json_typeof", "json_path_exists", "json_path_query"
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

  // ---- additional distance metrics ----

  class L1DistanceFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "l1_distance"
    override def inputTypes: Array[DataType] = Array(ArrayType(DoubleType), ArrayType(DoubleType))
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      val v1 = doubleArr(in, 0); val v2 = doubleArr(in, 1)
      if (v1 == null || v2 == null) dbl(Double.NaN)
      else dbl(v1.zip(v2).map { case (a, b) => math.abs(a - b) }.sum)
    }
  }

  class HammingDistanceFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "hamming_distance"
    override def inputTypes: Array[DataType] = Array(ArrayType(DoubleType), ArrayType(DoubleType))
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      val v1 = doubleArr(in, 0); val v2 = doubleArr(in, 1)
      if (v1 == null || v2 == null) dbl(Double.NaN)
      else dbl(v1.zip(v2).count { case (a, b) => a != b }.toDouble)
    }
  }

  class JaccardDistanceFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "jaccard_distance"
    override def inputTypes: Array[DataType] = Array(ArrayType(DoubleType), ArrayType(DoubleType))
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      val v1 = doubleArr(in, 0); val v2 = doubleArr(in, 1)
      if (v1 == null || v2 == null) dbl(Double.NaN)
      else {
        val s1 = v1.filter(_ != 0.0).toSet
        val s2 = v2.filter(_ != 0.0).toSet
        val union = (s1 ++ s2).size
        if (union == 0) dbl(0.0) else dbl(1.0 - (s1 & s2).size.toDouble / union)
      }
    }
  }

  // ---- vector transforms ----

  /**
   * Shared shape for the element-wise vector ops. Spark requires the
   * *concrete* class to declare `produceResult(InternalRow)` (an inherited
   * implementation is rejected with SCALAR_FUNCTION_NOT_FULLY_IMPLEMENTED), so
   * each subclass overrides it and delegates to `compute`.
   */
  /** Array-returning functions must produce catalyst `ArrayData`. */
  private def arrData(values: Array[Double]): ArrayData =
    new GenericArrayData(values.map(Double.box))

  private def intArrData(values: Array[Int]): ArrayData =
    new GenericArrayData(values.map(Int.box))

  abstract class BinaryVectorFn(fnName: String, op: (Array[Double], Array[Double]) => Array[Double])
      extends ScalarFunction[ArrayData] {
    override def name(): String = fnName
    override def inputTypes: Array[DataType] = Array(ArrayType(DoubleType), ArrayType(DoubleType))
    override def resultType: DataType = ArrayType(DoubleType)
    protected def compute(in: InternalRow): ArrayData = {
      val v1 = doubleArr(in, 0); val v2 = doubleArr(in, 1)
      if (v1 == null || v2 == null) null else arrData(op(v1, v2))
    }
  }

  class VectorAddFn extends BinaryVectorFn("vector_add", (a, b) => a.zip(b).map { case (x, y) => x + y }) {
    override def produceResult(in: InternalRow): ArrayData = compute(in)
  }
  class VectorSubFn extends BinaryVectorFn("vector_sub", (a, b) => a.zip(b).map { case (x, y) => x - y }) {
    override def produceResult(in: InternalRow): ArrayData = compute(in)
  }
  class VectorMulFn extends BinaryVectorFn("vector_mul", (a, b) => a.zip(b).map { case (x, y) => x * y }) {
    override def produceResult(in: InternalRow): ArrayData = compute(in)
  }
  class VectorConcatFn extends BinaryVectorFn("vector_concat", (a, b) => a ++ b) {
    override def produceResult(in: InternalRow): ArrayData = compute(in)
  }

  class VectorDimsFn extends ScalarFunction[java.lang.Integer] {
    override def name(): String = "vector_dims"
    override def inputTypes: Array[DataType] = Array(ArrayType(DoubleType))
    override def resultType: DataType = IntegerType
    override def produceResult(in: InternalRow): java.lang.Integer = {
      val v = doubleArr(in, 0)
      if (v == null) null else java.lang.Integer.valueOf(v.length)
    }
  }

  class VectorNormFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "vector_norm"
    override def inputTypes: Array[DataType] = Array(ArrayType(DoubleType))
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      val v = doubleArr(in, 0)
      if (v == null) dbl(Double.NaN) else dbl(math.sqrt(v.map(x => x * x).sum))
    }
  }

  class L2NormalizeFn extends ScalarFunction[ArrayData] {
    override def name(): String = "l2_normalize"
    override def inputTypes: Array[DataType] = Array(ArrayType(DoubleType))
    override def resultType: DataType = ArrayType(DoubleType)
    override def produceResult(in: InternalRow): ArrayData = {
      val v = doubleArr(in, 0)
      if (v == null) null
      else {
        val norm = math.sqrt(v.map(x => x * x).sum)
        arrData(if (norm > 0.0) v.map(_ / norm) else v)
      }
    }
  }

  class BinaryQuantizeFn extends ScalarFunction[ArrayData] {
    override def name(): String = "binary_quantize"
    override def inputTypes: Array[DataType] = Array(ArrayType(DoubleType))
    override def resultType: DataType = ArrayType(IntegerType)
    override def produceResult(in: InternalRow): ArrayData = {
      val v = doubleArr(in, 0)
      if (v == null) null else intArrData(v.map(x => if (x >= 0.0) 1 else 0))
    }
  }

  class SubvectorFn extends ScalarFunction[ArrayData] {
    override def name(): String = "subvector"
    override def inputTypes: Array[DataType] = Array(ArrayType(DoubleType), IntegerType, IntegerType)
    override def resultType: DataType = ArrayType(DoubleType)
    override def produceResult(in: InternalRow): ArrayData = {
      val v = doubleArr(in, 0)
      if (v == null || in.isNullAt(1) || in.isNullAt(2)) null
      else {
        val start = math.max(0, in.getInt(1))
        val count = math.max(0, in.getInt(2))
        arrData(v.slice(start, math.min(v.length, start + count)))
      }
    }
  }

  class VectorToBinaryFn extends ScalarFunction[ArrayData] {
    override def name(): String = "vector_to_binary"
    override def inputTypes: Array[DataType] = Array(ArrayType(DoubleType))
    override def resultType: DataType = ArrayType(IntegerType)
    override def produceResult(in: InternalRow): ArrayData = {
      val v = doubleArr(in, 0)
      if (v == null) null
      else {
        val packed = Array.fill((v.length + 7) / 8)(0)
        v.zipWithIndex.foreach { case (x, j) => if (x >= 0.0) packed(j / 8) |= 1 << (j % 8) }
        intArrData(packed)
      }
    }
  }

  // ---- lexical ----

  class Bm25ScoreFn extends ScalarFunction[java.lang.Double] {
    override def name(): String = "bm25_score"
    override def inputTypes: Array[DataType] = Array(StringType, StringType)
    override def resultType: DataType = DoubleType
    override def produceResult(in: InternalRow): java.lang.Double = {
      if (in.isNullAt(0) || in.isNullAt(1)) dbl(Double.NaN)
      else {
        val text = in.getString(0).toString
        val query = in.getString(1).toString
        val terms = query.toLowerCase.split("\\s+").filter(_.nonEmpty)
        val words = text.toLowerCase.split("\\s+").filter(_.nonEmpty)
        if (terms.isEmpty || words.isEmpty) dbl(0.0)
        else {
          val k1 = 1.2; val b = 0.75
          val avgLen = words.length.toDouble
          val score = terms.map { t =>
            val tf = words.count(_ == t).toDouble
            if (tf == 0.0) 0.0
            else tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * words.length / avgLen))
          }.sum
          dbl(score)
        }
      }
    }
  }

  class TfIdfFn extends ScalarFunction[ArrayData] {
    override def name(): String = "tf_idf"
    override def inputTypes: Array[DataType] = Array(StringType)
    override def resultType: DataType = ArrayType(DoubleType)
    override def produceResult(in: InternalRow): ArrayData = {
      if (in.isNullAt(0)) null
      else {
        val words = in.getString(0).toString.toLowerCase.split("\\s+").filter(_.nonEmpty)
        if (words.isEmpty) arrData(Array.empty[Double])
        else {
          // NOTE: avoid `view.mapValues` here — `IterableView` has no
          // `mapValues` on Scala 2.12 (the Spark 3.5 line). A plain `map`
          // builds the same `Map[String, Double]` on both 2.12 and 2.13.
          val counts = words.groupBy(identity).map { case (w, ws) => (w, ws.length.toDouble) }
          arrData(words.distinct.map(w => counts(w) / words.length))
        }
      }
    }
  }

  // ---- json ----

  /** String-returning functions must produce catalyst `UTF8String`. */
  private def utf8(s: String): UTF8String = if (s == null) null else UTF8String.fromString(s)

  class JsonExtractPathFn extends ScalarFunction[UTF8String] {
    override def name(): String = "json_extract_path"
    override def inputTypes: Array[DataType] = Array(StringType, StringType)
    override def resultType: DataType = StringType
    override def produceResult(in: InternalRow): UTF8String = {
      if (in.isNullAt(0) || in.isNullAt(1)) null
      else utf8(BenoStreamJson.extractPath(in.getString(0).toString, in.getString(1).toString))
    }
  }

  class JsonExtractPathTextFn extends ScalarFunction[UTF8String] {
    override def name(): String = "json_extract_path_text"
    override def inputTypes: Array[DataType] = Array(StringType, StringType)
    override def resultType: DataType = StringType
    override def produceResult(in: InternalRow): UTF8String = {
      if (in.isNullAt(0) || in.isNullAt(1)) null
      else utf8(BenoStreamJson.extractPathText(in.getString(0).toString, in.getString(1).toString))
    }
  }

  class JsonContainsFn extends ScalarFunction[java.lang.Boolean] {
    override def name(): String = "json_contains"
    override def inputTypes: Array[DataType] = Array(StringType, StringType)
    override def resultType: DataType = BooleanType
    override def produceResult(in: InternalRow): java.lang.Boolean = {
      if (in.isNullAt(0) || in.isNullAt(1)) null
      else java.lang.Boolean.valueOf(
        BenoStreamJson.contains(in.getString(0).toString, in.getString(1).toString)
      )
    }
  }

  class JsonExistsFn extends ScalarFunction[java.lang.Boolean] {
    override def name(): String = "json_exists"
    override def inputTypes: Array[DataType] = Array(StringType, StringType)
    override def resultType: DataType = BooleanType
    override def produceResult(in: InternalRow): java.lang.Boolean = {
      if (in.isNullAt(0) || in.isNullAt(1)) null
      else java.lang.Boolean.valueOf(
        BenoStreamJson.exists(in.getString(0).toString, in.getString(1).toString)
      )
    }
  }

  class JsonTypeofFn extends ScalarFunction[UTF8String] {
    override def name(): String = "json_typeof"
    override def inputTypes: Array[DataType] = Array(StringType)
    override def resultType: DataType = StringType
    override def produceResult(in: InternalRow): UTF8String = {
      if (in.isNullAt(0)) null else utf8(BenoStreamJson.typeof(in.getString(0).toString))
    }
  }

  class JsonPathExistsFn extends ScalarFunction[java.lang.Boolean] {
    override def name(): String = "json_path_exists"
    override def inputTypes: Array[DataType] = Array(StringType, StringType)
    override def resultType: DataType = BooleanType
    override def produceResult(in: InternalRow): java.lang.Boolean = {
      if (in.isNullAt(0) || in.isNullAt(1)) null
      else java.lang.Boolean.valueOf(
        BenoStreamJson.pathExists(in.getString(0).toString, in.getString(1).toString)
      )
    }
  }

  class JsonPathQueryFn extends ScalarFunction[UTF8String] {
    override def name(): String = "json_path_query"
    override def inputTypes: Array[DataType] = Array(StringType, StringType)
    override def resultType: DataType = StringType
    override def produceResult(in: InternalRow): UTF8String = {
      if (in.isNullAt(0) || in.isNullAt(1)) null
      else utf8(BenoStreamJson.pathQuery(in.getString(0).toString, in.getString(1).toString))
    }
  }

  private def build(name: String): BoundFunction = name.toLowerCase match {
    case "vector_distance" => new VectorDistanceFn
    case "cosine_distance" => new CosineDistanceFn
    case "l2_distance" => new L2DistanceFn
    case "dot_product" | "inner_product" => new DotProductFn
    case "l1_distance" => new L1DistanceFn
    case "hamming_distance" => new HammingDistanceFn
    case "jaccard_distance" => new JaccardDistanceFn
    case "sparse_dot_product" => new SparseDotProductFn
    case "hybrid_score" => new HybridScoreFn
    case "reciprocal_rank_fusion" => new ReciprocalRankFusionFn
    case "vector_add" => new VectorAddFn
    case "vector_sub" => new VectorSubFn
    case "vector_mul" => new VectorMulFn
    case "vector_concat" => new VectorConcatFn
    case "vector_dims" => new VectorDimsFn
    case "vector_norm" => new VectorNormFn
    case "l2_normalize" => new L2NormalizeFn
    case "binary_quantize" => new BinaryQuantizeFn
    case "subvector" => new SubvectorFn
    case "vector_to_binary" => new VectorToBinaryFn
    case "bm25_score" => new Bm25ScoreFn
    case "tf_idf" => new TfIdfFn
    case "json_extract_path" => new JsonExtractPathFn
    case "json_extract_path_text" => new JsonExtractPathTextFn
    case "json_contains" => new JsonContainsFn
    case "json_exists" => new JsonExistsFn
    case "json_typeof" => new JsonTypeofFn
    case "json_path_exists" => new JsonPathExistsFn
    case "json_path_query" => new JsonPathQueryFn
    case other => throw new IllegalArgumentException(s"Unknown BenoStreamDB function: $other")
  }

  private def describe(name: String): String = name.toLowerCase match {
    case "vector_distance" => "Distance between two vectors using the given metric ('cosine', 'l2', 'dot')"
    case "cosine_distance" => "Cosine distance between two vectors (0.0 = identical, 1.0 = orthogonal)"
    case "l2_distance" => "Euclidean (L2) distance between two vectors"
    case "dot_product" | "inner_product" =>
      "Inner product of two vectors (higher is more similar)"
    case "l1_distance" => "Manhattan (L1) distance between two vectors"
    case "hamming_distance" => "Number of differing positions between two vectors"
    case "jaccard_distance" => "Jaccard distance between the non-zero supports of two vectors"
    case "sparse_dot_product" => "Dot product between sparse vectors (indices, values)"
    case "hybrid_score" => "alpha * dense + (1 - alpha) * sparse score blending"
    case "reciprocal_rank_fusion" => "RRF score sum(1 / (k + rank_i)) over a list of ranks"
    case "vector_add" => "Element-wise vector addition"
    case "vector_sub" => "Element-wise vector subtraction"
    case "vector_mul" => "Element-wise vector multiplication"
    case "vector_concat" => "Concatenate two vectors"
    case "vector_dims" => "Dimensionality of a vector"
    case "vector_norm" => "Euclidean (L2) norm of a vector"
    case "l2_normalize" => "Unit-normalize a vector (L2)"
    case "binary_quantize" => "Sign-quantize a vector to 0/1"
    case "subvector" => "Slice a vector: (vector, start, length)"
    case "vector_to_binary" => "Bit-pack a vector's sign mask into bytes"
    case "bm25_score" => "BM25 relevance of a document text against a query"
    case "tf_idf" => "Per-term TF-IDF weights for a document text"
    case "json_extract_path" => "Extract a JSON value at a path (returns json)"
    case "json_extract_path_text" => "Extract a JSON value at a path (returns text)"
    case "json_contains" => "Recursive JSON containment (@>)"
    case "json_exists" => "Whether a top-level JSON key/element exists"
    case "json_typeof" => "JSON value type name"
    case "json_path_exists" => "Whether a jsonpath matches"
    case "json_path_query" => "Evaluate a jsonpath and return the matches"
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
