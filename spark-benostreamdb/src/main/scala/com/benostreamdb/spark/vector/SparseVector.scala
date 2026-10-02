package com.benostreamdb.spark.vector

import org.apache.spark.sql.types.{ArrayType, DataTypes, FloatType, LongType, StructField, StructType}

/**
 * Representation of a sparse vector (e.g. SPLADE or BM25 term weights).
 * Maps 1-to-1 with BenoStreamDB's Arrow Struct<indices: List<Int64>, values: List<Float32>>.
 */
case class SparseVector(indices: Array[Long], values: Array[Float]) {
  require(
    indices != null && values != null && indices.length == values.length,
    "indices and values must be non-null and have identical lengths"
  )

  def dot(other: SparseVector): Float = {
    VectorDistance.sparseDotProduct(this.indices, this.values, other.indices, other.values)
  }
}

object SparseVector {

  /**
   * Official Spark StructType representing a BenoStreamDB sparse vector column.
   */
  val schema: StructType = StructType(
    Seq(
      StructField("indices", ArrayType(LongType, containsNull = false), nullable = false),
      StructField("values", ArrayType(FloatType, containsNull = false), nullable = false)
    )
  )

  def apply(pairs: Seq[(Long, Float)]): SparseVector = {
    val sorted = pairs.sortBy(_._1)
    val indices = sorted.map(_._1).toArray
    val values = sorted.map(_._2).toArray
    new SparseVector(indices, values)
  }
}
