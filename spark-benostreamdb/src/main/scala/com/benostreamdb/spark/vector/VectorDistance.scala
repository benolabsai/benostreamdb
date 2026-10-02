package com.benostreamdb.spark.vector

import scala.math.sqrt

/**
 * High-performance vector distance and similarity functions for Spark SQL and DataFrames.
 */
object VectorDistance {

  def cosineDistance(v1: Array[Float], v2: Array[Float]): Float = {
    if (v1 == null || v2 == null || v1.length == 0 || v1.length != v2.length) return Float.NaN
    var dot = 0.0f
    var norm1 = 0.0f
    var norm2 = 0.0f
    var i = 0
    val len = v1.length
    while (i < len) {
      val a = v1(i)
      val b = v2(i)
      dot += a * b
      norm1 += a * a
      norm2 += b * b
      i += 1
    }
    val denom = (sqrt(norm1.toDouble) * sqrt(norm2.toDouble)).toFloat
    if (denom == 0.0f) 0.0f else (1.0f - (dot / denom)).max(0.0f)
  }

  def cosineDistanceDouble(v1: Array[Double], v2: Array[Double]): Double = {
    if (v1 == null || v2 == null || v1.length == 0 || v1.length != v2.length) return Double.NaN
    var dot = 0.0
    var norm1 = 0.0
    var norm2 = 0.0
    var i = 0
    val len = v1.length
    while (i < len) {
      val a = v1(i)
      val b = v2(i)
      dot += a * b
      norm1 += a * a
      norm2 += b * b
      i += 1
    }
    val denom = sqrt(norm1) * sqrt(norm2)
    if (denom == 0.0) 0.0 else (1.0 - (dot / denom)).max(0.0)
  }

  def l2Distance(v1: Array[Float], v2: Array[Float]): Float = {
    if (v1 == null || v2 == null || v1.length == 0 || v1.length != v2.length) return Float.NaN
    var sum = 0.0f
    var i = 0
    val len = v1.length
    while (i < len) {
      val diff = v1(i) - v2(i)
      sum += diff * diff
      i += 1
    }
    sqrt(sum.toDouble).toFloat
  }

  def l2DistanceDouble(v1: Array[Double], v2: Array[Double]): Double = {
    if (v1 == null || v2 == null || v1.length == 0 || v1.length != v2.length) return Double.NaN
    var sum = 0.0
    var i = 0
    val len = v1.length
    while (i < len) {
      val diff = v1(i) - v2(i)
      sum += diff * diff
      i += 1
    }
    sqrt(sum)
  }

  def dotProduct(v1: Array[Float], v2: Array[Float]): Float = {
    if (v1 == null || v2 == null || v1.length == 0 || v1.length != v2.length) return Float.NaN
    var dot = 0.0f
    var i = 0
    val len = v1.length
    while (i < len) {
      dot += v1(i) * v2(i)
      i += 1
    }
    dot
  }

  def dotProductDouble(v1: Array[Double], v2: Array[Double]): Double = {
    if (v1 == null || v2 == null || v1.length == 0 || v1.length != v2.length) return Double.NaN
    var dot = 0.0
    var i = 0
    val len = v1.length
    while (i < len) {
      dot += v1(i) * v2(i)
      i += 1
    }
    dot
  }

  def hammingDistance(v1: Array[Byte], v2: Array[Byte]): Long = {
    if (v1 == null || v2 == null || v1.length != v2.length) return -1L
    var dist = 0L
    var i = 0
    val len = v1.length
    while (i < len) {
      val xor = (v1(i) ^ v2(i)) & 0xFF
      dist += java.lang.Integer.bitCount(xor)
      i += 1
    }
    dist
  }

  def sparseDotProduct(
      indices1: Array[Long],
      values1: Array[Float],
      indices2: Array[Long],
      values2: Array[Float]
  ): Float = {
    if (indices1 == null || values1 == null || indices2 == null || values2 == null) return Float.NaN
    var i = 0
    var j = 0
    var dot = 0.0f
    val len1 = indices1.length.min(values1.length)
    val len2 = indices2.length.min(values2.length)
    while (i < len1 && j < len2) {
      val idx1 = indices1(i)
      val idx2 = indices2(j)
      if (idx1 == idx2) {
        dot += values1(i) * values2(j)
        i += 1
        j += 1
      } else if (idx1 < idx2) {
        i += 1
      } else {
        j += 1
      }
    }
    dot
  }

  def vectorDistance(v1: Seq[Double], v2: Seq[Double], metric: String): Double = {
    if (v1 == null || v2 == null) return Double.NaN
    val m = if (metric == null) "cosine" else metric.toLowerCase
    val a1 = v1.toArray
    val a2 = v2.toArray
    m match {
      case "cosine" | "cos" => cosineDistanceDouble(a1, a2)
      case "l2" | "euclidean" => l2DistanceDouble(a1, a2)
      case "dot" | "dot_product" | "inner_product" => -dotProductDouble(a1, a2) // lower is closer
      case _ => throw new IllegalArgumentException(s"Unsupported distance metric: $metric")
    }
  }

  def hybridScore(denseScore: Double, sparseScore: Double, alpha: Double): Double = {
    val a = alpha.max(0.0).min(1.0)
    a * denseScore + (1.0 - a) * sparseScore
  }

  def reciprocalRankFusion(ranks: Seq[Int], k: Int = 60): Double = {
    if (ranks == null || ranks.isEmpty) 0.0
    else {
      var rrf = 0.0
      for (r <- ranks) {
        if (r > 0) {
          rrf += 1.0 / (k + r)
        }
      }
      rrf
    }
  }
}
