package com.benostreamdb.spark.functions

import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions.udf
import com.benostreamdb.spark.vector.VectorDistance

object BenoStreamFunctions {

  val vectorDistanceUdf = udf((v1: Seq[Double], v2: Seq[Double], metric: String) => {
    VectorDistance.vectorDistance(v1, v2, metric)
  })

  val cosineDistanceUdf = udf((v1: Seq[Double], v2: Seq[Double]) => {
    if (v1 == null || v2 == null) Double.NaN
    else VectorDistance.cosineDistanceDouble(v1.toArray, v2.toArray)
  })

  val l2DistanceUdf = udf((v1: Seq[Double], v2: Seq[Double]) => {
    if (v1 == null || v2 == null) Double.NaN
    else VectorDistance.l2DistanceDouble(v1.toArray, v2.toArray)
  })

  val dotProductUdf = udf((v1: Seq[Double], v2: Seq[Double]) => {
    if (v1 == null || v2 == null) Double.NaN
    else VectorDistance.dotProductDouble(v1.toArray, v2.toArray)
  })

  val sparseDotProductUdf = udf((i1: Seq[Long], v1: Seq[Float], i2: Seq[Long], v2: Seq[Float]) => {
    if (i1 == null || v1 == null || i2 == null || v2 == null) Float.NaN
    else VectorDistance.sparseDotProduct(i1.toArray, v1.toArray, i2.toArray, v2.toArray)
  })

  val hybridScoreUdf = udf((denseScore: Double, sparseScore: Double, alpha: Double) => {
    VectorDistance.hybridScore(denseScore, sparseScore, alpha)
  })

  val reciprocalRankFusionUdf = udf((ranks: Seq[Int], k: Int) => {
    VectorDistance.reciprocalRankFusion(ranks, k)
  })

  /**
   * Registers all BenoStreamDB vector search and distance UDFs with the SparkSession.
   */
  def register(spark: SparkSession): Unit = {
    spark.udf.register("vector_distance", vectorDistanceUdf)
    spark.udf.register("cosine_distance", cosineDistanceUdf)
    spark.udf.register("l2_distance", l2DistanceUdf)
    spark.udf.register("dot_product", dotProductUdf)
    spark.udf.register("sparse_dot_product", sparseDotProductUdf)
    spark.udf.register("hybrid_score", hybridScoreUdf)
    spark.udf.register("reciprocal_rank_fusion", reciprocalRankFusionUdf)
  }
}

package object functions {

  def vector_distance(v1: Column, v2: Column, metric: String): Column = {
    org.apache.spark.sql.functions.lit(metric).cast("string") match {
      case m => BenoStreamFunctions.vectorDistanceUdf(v1, v2, m)
    }
  }

  def cosine_distance(v1: Column, v2: Column): Column = {
    BenoStreamFunctions.cosineDistanceUdf(v1, v2)
  }

  def l2_distance(v1: Column, v2: Column): Column = {
    BenoStreamFunctions.l2DistanceUdf(v1, v2)
  }

  def dot_product(v1: Column, v2: Column): Column = {
    BenoStreamFunctions.dotProductUdf(v1, v2)
  }

  def sparse_dot_product(i1: Column, v1: Column, i2: Column, v2: Column): Column = {
    BenoStreamFunctions.sparseDotProductUdf(i1, v1, i2, v2)
  }

  def hybrid_score(denseScore: Column, sparseScore: Column, alpha: Column): Column = {
    BenoStreamFunctions.hybridScoreUdf(denseScore, sparseScore, alpha)
  }

  def reciprocal_rank_fusion(ranks: Column, k: Column): Column = {
    BenoStreamFunctions.reciprocalRankFusionUdf(ranks, k)
  }
}
