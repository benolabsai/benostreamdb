package com.benostreamdb.spark

import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.{array, col, lit}
import com.benostreamdb.spark.functions.BenoStreamFunctions

/**
 * Implicit DataFrame extensions providing fluent vector search operations.
 */
object implicits {

  implicit class BenoStreamDataFrameOps(val df: DataFrame) extends AnyVal {

    /**
     * Compute vector distance against a target query vector and append the result as a column.
     *
     * @param outputCol Name of the resulting distance column (e.g. "_distance")
     * @param vectorCol Column name containing the row vector (Array of Float or Double)
     * @param query Query vector as Array[Double]
     * @param metric Distance metric: "cosine", "l2", or "dot"
     * @return DataFrame with new distance column appended
     */
    def withVectorDistance(
        outputCol: String,
        vectorCol: String,
        query: Array[Double],
        metric: String = "cosine"
    ): DataFrame = {
      val queryCol = array(query.map(lit): _*)
      val distCol = BenoStreamFunctions.vectorDistanceUdf(col(vectorCol), queryCol, lit(metric))
      df.withColumn(outputCol, distCol)
    }

    /**
     * Performs top-K nearest neighbor search against a target query vector.
     *
     * @param vectorCol Column name containing the row vector
     * @param query Query vector as Array[Double]
     * @param k Number of nearest neighbors to retrieve
     * @param metric Distance metric ("cosine", "l2", "dot")
     * @param distanceCol Name of the generated distance column (default "_distance")
     * @return Top-K DataFrame ordered by distance ascending
     */
    def vectorSearch(
        vectorCol: String,
        query: Array[Double],
        k: Int,
        metric: String = "cosine",
        distanceCol: String = "_distance"
    ): DataFrame = {
      withVectorDistance(distanceCol, vectorCol, query, metric)
        .orderBy(col(distanceCol).asc)
        .limit(k)
    }
  }
}
