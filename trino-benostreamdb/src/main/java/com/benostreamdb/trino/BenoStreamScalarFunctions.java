package com.benostreamdb.trino;

import io.airlift.slice.Slice;
import io.trino.spi.block.Block;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.DoubleType;
import io.trino.spi.type.IntegerType;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Distance and fusion functions, matching the Spark connector's seven UDFs name
 * for name and formula for formula.
 *
 * <p>Contributed the way Trino's developer guide specifies: an annotated class
 * returned from {@code Plugin.getFunctions()} as a {@code Set<Class<?>>}. Trino
 * derives signatures from the annotations, so there is no hand-built metadata
 * to keep in sync.</p>
 *
 * <p>Two details are easy to get wrong and both were established empirically:</p>
 * <ul>
 *   <li>An {@code array(...)} argument is passed as a {@link Block}, not a
 *       {@code List}; declaring {@code List} fails at specialization with
 *       "java types do not match".</li>
 *   <li>Plugin functions may not shadow built-ins. Trino 484 already ships
 *       {@code cosine_distance(array(double), array(double))} and
 *       {@code dot_product(array(double), array(double))}, so those two are not
 *       registered here and are provided by Trino.</li>
 * </ul>
 *
 * <p>Arithmetic is done in {@code double} and mirrors {@code VectorDistance} in
 * the Spark connector, including its quirks: cosine distance is floored at 0,
 * and the {@code dot} metric is negated because lower is closer for every
 * metric in this family.</p>
 */
public final class BenoStreamScalarFunctions {

    private BenoStreamScalarFunctions() {
    }

    // -------------------------------------------------------------------------
    // SQL functions
    // -------------------------------------------------------------------------

    @ScalarFunction("l2_distance")
    @Description("Euclidean distance")
    @SqlType("double")
    public static double l2DistanceSql(
            @SqlType("array(double)") Block left,
            @SqlType("array(double)") Block right) {
        return l2Distance(doubles(left), doubles(right));
    }

    @ScalarFunction("vector_distance")
    @Description("Distance by metric name: cosine, l2/euclidean, dot")
    @SqlType("double")
    public static double vectorDistance(
            @SqlType("array(double)") Block left,
            @SqlType("array(double)") Block right,
            @SqlType("varchar") Slice metric) {
        List<Double> leftValues = doubles(left);
        List<Double> rightValues = doubles(right);
        String name = metric == null
                ? "cosine"
                : metric.toStringUtf8().trim().toLowerCase(Locale.ENGLISH);
        return switch (name) {
            case "cosine", "cos" -> cosineDistance(leftValues, rightValues);
            case "l2", "euclidean" -> l2Distance(leftValues, rightValues);
            // Negated: the engine treats every metric in this family as
            // "lower is closer", so the inner product is inverted.
            case "dot", "dot_product", "inner_product" -> -dotProduct(leftValues, rightValues);
            default -> throw new IllegalArgumentException("Unsupported distance metric: " + name);
        };
    }

    @ScalarFunction("sparse_dot_product")
    @Description("Dot product of two sparse vectors given as parallel index/value arrays")
    @SqlType("double")
    public static double sparseDotProduct(
            @SqlType("array(bigint)") Block indicesLeft,
            @SqlType("array(double)") Block valuesLeft,
            @SqlType("array(bigint)") Block indicesRight,
            @SqlType("array(double)") Block valuesRight) {
        List<Long> leftIndices = longs(indicesLeft);
        List<Double> leftValues = doubles(valuesLeft);
        List<Long> rightIndices = longs(indicesRight);
        List<Double> rightValues = doubles(valuesRight);

        int left = 0;
        int right = 0;
        int leftLength = Math.min(leftIndices.size(), leftValues.size());
        int rightLength = Math.min(rightIndices.size(), rightValues.size());
        double dot = 0;
        while (left < leftLength && right < rightLength) {
            long leftIndex = leftIndices.get(left);
            long rightIndex = rightIndices.get(right);
            if (leftIndex == rightIndex) {
                dot += leftValues.get(left) * rightValues.get(right);
                left++;
                right++;
            } else if (leftIndex < rightIndex) {
                left++;
            } else {
                right++;
            }
        }
        return dot;
    }

    @ScalarFunction("hybrid_score")
    @Description("Weighted blend of dense and sparse scores; alpha is clamped to [0,1]")
    @SqlType("double")
    public static double hybridScore(
            @SqlType("double") double denseScore,
            @SqlType("double") double sparseScore,
            @SqlType("double") double alpha) {
        double weight = Math.max(0.0, Math.min(1.0, alpha));
        return weight * denseScore + (1.0 - weight) * sparseScore;
    }

    @ScalarFunction("reciprocal_rank_fusion")
    @Description("Reciprocal rank fusion; sums 1/(k + rank) over positive ranks")
    @SqlType("double")
    public static double reciprocalRankFusion(
            @SqlType("array(integer)") Block ranks,
            @SqlType("integer") long k) {
        double score = 0;
        for (int i = 0; i < ranks.getPositionCount(); i++) {
            int rank = IntegerType.INTEGER.getInt(ranks, i);
            if (rank > 0) {
                score += 1.0 / (k + rank);
            }
        }
        return score;
    }

    // -------------------------------------------------------------------------
    // Shared arithmetic
    // -------------------------------------------------------------------------

    public static double l2Distance(List<Double> left, List<Double> right) {
        int length = Math.min(left.size(), right.size());
        double sum = 0;
        for (int i = 0; i < length; i++) {
            double delta = left.get(i) - right.get(i);
            sum += delta * delta;
        }
        return Math.sqrt(sum);
    }

    /**
     * Not registered: Trino 484 ships {@code cosine_distance} with this exact
     * signature, and a plugin function may not shadow a built-in.
     */
    public static double cosineDistance(List<Double> left, List<Double> right) {
        int length = Math.min(left.size(), right.size());
        double dot = 0;
        double leftNorm = 0;
        double rightNorm = 0;
        for (int i = 0; i < length; i++) {
            double a = left.get(i);
            double b = right.get(i);
            dot += a * b;
            leftNorm += a * a;
            rightNorm += b * b;
        }
        double denominator = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
        if (denominator == 0.0) {
            return 0.0;
        }
        // Floors at 0; it does not cap at 1, matching Spark.
        return Math.max(0.0, 1.0 - (dot / denominator));
    }

    /** Not registered: Trino 484 ships {@code dot_product} with this signature. */
    public static double dotProduct(List<Double> left, List<Double> right) {
        int length = Math.min(left.size(), right.size());
        double dot = 0;
        for (int i = 0; i < length; i++) {
            dot += left.get(i) * right.get(i);
        }
        return dot;
    }

    private static List<Double> doubles(Block block) {
        List<Double> values = new ArrayList<>(block.getPositionCount());
        for (int i = 0; i < block.getPositionCount(); i++) {
            values.add(DoubleType.DOUBLE.getDouble(block, i));
        }
        return values;
    }

    private static List<Long> longs(Block block) {
        List<Long> values = new ArrayList<>(block.getPositionCount());
        for (int i = 0; i < block.getPositionCount(); i++) {
            values.add(BigintType.BIGINT.getLong(block, i));
        }
        return values;
    }
}
