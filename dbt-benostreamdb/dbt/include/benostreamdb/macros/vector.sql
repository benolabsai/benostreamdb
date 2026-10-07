{% macro type_vector(dimensions=none) -%}
  {{ return(adapter.dispatch('type_vector', 'dbt')(dimensions)) }}
{%- endmacro %}

{% macro default__type_vector(dimensions=none) -%}
  {{ exceptions.raise_compiler_error("type_vector is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__type_vector(dimensions=none) -%}
  {%- if dimensions is not none -%}
    FLOAT[{{ dimensions }}]
  {%- else -%}
    FLOAT[]
  {%- endif -%}
{%- endmacro %}


{% macro type_sparsevec(dimensions=none) -%}
  {{ return(adapter.dispatch('type_sparsevec', 'dbt')(dimensions)) }}
{%- endmacro %}

{% macro default__type_sparsevec(dimensions=none) -%}
  {{ exceptions.raise_compiler_error("type_sparsevec is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__type_sparsevec(dimensions=none) -%}
  {%- if dimensions is not none -%}
    FLOAT[{{ dimensions }}]
  {%- else -%}
    FLOAT[]
  {%- endif -%}
{%- endmacro %}


{% macro vector_distance(column, query_vec, metric='l2') -%}
  {{ return(adapter.dispatch('vector_distance', 'dbt')(column, query_vec, metric)) }}
{%- endmacro %}

{% macro default__vector_distance(column, query_vec, metric='l2') -%}
  {{ exceptions.raise_compiler_error("vector_distance is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__vector_distance(column, query_vec, metric='l2') -%}
  {%- if metric == 'l2' -%}
    dist_l2({{ column }}, {{ query_vec }})
  {%- elif metric == 'cosine' -%}
    dist_cosine({{ column }}, {{ query_vec }})
  {%- elif metric == 'inner_product' -%}
    dist_ip({{ column }}, {{ query_vec }})
  {%- elif metric == 'l1' -%}
    dist_l1({{ column }}, {{ query_vec }})
  {%- elif metric == 'hamming' -%}
    dist_hamming({{ column }}, {{ query_vec }})
  {%- elif metric == 'jaccard' -%}
    dist_jaccard({{ column }}, {{ query_vec }})
  {%- else -%}
    {{ exceptions.raise_compiler_error("Unsupported metric: " ~ metric) }}
  {%- endif -%}
{%- endmacro %}


{% macro vector_avg(column) -%}
  {{ return(adapter.dispatch('vector_avg', 'dbt')(column)) }}
{%- endmacro %}

{% macro default__vector_avg(column) -%}
  {{ exceptions.raise_compiler_error("vector_avg is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__vector_avg(column) -%}
  vector_avg({{ column }})
{%- endmacro %}


{% macro knn_search(relation, column, query_vec, k=10, metric='l2') -%}
  {{ return(adapter.dispatch('knn_search', 'dbt')(relation, column, query_vec, k, metric)) }}
{%- endmacro %}

{% macro default__knn_search(relation, column, query_vec, k=10, metric='l2') -%}
  {{ exceptions.raise_compiler_error("knn_search is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__knn_search(relation, column, query_vec, k=10, metric='l2') -%}
  select *,
    {{ adapter.dispatch('vector_distance', 'dbt')(column, query_vec, metric) }} as distance
  from {{ relation }}
  order by distance
  limit {{ k }}
{%- endmacro %}


-- =============================================================================
-- Named distance functions (pgvector-compatible spellings)
-- =============================================================================

{% macro l2_distance(column, query_vec) -%}
  {{ return(adapter.dispatch('l2_distance', 'dbt')(column, query_vec)) }}
{%- endmacro %}
{% macro default__l2_distance(column, query_vec) -%}
  {{ exceptions.raise_compiler_error("l2_distance is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__l2_distance(column, query_vec) -%}
  dist_l2({{ column }}, {{ query_vec }})
{%- endmacro %}

{% macro cosine_distance(column, query_vec) -%}
  {{ return(adapter.dispatch('cosine_distance', 'dbt')(column, query_vec)) }}
{%- endmacro %}
{% macro default__cosine_distance(column, query_vec) -%}
  {{ exceptions.raise_compiler_error("cosine_distance is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__cosine_distance(column, query_vec) -%}
  dist_cosine({{ column }}, {{ query_vec }})
{%- endmacro %}

{% macro inner_product(column, query_vec) -%}
  {{ return(adapter.dispatch('inner_product', 'dbt')(column, query_vec)) }}
{%- endmacro %}
{% macro default__inner_product(column, query_vec) -%}
  {{ exceptions.raise_compiler_error("inner_product is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__inner_product(column, query_vec) -%}
  dist_ip({{ column }}, {{ query_vec }})
{%- endmacro %}

{% macro l1_distance(column, query_vec) -%}
  {{ return(adapter.dispatch('l1_distance', 'dbt')(column, query_vec)) }}
{%- endmacro %}
{% macro default__l1_distance(column, query_vec) -%}
  {{ exceptions.raise_compiler_error("l1_distance is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__l1_distance(column, query_vec) -%}
  dist_l1({{ column }}, {{ query_vec }})
{%- endmacro %}

{% macro hamming_distance(column, query_vec) -%}
  {{ return(adapter.dispatch('hamming_distance', 'dbt')(column, query_vec)) }}
{%- endmacro %}
{% macro default__hamming_distance(column, query_vec) -%}
  {{ exceptions.raise_compiler_error("hamming_distance is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__hamming_distance(column, query_vec) -%}
  dist_hamming({{ column }}, {{ query_vec }})
{%- endmacro %}

{% macro jaccard_distance(column, query_vec) -%}
  {{ return(adapter.dispatch('jaccard_distance', 'dbt')(column, query_vec)) }}
{%- endmacro %}
{% macro default__jaccard_distance(column, query_vec) -%}
  {{ exceptions.raise_compiler_error("jaccard_distance is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__jaccard_distance(column, query_vec) -%}
  dist_jaccard({{ column }}, {{ query_vec }})
{%- endmacro %}


-- =============================================================================
-- Vector transforms (element-wise ops require Float32 vectors)
-- =============================================================================

{% macro vector_add(a, b) -%}
  {{ return(adapter.dispatch('vector_add', 'dbt')(a, b)) }}
{%- endmacro %}
{% macro default__vector_add(a, b) -%}
  {{ exceptions.raise_compiler_error("vector_add is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_add(a, b) -%}
  vector_add(({{ a }})::FLOAT[], ({{ b }})::FLOAT[])
{%- endmacro %}

{% macro vector_sub(a, b) -%}
  {{ return(adapter.dispatch('vector_sub', 'dbt')(a, b)) }}
{%- endmacro %}
{% macro default__vector_sub(a, b) -%}
  {{ exceptions.raise_compiler_error("vector_sub is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_sub(a, b) -%}
  vector_sub(({{ a }})::FLOAT[], ({{ b }})::FLOAT[])
{%- endmacro %}

{% macro vector_mul(a, b) -%}
  {{ return(adapter.dispatch('vector_mul', 'dbt')(a, b)) }}
{%- endmacro %}
{% macro default__vector_mul(a, b) -%}
  {{ exceptions.raise_compiler_error("vector_mul is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_mul(a, b) -%}
  vector_mul(({{ a }})::FLOAT[], ({{ b }})::FLOAT[])
{%- endmacro %}

{% macro vector_concat(a, b) -%}
  {{ return(adapter.dispatch('vector_concat', 'dbt')(a, b)) }}
{%- endmacro %}
{% macro default__vector_concat(a, b) -%}
  {{ exceptions.raise_compiler_error("vector_concat is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_concat(a, b) -%}
  vector_concat(({{ a }})::FLOAT[], ({{ b }})::FLOAT[])
{%- endmacro %}

{% macro vector_dims(column) -%}
  {{ return(adapter.dispatch('vector_dims', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__vector_dims(column) -%}
  {{ exceptions.raise_compiler_error("vector_dims is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_dims(column) -%}
  vector_dims({{ column }})
{%- endmacro %}

{% macro vector_norm(column) -%}
  {{ return(adapter.dispatch('vector_norm', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__vector_norm(column) -%}
  {{ exceptions.raise_compiler_error("vector_norm is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_norm(column) -%}
  vector_norm({{ column }})
{%- endmacro %}

{% macro l2_normalize(column) -%}
  {{ return(adapter.dispatch('l2_normalize', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__l2_normalize(column) -%}
  {{ exceptions.raise_compiler_error("l2_normalize is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__l2_normalize(column) -%}
  l2_normalize(({{ column }})::FLOAT[])
{%- endmacro %}

{% macro binary_quantize(column) -%}
  {{ return(adapter.dispatch('binary_quantize', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__binary_quantize(column) -%}
  {{ exceptions.raise_compiler_error("binary_quantize is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__binary_quantize(column) -%}
  binary_quantize({{ column }})
{%- endmacro %}

{% macro subvector(column, start, length) -%}
  {{ return(adapter.dispatch('subvector', 'dbt')(column, start, length)) }}
{%- endmacro %}
{% macro default__subvector(column, start, length) -%}
  {{ exceptions.raise_compiler_error("subvector is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__subvector(column, start, length) -%}
  subvector(({{ column }})::FLOAT[], arrow_cast({{ start }}, 'Int32'), arrow_cast({{ length }}, 'Int32'))
{%- endmacro %}

{% macro vector_to_binary(column) -%}
  {{ return(adapter.dispatch('vector_to_binary', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__vector_to_binary(column) -%}
  {{ exceptions.raise_compiler_error("vector_to_binary is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_to_binary(column) -%}
  vector_to_binary({{ column }})
{%- endmacro %}


-- =============================================================================
-- Vector aggregates
-- =============================================================================

{% macro vector_sum(column) -%}
  {{ return(adapter.dispatch('vector_sum', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__vector_sum(column) -%}
  {{ exceptions.raise_compiler_error("vector_sum is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_sum(column) -%}
  vector_sum(({{ column }})::FLOAT[])
{%- endmacro %}

{% macro centroid(column) -%}
  {{ return(adapter.dispatch('centroid', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__centroid(column) -%}
  {{ exceptions.raise_compiler_error("centroid is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__centroid(column) -%}
  centroid({{ column }})
{%- endmacro %}

{% macro vector_median(column) -%}
  {{ return(adapter.dispatch('vector_median', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__vector_median(column) -%}
  {{ exceptions.raise_compiler_error("vector_median is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_median(column) -%}
  vector_median({{ column }})
{%- endmacro %}

{% macro vector_stddev(column) -%}
  {{ return(adapter.dispatch('vector_stddev', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__vector_stddev(column) -%}
  {{ exceptions.raise_compiler_error("vector_stddev is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_stddev(column) -%}
  vector_stddev({{ column }})
{%- endmacro %}

{% macro vector_min(column) -%}
  {{ return(adapter.dispatch('vector_min', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__vector_min(column) -%}
  {{ exceptions.raise_compiler_error("vector_min is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_min(column) -%}
  vector_min({{ column }})
{%- endmacro %}

{% macro vector_max(column) -%}
  {{ return(adapter.dispatch('vector_max', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__vector_max(column) -%}
  {{ exceptions.raise_compiler_error("vector_max is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_max(column) -%}
  vector_max({{ column }})
{%- endmacro %}


-- =============================================================================
-- Sparse vectors
-- =============================================================================

{% macro sparse_to_vector(column) -%}
  {{ return(adapter.dispatch('sparse_to_vector', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__sparse_to_vector(column) -%}
  {{ exceptions.raise_compiler_error("sparse_to_vector is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__sparse_to_vector(column) -%}
  sparse_to_vector({{ column }})
{%- endmacro %}

{% macro vector_to_sparse(column) -%}
  {{ return(adapter.dispatch('vector_to_sparse', 'dbt')(column)) }}
{%- endmacro %}
{% macro default__vector_to_sparse(column) -%}
  {{ exceptions.raise_compiler_error("vector_to_sparse is not supported on this adapter") }}
{%- endmacro %}
{% macro benostreamdb__vector_to_sparse(column) -%}
  vector_to_sparse({{ column }})
{%- endmacro %}
