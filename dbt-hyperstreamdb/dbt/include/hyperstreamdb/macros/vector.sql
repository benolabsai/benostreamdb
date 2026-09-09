{% macro type_vector(dimensions=none) -%}
  {{ return(adapter.dispatch('type_vector', 'dbt')(dimensions)) }}
{%- endmacro %}

{% macro default__type_vector(dimensions=none) -%}
  {{ exceptions.raise_compiler_error("type_vector is not supported on this adapter") }}
{%- endmacro %}

{% macro hyperstreamdb__type_vector(dimensions=none) -%}
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

{% macro hyperstreamdb__type_sparsevec(dimensions=none) -%}
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

{% macro hyperstreamdb__vector_distance(column, query_vec, metric='l2') -%}
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

{% macro hyperstreamdb__vector_avg(column) -%}
  vector_avg({{ column }})
{%- endmacro %}


{% macro knn_search(relation, column, query_vec, k=10, metric='l2') -%}
  {{ return(adapter.dispatch('knn_search', 'dbt')(relation, column, query_vec, k, metric)) }}
{%- endmacro %}

{% macro default__knn_search(relation, column, query_vec, k=10, metric='l2') -%}
  {{ exceptions.raise_compiler_error("knn_search is not supported on this adapter") }}
{%- endmacro %}

{% macro hyperstreamdb__knn_search(relation, column, query_vec, k=10, metric='l2') -%}
  select *,
    {{ adapter.dispatch('vector_distance', 'dbt')(column, query_vec, metric) }} as distance
  from {{ relation }}
  order by distance
  limit {{ k }}
{%- endmacro %}
