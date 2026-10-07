-- =============================================================================
-- BenoStreamDB dbt Search Macros
-- Lexical scoring (BM25 / TF-IDF) over text columns
-- =============================================================================

-- -----------------------------------------------------------------------------
-- BM25 SCORE
-- -----------------------------------------------------------------------------
{% macro bm25_score(text_column, query) -%}
  {{ return(adapter.dispatch('bm25_score', 'dbt')(text_column, query)) }}
{%- endmacro %}

{% macro default__bm25_score(text_column, query) -%}
  {{ exceptions.raise_compiler_error("bm25_score is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__bm25_score(text_column, query) -%}
  bm25_score({{ text_column }}, {{ query }})
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- TF-IDF
-- -----------------------------------------------------------------------------
{% macro tf_idf(text_column) -%}
  {{ return(adapter.dispatch('tf_idf', 'dbt')(text_column)) }}
{%- endmacro %}

{% macro default__tf_idf(text_column) -%}
  {{ exceptions.raise_compiler_error("tf_idf is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__tf_idf(text_column) -%}
  tf_idf({{ text_column }})
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- KEYWORD SEARCH (BM25 ranked retrieval)
-- -----------------------------------------------------------------------------
{% macro keyword_search(relation, text_column, query, k=10) -%}
  {{ return(adapter.dispatch('keyword_search', 'dbt')(relation, text_column, query, k)) }}
{%- endmacro %}

{% macro default__keyword_search(relation, text_column, query, k=10) -%}
  {{ exceptions.raise_compiler_error("keyword_search is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__keyword_search(relation, text_column, query, k=10) -%}
  select *,
    {{ adapter.dispatch('bm25_score', 'dbt')(text_column, query) }} as score
  from {{ relation }}
  order by score desc
  limit {{ k }}
{%- endmacro %}
