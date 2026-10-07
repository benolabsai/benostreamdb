-- =============================================================================
-- BenoStreamDB dbt JSON Macros
-- PostgreSQL-compatible `json` path functions (JSON is stored as text)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- JSON EXTRACT PATH (returns json)
-- -----------------------------------------------------------------------------
{% macro json_extract_path(json_column, path) -%}
  {{ return(adapter.dispatch('json_extract_path', 'dbt')(json_column, path)) }}
{%- endmacro %}

{% macro default__json_extract_path(json_column, path) -%}
  {{ exceptions.raise_compiler_error("json_extract_path is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__json_extract_path(json_column, path) -%}
  json_extract_path({{ json_column }}, {{ path }})
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- JSON EXTRACT PATH TEXT (returns text)
-- -----------------------------------------------------------------------------
{% macro json_extract_path_text(json_column, path) -%}
  {{ return(adapter.dispatch('json_extract_path_text', 'dbt')(json_column, path)) }}
{%- endmacro %}

{% macro default__json_extract_path_text(json_column, path) -%}
  {{ exceptions.raise_compiler_error("json_extract_path_text is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__json_extract_path_text(json_column, path) -%}
  json_extract_path_text({{ json_column }}, {{ path }})
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- JSON CONTAINS (recursive containment, `@>`)
-- -----------------------------------------------------------------------------
{% macro json_contains(json_column, candidate) -%}
  {{ return(adapter.dispatch('json_contains', 'dbt')(json_column, candidate)) }}
{%- endmacro %}

{% macro default__json_contains(json_column, candidate) -%}
  {{ exceptions.raise_compiler_error("json_contains is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__json_contains(json_column, candidate) -%}
  json_contains({{ json_column }}, {{ candidate }})
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- JSON EXISTS (top-level key/element)
-- -----------------------------------------------------------------------------
{% macro json_exists(json_column, key) -%}
  {{ return(adapter.dispatch('json_exists', 'dbt')(json_column, key)) }}
{%- endmacro %}

{% macro default__json_exists(json_column, key) -%}
  {{ exceptions.raise_compiler_error("json_exists is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__json_exists(json_column, key) -%}
  json_exists({{ json_column }}, {{ key }})
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- JSON TYPEOF
-- -----------------------------------------------------------------------------
{% macro json_typeof(json_column) -%}
  {{ return(adapter.dispatch('json_typeof', 'dbt')(json_column)) }}
{%- endmacro %}

{% macro default__json_typeof(json_column) -%}
  {{ exceptions.raise_compiler_error("json_typeof is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__json_typeof(json_column) -%}
  json_typeof({{ json_column }})
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- JSON PATH EXISTS (jsonpath subset)
-- -----------------------------------------------------------------------------
{% macro json_path_exists(json_column, path) -%}
  {{ return(adapter.dispatch('json_path_exists', 'dbt')(json_column, path)) }}
{%- endmacro %}

{% macro default__json_path_exists(json_column, path) -%}
  {{ exceptions.raise_compiler_error("json_path_exists is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__json_path_exists(json_column, path) -%}
  json_path_exists({{ json_column }}, {{ path }})
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- JSON PATH QUERY (jsonpath subset)
-- -----------------------------------------------------------------------------
{% macro json_path_query(json_column, path) -%}
  {{ return(adapter.dispatch('json_path_query', 'dbt')(json_column, path)) }}
{%- endmacro %}

{% macro default__json_path_query(json_column, path) -%}
  {{ exceptions.raise_compiler_error("json_path_query is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__json_path_query(json_column, path) -%}
  json_path_query({{ json_column }}, {{ path }})
{%- endmacro %}
