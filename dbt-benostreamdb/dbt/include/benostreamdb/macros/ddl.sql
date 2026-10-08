-- =============================================================================
-- BenoStreamDB dbt DDL Macros
-- Index lifecycle (CREATE INDEX / DROP INDEX) over Apache Iceberg tables.
--
-- The engine intercepts these statements before DataFusion planning and
-- dispatches them to the core `Table` API, so they work from any dbt model or
-- hook. `type` is one of the engine's index algorithms (hnsw, hnsw_pq,
-- hnsw_tq4, hnsw_tq8, bm25, bloom, bitmap, composite_bitmap, csr_graph,
-- json_path).
-- =============================================================================

{% macro create_index(relation, column, type='bitmap', name=none) -%}
  {{ return(adapter.dispatch('create_index', 'dbt')(relation, column, type, name)) }}
{%- endmacro %}

{% macro default__create_index(relation, column, type='bitmap', name=none) -%}
  {{ exceptions.raise_compiler_error("create_index is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__create_index(relation, column, type='bitmap', name=none) -%}
  {%- set idx_name = name if name is not none else relation.identifier ~ '_' ~ column ~ '_idx' -%}
  create index {{ idx_name }} on {{ relation }} ({{ column }}) using {{ type }}
{%- endmacro %}


{% macro drop_index(relation, name) -%}
  {{ return(adapter.dispatch('drop_index', 'dbt')(relation, name)) }}
{%- endmacro %}

{% macro default__drop_index(relation, name) -%}
  {{ exceptions.raise_compiler_error("drop_index is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__drop_index(relation, name) -%}
  alter table {{ relation }} drop index {{ name }}
{%- endmacro %}
