-- =============================================================================
-- BenoStreamDB dbt DDL Macros
-- Index lifecycle (CREATE INDEX / DROP INDEX) over Apache Iceberg tables.
--
-- The engine intercepts these statements before DataFusion planning and
-- dispatches them to the core `Table` API, so they work from any dbt model or
-- hook. `type` is one of the engine's index algorithms (hnsw, hnsw_pq,
-- hnsw_tq4, hnsw_tq8, hnsw_sq8, bm25, bloom, bitmap, composite_bitmap,
-- csr_graph, json_path).
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


{% macro add_primary_key(relation, columns) -%}
  {{ return(adapter.dispatch('add_primary_key', 'dbt')(relation, columns)) }}
{%- endmacro %}

{% macro default__add_primary_key(relation, columns) -%}
  {{ exceptions.raise_compiler_error("add_primary_key is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__add_primary_key(relation, columns) -%}
  {%- set cols = columns if columns is string else columns | join(', ') -%}
  alter table {{ relation }} add primary key ({{ cols }})
{%- endmacro %}


{% macro drop_primary_key(relation) -%}
  {{ return(adapter.dispatch('drop_primary_key', 'dbt')(relation)) }}
{%- endmacro %}

{% macro default__drop_primary_key(relation) -%}
  {{ exceptions.raise_compiler_error("drop_primary_key is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__drop_primary_key(relation) -%}
  alter table {{ relation }} drop primary key
{%- endmacro %}


{% macro rebuild_cluster_index(relation, column='embedding', k=1024, m=16, ef_construction=200, target_shard_mb=512) -%}
  {{ return(adapter.dispatch('rebuild_cluster_index', 'dbt')(relation, column, k, m, ef_construction, target_shard_mb)) }}
{%- endmacro %}

{% macro default__rebuild_cluster_index(relation, column='embedding', k=1024, m=16, ef_construction=200, target_shard_mb=512) -%}
  {{ exceptions.raise_compiler_error("rebuild_cluster_index is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__rebuild_cluster_index(relation, column='embedding', k=1024, m=16, ef_construction=200, target_shard_mb=512) -%}
  alter table {{ relation }} execute rebuild_cluster_index(column => '{{ column }}', k => {{ k }}, m => {{ m }}, ef_construction => {{ ef_construction }}, target_shard_mb => {{ target_shard_mb }})
{%- endmacro %}


{% macro warmup(relation, n_probes=100, max_memory_gb=8.0) -%}
  {{ return(adapter.dispatch('warmup', 'dbt')(relation, n_probes, max_memory_gb)) }}
{%- endmacro %}

{% macro default__warmup(relation, n_probes=100, max_memory_gb=8.0) -%}
  {{ exceptions.raise_compiler_error("warmup is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__warmup(relation, n_probes=100, max_memory_gb=8.0) -%}
  alter table {{ relation }} execute warmup(n_probes => {{ n_probes }}, max_memory_gb => {{ max_memory_gb }})
{%- endmacro %}
