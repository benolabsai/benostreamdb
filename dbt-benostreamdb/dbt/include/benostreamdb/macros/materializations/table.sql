{% materialization table, adapter='benostreamdb' %}
  {%- set target_relation = this.incorporate(type='table') -%}

  {#- The embedded session starts with an empty relation cache, so
      `load_cached_relation` never sees warehouse-resident tables. Always issue
      an idempotent `drop table if exists` — the engine deletes the underlying
      data, which makes dbt's drop-then-create cycle re-runnable. -#}
  {% call statement('drop_existing', auto_begin=False) %}
    drop table if exists {{ target_relation }}
  {% endcall %}

  {% call statement('main') %}
    {{ get_create_table_as_sql(False, target_relation, sql) }}
  {% endcall %}

  {{ create_indexes(target_relation) }}

  {{ return({'relations': [target_relation]}) }}
{% endmaterialization %}
