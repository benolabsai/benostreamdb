{% materialization table, adapter='hyperstreamdb' %}
  {%- set target_relation = this.incorporate(type='table') -%}

  {%- set existing_relation = load_cached_relation(this) -%}
  {%- if existing_relation is not none -%}
    {{ adapter.drop_relation(existing_relation) }}
  {%- endif -%}

  {% call statement('main') %}
    {{ get_create_table_as_sql(False, target_relation, sql) }}
  {% endcall %}

  {{ create_indexes(target_relation) }}

  {{ return({'relations': [target_relation]}) }}
{% endmaterialization %}
