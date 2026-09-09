{% macro hyperstreamdb__create_table_as(temporary, relation, sql) -%}
  {%- set sql_header = config.get('sql_header', none) -%}
  {%- set partition_by = config.get('partition_by') -%}
  {%- if partition_by is none -%}
    {%- set partition_by = config.get('partitioned_by') -%}
  {%- endif -%}

  {{ sql_header if sql_header is not none }}

  create table {{ relation }}
  {% if partition_by %}
    {% if partition_by is string %}
      PARTITIONED BY ({{ partition_by }})
    {% else %}
      PARTITIONED BY ({{ partition_by | join(', ') }})
    {% endif %}
  {% endif %}
  as (
    {{ sql }}
  );
{%- endmacro %}

{% macro hyperstreamdb__create_view_as(relation, sql) -%}
  {%- set sql_header = config.get('sql_header', none) -%}

  {{ sql_header if sql_header is not none }}

  create view {{ relation }} as (
    {{ sql }}
  );
{%- endmacro %}

{% macro hyperstreamdb__drop_relation(relation) -%}
  {% call statement('drop_relation', auto_begin=False) -%}
    drop {{ relation.type or 'table' }} if exists {{ relation }}
  {%- endcall %}
{% endmacro %}

{% macro hyperstreamdb__rename_relation(from_relation, to_relation) -%}
  {% call statement('rename_relation') -%}
    alter {{ from_relation.type }} {{ from_relation }} rename to {{ to_relation }}
  {%- endcall %}
{% endmacro %}

{% macro hyperstreamdb__create_schema(relation) -%}
  {%- call statement('create_schema') -%}
    create schema if not exists {{ relation.without_identifier().include(database=False) }}
  {%- endcall -%}
{% endmacro %}

{% macro hyperstreamdb__drop_schema(relation) -%}
  {%- call statement('drop_schema') -%}
    drop schema if exists {{ relation.without_identifier().include(database=False) }}
  {%- endcall -%}
{% endmacro %}

{% macro hyperstreamdb__check_schema_exists(information_schema, schema) -%}
  {% call statement('check_schema_exists', fetch_result=True, auto_begin=False) %}
    select count(*)
    from information_schema.schemata
    where schema_name = '{{ schema }}'
  {% endcall %}
  {{ return(load_result('check_schema_exists').table) }}
{% endmacro %}

{% macro hyperstreamdb__get_columns_in_relation(relation) -%}
  {% call statement('get_columns_in_relation', fetch_result=True) %}
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale
      from information_schema.columns
      where table_name = '{{ relation.identifier }}'
        {% if relation.schema %}
        and table_schema = '{{ relation.schema }}'
        {% endif %}
      order by ordinal_position
  {% endcall %}
  {% set table = load_result('get_columns_in_relation').table %}
  {{ return(sql_convert_columns_in_relation(table)) }}
{% endmacro %}

{% macro hyperstreamdb__get_create_index_sql(relation, index_dict) -%}
  {%- set index_columns = index_dict.get('columns', []) -%}
  {%- set index_name = index_dict.get('name', relation.identifier ~ '_' ~ index_columns | join('_') ~ '_idx') -%}
  {{ log("HyperStreamDB manages indices natively; skipping CREATE INDEX for " ~ index_name, info=True) }}
  {% do return(None) %}
{%- endmacro %}
