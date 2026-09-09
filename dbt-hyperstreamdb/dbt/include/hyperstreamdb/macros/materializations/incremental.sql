{% materialization incremental, adapter='hyperstreamdb' %}

  {% set unique_key = config.get('unique_key') %}
  {% set partition_by = config.get('partition_by') %}
  {% if partition_by is none %}
    {% set partition_by = config.get('partitioned_by') %}
  {% endif %}

  {% set incremental_strategy = config.get('incremental_strategy') or 'append' %}

  {% set target_relation = this.incorporate(type='table') %}
  {% set existing_relation = load_cached_relation(this) %}
  {% set tmp_relation = make_temp_relation(target_relation) %}

  {{ run_hooks(pre_hooks, inside_transaction=False) }}
  {{ run_hooks(pre_hooks, inside_transaction=True) }}

  {% set to_drop = [] %}

  {% if existing_relation is none %}
      {% do adapter.drop_relation(target_relation) %}
      {% set build_sql = get_create_table_as_sql(False, target_relation, sql) %}
  {% elif existing_relation.is_view or should_full_refresh() %}
      {% do adapter.drop_relation(existing_relation) %}
      {% set build_sql = get_create_table_as_sql(False, target_relation, sql) %}
  {% else %}
      {% do adapter.drop_relation(tmp_relation) %}
      {% call statement('create_tmp_relation') %}
          {{ get_create_table_as_sql(True, tmp_relation, sql) }}
      {% endcall %}
      {% do to_drop.append(tmp_relation) %}

      {% if incremental_strategy == 'insert_overwrite' %}
          {% if not partition_by %}
              {{ exceptions.raise_compiler_error("insert_overwrite requires a `partition_by` or `partitioned_by` config") }}
          {% endif %}
          
          {% set partition_col = partition_by %}
          {% if partition_by is not string %}
             {% set partition_col = partition_by[0] %}
          {% endif %}
          
          {# Looping over partitions #}
          {% set get_partitions_sql %}
              SELECT DISTINCT {{ partition_col }} FROM {{ tmp_relation }}
          {% endset %}
          
          {% set partitions = run_query(get_partitions_sql) %}
          
          {% for row in partitions %}
              {% set p_val = row[0] %}
              {% set del_sql %}
                  DELETE FROM {{ target_relation }} WHERE {{ partition_col }} = '{{ p_val }}'
              {% endset %}
              {% call statement('delete_partition') %}
                  {{ del_sql }}
              {% endcall %}
              
              {% set ins_sql %}
                  INSERT INTO {{ target_relation }} SELECT * FROM {{ tmp_relation }} WHERE {{ partition_col }} = '{{ p_val }}'
              {% endset %}
              {% call statement('insert_partition') %}
                  {{ ins_sql }}
              {% endcall %}
          {% endfor %}
          
          {% set build_sql = "SELECT 1 as result" %} {# dummy statement for the main call #}
          
      {% elif incremental_strategy == 'append' %}
          {% set build_sql %}
              INSERT INTO {{ target_relation }} SELECT * FROM {{ tmp_relation }}
          {% endset %}
      {% elif incremental_strategy == 'delete+insert' %}
          {% if not unique_key %}
              {{ exceptions.raise_compiler_error("delete+insert requires a `unique_key` config") }}
          {% endif %}
          
          {% set del_sql %}
              DELETE FROM {{ target_relation }} WHERE {{ unique_key }} IN (SELECT {{ unique_key }} FROM {{ tmp_relation }})
          {% endset %}
          {% call statement('delete_unique_key') %}
              {{ del_sql }}
          {% endcall %}
          
          {% set build_sql %}
              INSERT INTO {{ target_relation }} SELECT * FROM {{ tmp_relation }}
          {% endset %}
      {% else %}
          {{ exceptions.raise_compiler_error("invalid incremental_strategy: " ~ incremental_strategy) }}
      {% endif %}
  {% endif %}

  {% call statement("main") %}
      {{ build_sql }}
  {% endcall %}

  {% for rel in to_drop %}
      {% do adapter.drop_relation(rel) %}
  {% endfor %}

  {{ create_indexes(target_relation) }}

  {{ run_hooks(post_hooks, inside_transaction=True) }}
  {{ run_hooks(post_hooks, inside_transaction=False) }}

  {{ return({'relations': [target_relation]}) }}

{% endmaterialization %}
