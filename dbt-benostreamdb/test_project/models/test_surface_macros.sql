{{ config(
    materialized='table',
    schema='custom_schema',
    post_hook="{{ create_index(this, 'node', 'bitmap') }}"
) }}

-- Graph traversal via the table-function macro (`FROM graph_neighbors(...)`),
-- plus index creation via the DDL macro in a post-hook. The macro emits the
-- full `select * from graph_neighbors(...)`.
{{ graph_neighbors_table(ref('edges'), '0', 2) }}
