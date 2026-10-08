-- =============================================================================
-- BenoStreamDB dbt Graph Macros
-- Native graph analytics & Graph RAG acceleration over Apache Iceberg edge tables
-- =============================================================================

-- Internal helper: render a list of values as a make_array(...) of arrow_cast
-- expressions, e.g. [1, 2] -> make_array(arrow_cast(1, 'UInt64'), arrow_cast(2, 'UInt64')).
{%- macro _bsdb_arrow_array(values, sql_type) -%}
  {%- set parts = [] -%}
  {%- for v in (values or []) -%}
    {%- set _ = parts.append("arrow_cast(" ~ v ~ ", '" ~ sql_type ~ "')") -%}
  {%- endfor -%}
  make_array({{ parts | join(', ') }})
{%- endmacro -%}

-- -----------------------------------------------------------------------------
-- 1. PAGERANK
-- -----------------------------------------------------------------------------
{% macro pagerank(relation, damping=0.85, iterations=30, source='source', target='target') -%}
  {{ return(adapter.dispatch('pagerank', 'dbt')(relation, damping, iterations, source, target)) }}
{%- endmacro %}

{% macro default__pagerank(relation, damping=0.85, iterations=30, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("pagerank is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__pagerank(relation, damping=0.85, iterations=30, source='source', target='target') -%}
  select unnest(graph_pagerank(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ damping }}, 'Float64'), arrow_cast({{ iterations }}, 'UInt32')))
  from {{ relation }}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 2. PERSONALIZED PAGERANK (HippoRAG-style)
-- -----------------------------------------------------------------------------
{% macro personalized_pagerank(relation, seeds, damping=0.85, iterations=30, directed=false, seed_weights=none, source='source', target='target') -%}
  {{ return(adapter.dispatch('personalized_pagerank', 'dbt')(relation, seeds, damping, iterations, directed, seed_weights, source, target)) }}
{%- endmacro %}

{% macro default__personalized_pagerank(relation, seeds, damping=0.85, iterations=30, directed=false, seed_weights=none, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("personalized_pagerank is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__personalized_pagerank(relation, seeds, damping=0.85, iterations=30, directed=false, seed_weights=none, source='source', target='target') -%}
  {%- set seed_arr = _bsdb_arrow_array(seeds, 'UInt64') -%}
  {%- if seed_weights is not none and seed_weights | length > 0 -%}
    {%- set weight_arr = _bsdb_arrow_array(seed_weights, 'Float64') -%}
    select unnest(graph_personalized_pagerank(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), {{ seed_arr }}, arrow_cast({{ damping }}, 'Float64'), arrow_cast({{ iterations }}, 'UInt32'), {{ directed | lower }}, {{ weight_arr }}))
    from {{ relation }}
  {%- else -%}
    select unnest(graph_personalized_pagerank(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), {{ seed_arr }}, arrow_cast({{ damping }}, 'Float64'), arrow_cast({{ iterations }}, 'UInt32'), {{ directed | lower }}))
    from {{ relation }}
  {%- endif -%}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 3. COMMUNITY DETECTION (Louvain & Label Propagation)
-- -----------------------------------------------------------------------------
{% macro community_detect(relation, algorithm='louvain', resolution=1.0, source='source', target='target') -%}
  {{ return(adapter.dispatch('community_detect', 'dbt')(relation, algorithm, resolution, source, target)) }}
{%- endmacro %}

{% macro default__community_detect(relation, algorithm='louvain', resolution=1.0, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("community_detect is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__community_detect(relation, algorithm='louvain', resolution=1.0, source='source', target='target') -%}
  {%- if algorithm == 'louvain' -%}
    select unnest(graph_louvain_communities(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast(1.0, 'Float32'), arrow_cast({{ resolution }}, 'Float32'))) as community
    from {{ relation }}
  {%- elif algorithm == 'leiden' -%}
    select unnest(graph_leiden_communities(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ resolution }}, 'Float32'))) as community
    from {{ relation }}
  {%- elif algorithm == 'label_propagation' -%}
    select unnest(graph_label_propagation(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'))) as community
    from {{ relation }}
  {%- else -%}
    {{ exceptions.raise_compiler_error("Unsupported community detection algorithm: " ~ algorithm) }}
  {%- endif -%}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 4. GRAPH NEIGHBORS
-- -----------------------------------------------------------------------------
{% macro graph_neighbors(relation, entity_id, hops=2, source='source', target='target') -%}
  {{ return(adapter.dispatch('graph_neighbors', 'dbt')(relation, entity_id, hops, source, target)) }}
{%- endmacro %}

{% macro default__graph_neighbors(relation, entity_id, hops=2, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("graph_neighbors is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__graph_neighbors(relation, entity_id, hops=2, source='source', target='target') -%}
  select unnest(graph_neighbors(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ entity_id }}, 'UInt64'), arrow_cast({{ hops }}, 'UInt32'))) as neighbor
  from {{ relation }}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 5. SUBGRAPH EXTRACTION
-- -----------------------------------------------------------------------------
{% macro subgraph(relation, seeds, hops=1, directed=false, source='source', target='target') -%}
  {{ return(adapter.dispatch('subgraph', 'dbt')(relation, seeds, hops, directed, source, target)) }}
{%- endmacro %}

{% macro default__subgraph(relation, seeds, hops=1, directed=false, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("subgraph is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__subgraph(relation, seeds, hops=1, directed=false, source='source', target='target') -%}
  {%- set seed_arr = _bsdb_arrow_array(seeds, 'UInt64') -%}
  select unnest(graph_subgraph(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), {{ seed_arr }}, arrow_cast({{ hops }}, 'UInt32'), {{ directed | lower }}))
  from {{ relation }}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 6. CONNECTING PATHS
-- -----------------------------------------------------------------------------
{% macro connecting_paths(relation, seeds, directed=false, source='source', target='target') -%}
  {{ return(adapter.dispatch('connecting_paths', 'dbt')(relation, seeds, directed, source, target)) }}
{%- endmacro %}

{% macro default__connecting_paths(relation, seeds, directed=false, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("connecting_paths is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__connecting_paths(relation, seeds, directed=false, source='source', target='target') -%}
  {%- set seed_arr = _bsdb_arrow_array(seeds, 'UInt64') -%}
  select unnest(graph_connecting_paths(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), {{ seed_arr }}, {{ directed | lower }})) as path
  from {{ relation }}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 7. SHORTEST PATH
-- -----------------------------------------------------------------------------
{% macro shortest_path(relation, start_node, end_node, source='source', target='target') -%}
  {{ return(adapter.dispatch('shortest_path', 'dbt')(relation, start_node, end_node, source, target)) }}
{%- endmacro %}

{% macro default__shortest_path(relation, start_node, end_node, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("shortest_path is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__shortest_path(relation, start_node, end_node, source='source', target='target') -%}
  select unnest(graph_shortest_path(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ start_node }}, 'UInt64'), arrow_cast({{ end_node }}, 'UInt64'))) as node
  from {{ relation }}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 8. CONNECTED COMPONENTS
-- -----------------------------------------------------------------------------
{% macro connected_components(relation, directed=false, source='source', target='target') -%}
  {{ return(adapter.dispatch('connected_components', 'dbt')(relation, directed, source, target)) }}
{%- endmacro %}

{% macro default__connected_components(relation, directed=false, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("connected_components is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__connected_components(relation, directed=false, source='source', target='target') -%}
  {%- if directed -%}
    select unnest(graph_strongly_connected_components(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'))) as scc_id
    from {{ relation }}
  {%- else -%}
    select unnest(graph_connected_components(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'))) as component
    from {{ relation }}
  {%- endif -%}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 9. DEGREE CENTRALITY
-- -----------------------------------------------------------------------------
{% macro degree_centrality(relation, source='source', target='target') -%}
  {{ return(adapter.dispatch('degree_centrality', 'dbt')(relation, source, target)) }}
{%- endmacro %}

{% macro default__degree_centrality(relation, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("degree_centrality is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__degree_centrality(relation, source='source', target='target') -%}
  select unnest(graph_degree_centrality(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64')))
  from {{ relation }}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 10. NODE SIMILARITY (Link Prediction)
-- -----------------------------------------------------------------------------
{% macro node_similarity(relation, node_a, node_b, method='jaccard', source='source', target='target') -%}
  {{ return(adapter.dispatch('node_similarity', 'dbt')(relation, node_a, node_b, method, source, target)) }}
{%- endmacro %}

{% macro default__node_similarity(relation, node_a, node_b, method='jaccard', source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("node_similarity is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__node_similarity(relation, node_a, node_b, method='jaccard', source='source', target='target') -%}
  {%- if method == 'jaccard' -%}
    select graph_jaccard_coefficient(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ node_a }}, 'UInt64'), arrow_cast({{ node_b }}, 'UInt64')) as score
    from {{ relation }}
  {%- elif method == 'adamic_adar' -%}
    select coalesce(sum(1.0 / ln(cast(deg.degree as double))), 0.0) as score
    from (
      select {{ target }} as neighbor from {{ relation }} where {{ source }} = {{ node_a }}
      union
      select {{ source }} as neighbor from {{ relation }} where {{ target }} = {{ node_a }}
    ) start_neighbors
    join (
      select {{ target }} as neighbor from {{ relation }} where {{ source }} = {{ node_b }}
      union
      select {{ source }} as neighbor from {{ relation }} where {{ target }} = {{ node_b }}
    ) end_neighbors on start_neighbors.neighbor = end_neighbors.neighbor
    join (
      select node_id, count(*) as degree from (
        select {{ source }} as node_id from {{ relation }}
        union all
        select {{ target }} as node_id from {{ relation }}
      ) group by node_id
    ) deg on start_neighbors.neighbor = deg.node_id
    where deg.degree > 1
  {%- elif method == 'preferential_attachment' -%}
    select graph_preferential_attachment(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ node_a }}, 'UInt64'), arrow_cast({{ node_b }}, 'UInt64')) as score
    from {{ relation }}
  {%- elif method == 'resource_allocation' -%}
    select coalesce(sum(1.0 / cast(deg.degree as double)), 0.0) as score
    from (
      select {{ target }} as neighbor from {{ relation }} where {{ source }} = {{ node_a }}
      union
      select {{ source }} as neighbor from {{ relation }} where {{ target }} = {{ node_a }}
    ) start_neighbors
    join (
      select {{ target }} as neighbor from {{ relation }} where {{ source }} = {{ node_b }}
      union
      select {{ source }} as neighbor from {{ relation }} where {{ target }} = {{ node_b }}
    ) end_neighbors on start_neighbors.neighbor = end_neighbors.neighbor
    join (
      select node_id, count(*) as degree from (
        select {{ source }} as node_id from {{ relation }}
        union all
        select {{ target }} as node_id from {{ relation }}
      ) group by node_id
    ) deg on start_neighbors.neighbor = deg.node_id
    where deg.degree > 0
  {%- else -%}
    {{ exceptions.raise_compiler_error("Unsupported node similarity method: " ~ method) }}
  {%- endif -%}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 11. TRIANGLE COUNT
-- -----------------------------------------------------------------------------
{% macro triangle_count(relation, source='source', target='target') -%}
  {{ return(adapter.dispatch('triangle_count', 'dbt')(relation, source, target)) }}
{%- endmacro %}

{% macro default__triangle_count(relation, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("triangle_count is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__triangle_count(relation, source='source', target='target') -%}
  select graph_triangle_count(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64')) as triangle_count
  from {{ relation }}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 12. MODULARITY (given a community assignment)
-- -----------------------------------------------------------------------------
{% macro modularity(relation, community_column, weight=none, source='source', target='target') -%}
  {{ return(adapter.dispatch('modularity', 'dbt')(relation, community_column, weight, source, target)) }}
{%- endmacro %}

{% macro default__modularity(relation, community_column, weight=none, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("modularity is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__modularity(relation, community_column, weight=none, source='source', target='target') -%}
  {%- if weight is not none -%}
    select graph_modularity(
      arrow_cast({{ source }}, 'UInt64'),
      arrow_cast({{ target }}, 'UInt64'),
      arrow_cast({{ weight }}, 'Float64'),
      arrow_cast({{ community_column }}, 'UInt64'),
      arrow_cast({{ community_column }}, 'UInt64')
    ) as modularity
    from {{ relation }}
  {%- else -%}
    select graph_modularity(
      arrow_cast({{ source }}, 'UInt64'),
      arrow_cast({{ target }}, 'UInt64'),
      arrow_cast({{ community_column }}, 'UInt64'),
      arrow_cast({{ community_column }}, 'UInt64')
    ) as modularity
    from {{ relation }}
  {%- endif -%}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 13. CLOSENESS CENTRALITY
-- -----------------------------------------------------------------------------
{% macro closeness_centrality(relation, directed=false, source='source', target='target') -%}
  {{ return(adapter.dispatch('closeness_centrality', 'dbt')(relation, directed, source, target)) }}
{%- endmacro %}

{% macro default__closeness_centrality(relation, directed=false, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("closeness_centrality is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__closeness_centrality(relation, directed=false, source='source', target='target') -%}
  select unnest(graph_closeness_centrality(
    arrow_cast({{ source }}, 'UInt64'),
    arrow_cast({{ target }}, 'UInt64'),
    {{ directed | lower }}
  ))
  from {{ relation }}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 14. BETWEENNESS CENTRALITY
-- -----------------------------------------------------------------------------
{% macro betweenness_centrality(relation, directed=false, source='source', target='target') -%}
  {{ return(adapter.dispatch('betweenness_centrality', 'dbt')(relation, directed, source, target)) }}
{%- endmacro %}

{% macro default__betweenness_centrality(relation, directed=false, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("betweenness_centrality is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__betweenness_centrality(relation, directed=false, source='source', target='target') -%}
  select unnest(graph_betweenness_centrality(
    arrow_cast({{ source }}, 'UInt64'),
    arrow_cast({{ target }}, 'UInt64'),
    {{ directed | lower }}
  ))
  from {{ relation }}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 15. ALL SHORTEST PATHS
-- -----------------------------------------------------------------------------
{% macro all_shortest_paths(relation, start_node=none, end_node=none, source='source', target='target') -%}
  {{ return(adapter.dispatch('all_shortest_paths', 'dbt')(relation, start_node, end_node, source, target)) }}
{%- endmacro %}

{% macro default__all_shortest_paths(relation, start_node=none, end_node=none, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("all_shortest_paths is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__all_shortest_paths(relation, start_node=none, end_node=none, source='source', target='target') -%}
  {%- if start_node is not none and end_node is not none -%}
    select unnest(graph_all_shortest_paths(
      arrow_cast({{ source }}, 'UInt64'),
      arrow_cast({{ target }}, 'UInt64'),
      arrow_cast({{ start_node }}, 'UInt64'),
      arrow_cast({{ end_node }}, 'UInt64')
    )) as path
    from {{ relation }}
  {%- else -%}
    select unnest(graph_all_shortest_paths(
      arrow_cast({{ source }}, 'UInt64'),
      arrow_cast({{ target }}, 'UInt64')
    )) as path
    from {{ relation }}
  {%- endif -%}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 16. DRIFT SEARCH (GraphRAG query drift)
-- -----------------------------------------------------------------------------
{% macro drift_search(relation, query, top_communities, n_depth=2, graph_uri='', mode='local', source='source', target='target') -%}
  {{ return(adapter.dispatch('drift_search', 'dbt')(relation, query, top_communities, n_depth, graph_uri, mode, source, target)) }}
{%- endmacro %}

{% macro default__drift_search(relation, query, top_communities, n_depth=2, graph_uri='', mode='local', source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("drift_search is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__drift_search(relation, query, top_communities, n_depth=2, graph_uri='', mode='local', source='source', target='target') -%}
  select unnest(drift_search(
    arrow_cast({{ source }}, 'UInt64'),
    arrow_cast({{ target }}, 'UInt64'),
    {{ query }},
    {{ _bsdb_arrow_array(top_communities, 'UInt64') }},
    arrow_cast({{ n_depth }}, 'UInt32'),
    '{{ graph_uri }}',
    '{{ mode }}'
  )) as node
  from {{ relation }}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 18. GRAPH TRAVERSAL TABLE FUNCTIONS (FROM graph_*(...))
-- -----------------------------------------------------------------------------
-- These emit the engine's DataFusion table functions, so a graph walk is a
-- `FROM` source rather than an aggregate. The table is passed as an unquoted
-- `schema.table` name (the engine resolves it in the session's default catalog).
-- `mode` is one of auto | in_memory | out_of_core | cached.

{% macro graph_neighbors_table(relation, seeds, hops=1, mode='auto', source=none, target=none) -%}
  {{ return(adapter.dispatch('graph_neighbors_table', 'dbt')(relation, seeds, hops, mode, source, target)) }}
{%- endmacro %}

{% macro default__graph_neighbors_table(relation, seeds, hops=1, mode='auto', source=none, target=none) -%}
  {{ exceptions.raise_compiler_error("graph_neighbors_table is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__graph_neighbors_table(relation, seeds, hops=1, mode='auto', source=none, target=none) -%}
  {%- set seed_str = seeds if seeds is string else seeds | join(',') -%}
  select * from graph_neighbors('{{ relation.database }}.{{ relation.schema }}.{{ relation.identifier }}', '{{ seed_str }}', {{ hops }}, '{{ mode }}'{% if source is not none %}, '{{ source }}'{% endif %}{% if target is not none %}, '{{ target }}'{% endif %})
{%- endmacro %}


{% macro graph_shortest_path_table(relation, source_node, target_node, mode='auto', source=none, target=none) -%}
  {{ return(adapter.dispatch('graph_shortest_path_table', 'dbt')(relation, source_node, target_node, mode, source, target)) }}
{%- endmacro %}

{% macro default__graph_shortest_path_table(relation, source_node, target_node, mode='auto', source=none, target=none) -%}
  {{ exceptions.raise_compiler_error("graph_shortest_path_table is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__graph_shortest_path_table(relation, source_node, target_node, mode='auto', source=none, target=none) -%}
  select * from graph_shortest_path('{{ relation.database }}.{{ relation.schema }}.{{ relation.identifier }}', {{ source_node }}, {{ target_node }}, '{{ mode }}'{% if source is not none %}, '{{ source }}'{% endif %}{% if target is not none %}, '{{ target }}'{% endif %})
{%- endmacro %}


{% macro graph_all_shortest_paths_table(relation, source_node, target_node, mode='auto', source=none, target=none) -%}
  {{ return(adapter.dispatch('graph_all_shortest_paths_table', 'dbt')(relation, source_node, target_node, mode, source, target)) }}
{%- endmacro %}

{% macro default__graph_all_shortest_paths_table(relation, source_node, target_node, mode='auto', source=none, target=none) -%}
  {{ exceptions.raise_compiler_error("graph_all_shortest_paths_table is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__graph_all_shortest_paths_table(relation, source_node, target_node, mode='auto', source=none, target=none) -%}
  select * from graph_all_shortest_paths('{{ relation.database }}.{{ relation.schema }}.{{ relation.identifier }}', {{ source_node }}, {{ target_node }}, '{{ mode }}'{% if source is not none %}, '{{ source }}'{% endif %}{% if target is not none %}, '{{ target }}'{% endif %})
{%- endmacro %}


{% macro graph_subgraph_table(relation, seeds, hops=1, mode='auto', source=none, target=none) -%}
  {{ return(adapter.dispatch('graph_subgraph_table', 'dbt')(relation, seeds, hops, mode, source, target)) }}
{%- endmacro %}

{% macro default__graph_subgraph_table(relation, seeds, hops=1, mode='auto', source=none, target=none) -%}
  {{ exceptions.raise_compiler_error("graph_subgraph_table is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__graph_subgraph_table(relation, seeds, hops=1, mode='auto', source=none, target=none) -%}
  {%- set seed_str = seeds if seeds is string else seeds | join(',') -%}
  select * from graph_subgraph('{{ relation.database }}.{{ relation.schema }}.{{ relation.identifier }}', '{{ seed_str }}', {{ hops }}, '{{ mode }}'{% if source is not none %}, '{{ source }}'{% endif %}{% if target is not none %}, '{{ target }}'{% endif %})
{%- endmacro %}


{% macro graph_connecting_paths_table(relation, seeds, mode='auto', source=none, target=none) -%}
  {{ return(adapter.dispatch('graph_connecting_paths_table', 'dbt')(relation, seeds, mode, source, target)) }}
{%- endmacro %}

{% macro default__graph_connecting_paths_table(relation, seeds, mode='auto', source=none, target=none) -%}
  {{ exceptions.raise_compiler_error("graph_connecting_paths_table is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__graph_connecting_paths_table(relation, seeds, mode='auto', source=none, target=none) -%}
  {%- set seed_str = seeds if seeds is string else seeds | join(',') -%}
  select * from graph_connecting_paths('{{ relation.database }}.{{ relation.schema }}.{{ relation.identifier }}', '{{ seed_str }}', '{{ mode }}'{% if source is not none %}, '{{ source }}'{% endif %}{% if target is not none %}, '{{ target }}'{% endif %})
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 17. REGIONAL DRIFT SEARCH (seeded GraphRAG drift)
-- -----------------------------------------------------------------------------
{% macro regional_drift(relation, query, seeds, hops=1, n_depth=2, graph_uri='', mode='local', source='source', target='target') -%}
  {{ return(adapter.dispatch('regional_drift', 'dbt')(relation, query, seeds, hops, n_depth, graph_uri, mode, source, target)) }}
{%- endmacro %}

{% macro default__regional_drift(relation, query, seeds, hops=1, n_depth=2, graph_uri='', mode='local', source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("regional_drift is not supported on this adapter") }}
{%- endmacro %}

{% macro benostreamdb__regional_drift(relation, query, seeds, hops=1, n_depth=2, graph_uri='', mode='local', source='source', target='target') -%}
  select unnest(regional_drift(
    arrow_cast({{ source }}, 'UInt64'),
    arrow_cast({{ target }}, 'UInt64'),
    {{ query }},
    {{ _bsdb_arrow_array(seeds, 'UInt64') }},
    arrow_cast({{ hops }}, 'UInt32'),
    arrow_cast({{ n_depth }}, 'UInt32'),
    '{{ graph_uri }}',
    '{{ mode }}'
  )) as node
  from {{ relation }}
{%- endmacro %}
