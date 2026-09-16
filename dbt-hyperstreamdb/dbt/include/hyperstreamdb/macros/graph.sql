-- =============================================================================
-- HyperStreamDB dbt Graph Macros
-- Native graph analytics & Graph RAG acceleration over Apache Iceberg edge tables
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. PAGERANK
-- -----------------------------------------------------------------------------
{% macro pagerank(relation, damping=0.85, iterations=30, source='source', target='target') -%}
  {{ return(adapter.dispatch('pagerank', 'dbt')(relation, damping, iterations, source, target)) }}
{%- endmacro %}

{% macro default__pagerank(relation, damping=0.85, iterations=30, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("pagerank is not supported on this adapter") }}
{%- endmacro %}

{% macro hyperstreamdb__pagerank(relation, damping=0.85, iterations=30, source='source', target='target') -%}
  select unnest(pagerank({{ source }}, {{ target }}, arrow_cast({{ damping }}, 'Float64'), arrow_cast({{ iterations }}, 'UInt32')))
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

{% macro hyperstreamdb__personalized_pagerank(relation, seeds, damping=0.85, iterations=30, directed=false, seed_weights=none, source='source', target='target') -%}
  {%- set seed_arr = "make_array(" ~ (seeds | map('string') | map('format', "arrow_cast(%s, 'UInt64')") | join(', ')) ~ ")" if seeds else "make_array()" -%}
  {%- if seed_weights is not none and seed_weights | length > 0 -%}
    {%- set weight_arr = "make_array(" ~ (seed_weights | map('string') | map('format', "arrow_cast(%s, 'Float64')") | join(', ')) ~ ")" -%}
    select unnest(personalized_pagerank(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), {{ seed_arr }}, arrow_cast({{ damping }}, 'Float64'), arrow_cast({{ iterations }}, 'UInt32'), {{ directed | lower }}, {{ weight_arr }}))
    from {{ relation }}
  {%- else -%}
    select unnest(personalized_pagerank(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), {{ seed_arr }}, arrow_cast({{ damping }}, 'Float64'), arrow_cast({{ iterations }}, 'UInt32'), {{ directed | lower }}))
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

{% macro hyperstreamdb__community_detect(relation, algorithm='louvain', resolution=1.0, source='source', target='target') -%}
  {%- if algorithm == 'louvain' -%}
    select unnest(louvain_communities(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast(1.0, 'Float32'), arrow_cast({{ resolution }}, 'Float32'))) as community
    from {{ relation }}
  {%- elif algorithm == 'label_propagation' -%}
    select unnest(label_propagation(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'))) as community
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

{% macro hyperstreamdb__graph_neighbors(relation, entity_id, hops=2, source='source', target='target') -%}
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

{% macro hyperstreamdb__subgraph(relation, seeds, hops=1, directed=false, source='source', target='target') -%}
  {%- set seed_arr = "make_array(" ~ (seeds | map('string') | map('format', "arrow_cast(%s, 'UInt64')") | join(', ')) ~ ")" if seeds else "make_array()" -%}
  select unnest(subgraph(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), {{ seed_arr }}, arrow_cast({{ hops }}, 'UInt32'), {{ directed | lower }}))
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

{% macro hyperstreamdb__connecting_paths(relation, seeds, directed=false, source='source', target='target') -%}
  {%- set seed_arr = "make_array(" ~ (seeds | map('string') | map('format', "arrow_cast(%s, 'UInt64')") | join(', ')) ~ ")" if seeds else "make_array()" -%}
  select unnest(connecting_paths(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), {{ seed_arr }}, {{ directed | lower }})) as path
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

{% macro hyperstreamdb__shortest_path(relation, start_node, end_node, source='source', target='target') -%}
  select unnest(shortest_path(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ start_node }}, 'UInt64'), arrow_cast({{ end_node }}, 'UInt64'))) as node
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

{% macro hyperstreamdb__connected_components(relation, directed=false, source='source', target='target') -%}
  {%- if directed -%}
    select unnest(strongly_connected_components(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'))) as scc_id
    from {{ relation }}
  {%- else -%}
    select unnest(connected_components(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'))) as component
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

{% macro hyperstreamdb__degree_centrality(relation, source='source', target='target') -%}
  select unnest(degree_centrality(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64')))
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

{% macro hyperstreamdb__node_similarity(relation, node_a, node_b, method='jaccard', source='source', target='target') -%}
  {%- if method == 'jaccard' -%}
    select jaccard_coefficient(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ node_a }}, 'UInt64'), arrow_cast({{ node_b }}, 'UInt64')) as score
    from {{ relation }}
  {%- elif method == 'adamic_adar' -%}
    select adamic_adar(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ node_a }}, 'UInt64'), arrow_cast({{ node_b }}, 'UInt64')) as score
    from {{ relation }}
  {%- elif method == 'preferential_attachment' -%}
    select preferential_attachment(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ node_a }}, 'UInt64'), arrow_cast({{ node_b }}, 'UInt64')) as score
    from {{ relation }}
  {%- elif method == 'resource_allocation' -%}
    select resource_allocation(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'), arrow_cast({{ node_a }}, 'UInt64'), arrow_cast({{ node_b }}, 'UInt64')) as score
    from {{ relation }}
  {%- else -%}
    {{ exceptions.raise_compiler_error("Unsupported node similarity method: " ~ method) }}
  {%- endif -%}
{%- endmacro %}


-- -----------------------------------------------------------------------------
-- 11. TOPOLOGICAL SORT
-- -----------------------------------------------------------------------------
{% macro topological_sort(relation, source='source', target='target') -%}
  {{ return(adapter.dispatch('topological_sort', 'dbt')(relation, source, target)) }}
{%- endmacro %}

{% macro default__topological_sort(relation, source='source', target='target') -%}
  {{ exceptions.raise_compiler_error("topological_sort is not supported on this adapter") }}
{%- endmacro %}

{% macro hyperstreamdb__topological_sort(relation, source='source', target='target') -%}
  select unnest(topological_sort(arrow_cast({{ source }}, 'UInt64'), arrow_cast({{ target }}, 'UInt64'))) as node
  from {{ relation }}
{%- endmacro %}
