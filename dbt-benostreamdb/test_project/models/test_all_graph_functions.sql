{{ config(
    materialized='table',
    schema='custom_schema'
) }}

-- Exercises every graph macro end-to-end. Node-list results are unioned into a
-- single relation; scalar results are cross-joined so each function executes.

with pr as (
    select * from ({{ pagerank(ref('edges'), damping=0.85, iterations=10) }}) t
),
ppr as (
    select * from ({{ personalized_pagerank(ref('edges'), seeds=[1, 2], seed_weights=[0.9, 0.1]) }}) t
),
louvain as (
    select * from ({{ community_detect(ref('edges'), algorithm='louvain') }}) t
),
leiden as (
    select * from ({{ community_detect(ref('edges'), algorithm='leiden') }}) t
),
label_prop as (
    select * from ({{ community_detect(ref('edges'), algorithm='label_propagation') }}) t
),
components as (
    select * from ({{ connected_components(ref('edges')) }}) t
),
scc as (
    select * from ({{ connected_components(ref('edges'), directed=true) }}) t
),
degree as (
    select * from ({{ degree_centrality(ref('edges')) }}) t
),
closeness as (
    select * from ({{ closeness_centrality(ref('edges')) }}) t
),
betweenness as (
    select * from ({{ betweenness_centrality(ref('edges')) }}) t
),
neighbors as (
    select * from ({{ graph_neighbors(ref('edges'), entity_id=2, hops=1) }}) t
),
subgraph_cte as (
    select * from ({{ subgraph(ref('edges'), seeds=[1, 2], hops=1) }}) t
),
shortest as (
    select * from ({{ shortest_path(ref('edges'), start_node=1, end_node=4) }}) t
),
all_shortest as (
    select * from ({{ all_shortest_paths(ref('edges'), start_node=1, end_node=4) }}) t
),
connecting as (
    select * from ({{ connecting_paths(ref('edges'), seeds=[1, 4]) }}) t
),
similarity as (
    select * from ({{ node_similarity(ref('edges'), node_a=1, node_b=3, method='jaccard') }}) t
),
pref_attach as (
    select * from ({{ node_similarity(ref('edges'), node_a=1, node_b=3, method='preferential_attachment') }}) t
),
triangles as (
    select * from ({{ triangle_count(ref('edges')) }}) t
),
modularity_cte as (
    select * from ({{ modularity(ref('edges'), community_column='source') }}) t
),
drift as (
    select * from ({{ drift_search(ref('edges'), query="'graph rag'", top_communities=[1, 2], n_depth=1) }}) t
),
regional as (
    select * from ({{ regional_drift(ref('edges'), query="'graph rag'", seeds=[1], hops=1, n_depth=1) }}) t
)

select 'pagerank' as fn, count(*) as rows_returned from pr
union all select 'personalized_pagerank', count(*) from ppr
union all select 'louvain', count(*) from louvain
union all select 'leiden', count(*) from leiden
union all select 'label_propagation', count(*) from label_prop
union all select 'connected_components', count(*) from components
union all select 'strongly_connected_components', count(*) from scc
union all select 'degree_centrality', count(*) from degree
union all select 'closeness_centrality', count(*) from closeness
union all select 'betweenness_centrality', count(*) from betweenness
union all select 'graph_neighbors', count(*) from neighbors
union all select 'subgraph', count(*) from subgraph_cte
union all select 'shortest_path', count(*) from shortest
union all select 'all_shortest_paths', count(*) from all_shortest
union all select 'connecting_paths', count(*) from connecting
union all select 'jaccard_coefficient', count(*) from similarity
union all select 'preferential_attachment', count(*) from pref_attach
union all select 'triangle_count', count(*) from triangles
union all select 'modularity', count(*) from modularity_cte
union all select 'drift_search', count(*) from drift
union all select 'regional_drift', count(*) from regional
