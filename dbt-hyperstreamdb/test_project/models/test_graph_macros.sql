{{ config(
    materialized='table',
    schema='custom_schema'
) }}

-- Example dbt models demonstrating HyperStreamDB Graph UDF macros
-- 1. Materialize PageRank importance scores
with pr_scores as (
    {{ pagerank(ref('edges'), damping=0.85, iterations=30) }}
),

-- 2. Materialize Louvain community assignments
louvain_clusters as (
    {{ community_detect(ref('edges'), algorithm='louvain', resolution=1.0) }}
),

-- 3. Extract 2-hop neighborhood of entity 42
entity_neighborhood as (
    {{ graph_neighbors(ref('edges'), entity_id=42, hops=2) }}
),

-- 4. Extract induced subgraph for seed entities
seed_subgraph as (
    {{ subgraph(ref('edges'), seeds=[1, 2, 3], hops=1) }}
),

-- 5. Personalized PageRank grounded at seed entities with HippoRAG weights
hippo_ppr as (
    {{ personalized_pagerank(ref('edges'), seeds=[1, 2], damping=0.85, seed_weights=[0.9, 0.1]) }}
)

select * from pr_scores
