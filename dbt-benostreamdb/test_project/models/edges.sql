{{ config(
    materialized='table',
    schema='custom_schema'
) }}

-- Small demo edge table (Graph RAG style) consumed by the graph macros.
-- source/target are the edge endpoints; weight drives weighted algorithms.
select 1 as source, 2 as target, 'links' as relation, 1.0 as weight
union all
select 2 as source, 3 as target, 'links' as relation, 1.0 as weight
union all
select 2 as source, 4 as target, 'links' as relation, 1.0 as weight
union all
select 3 as source, 4 as target, 'links' as relation, 1.0 as weight
union all
select 4 as source, 1 as target, 'links' as relation, 1.0 as weight
