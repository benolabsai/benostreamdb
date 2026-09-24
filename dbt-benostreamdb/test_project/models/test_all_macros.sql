{{ config(
    materialized='table',
    schema='custom_schema',
    indexes=[
      {'columns': ['embedding'], 'type': 'hnsw'}
    ]
) }}

with source_data as (
    select 1 as id, ARRAY[1.0, 2.0, 3.0]::{{ type_vector() }} as embedding, ARRAY[0.1, 0.0, 0.0]::{{ type_sparsevec() }} as sparse_embedding
    union all
    select 2 as id, ARRAY[4.0, 5.0, 6.0]::{{ type_vector(3) }} as embedding, ARRAY[0.0, 0.2, 0.0]::{{ type_sparsevec(3) }} as sparse_embedding
    union all
    select 3 as id, ARRAY[7.0, 8.0, 9.0]::{{ type_vector() }} as embedding, ARRAY[0.0, 0.0, 0.3]::{{ type_sparsevec() }} as sparse_embedding
)

select *,
    {{ vector_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]', 'l2') }} as dist_l2,
    {{ vector_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]', 'cosine') }} as dist_cosine,
    {{ vector_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]', 'inner_product') }} as dist_ip,
    {{ vector_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]', 'l1') }} as dist_l1,
    {{ vector_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]', 'hamming') }} as dist_hamming,
    {{ vector_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]', 'jaccard') }} as dist_jaccard,
    {{ vector_avg('embedding') }} over () as avg_embedding
from source_data
