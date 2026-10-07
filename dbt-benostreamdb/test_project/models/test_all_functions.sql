{{ config(
    materialized='table',
    schema='custom_schema'
) }}

-- Per-row vector functions: named distances + transforms.
-- (Aggregates, lexical and JSON functions live in their own models so each
-- query stays a single simple projection — DataFusion's nested-loop/cross-join
-- executors reject multi-partition inputs.)

with vectors as (
    select 1 as id, ARRAY[1.0, 2.0, 3.0]::{{ type_vector() }} as embedding
    union all
    select 2 as id, ARRAY[4.0, 5.0, 6.0]::{{ type_vector() }} as embedding
)

select
    id,
    {{ l2_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]') }} as d_l2,
    {{ cosine_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]') }} as d_cosine,
    {{ inner_product('embedding', 'ARRAY[1.0, 1.0, 1.0]') }} as d_ip,
    {{ l1_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]') }} as d_l1,
    {{ hamming_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]') }} as d_hamming,
    {{ jaccard_distance('embedding', 'ARRAY[1.0, 1.0, 1.0]') }} as d_jaccard,
    {{ vector_dims('embedding') }} as dims,
    {{ vector_norm('embedding') }} as norm,
    {{ vector_add('embedding', 'ARRAY[1.0, 1.0, 1.0]') }} as added,
    {{ vector_sub('embedding', 'ARRAY[1.0, 1.0, 1.0]') }} as subbed,
    {{ vector_mul('embedding', 'ARRAY[2.0, 2.0, 2.0]') }} as mulled,
    {{ vector_concat('embedding', 'ARRAY[9.0]') }} as concatenated,
    {{ l2_normalize('embedding') }} as normalized,
    {{ subvector('embedding', 0, 2) }} as sub,
    {{ binary_quantize('embedding') }} as quantized
from vectors
