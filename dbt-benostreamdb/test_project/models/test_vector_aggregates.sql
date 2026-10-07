{{ config(
    materialized='table',
    schema='custom_schema'
) }}

-- Vector aggregate functions over a vector column.

with vectors as (
    select 1 as id, ARRAY[1.0, 2.0, 3.0]::{{ type_vector() }} as embedding
    union all
    select 2 as id, ARRAY[4.0, 5.0, 6.0]::{{ type_vector() }} as embedding
)

select
    {{ vector_sum('embedding') }} as summed,
    {{ vector_avg('embedding') }} as averaged,
    {{ centroid('embedding') }} as centroid_v,
    {{ vector_median('embedding') }} as median_v,
    {{ vector_stddev('embedding') }} as stddev_v,
    {{ vector_min('embedding') }} as min_v,
    {{ vector_max('embedding') }} as max_v
from vectors
