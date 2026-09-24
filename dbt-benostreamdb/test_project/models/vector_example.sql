{{ config(materialized='table') }}

with source_data as (
    select 1 as id, ARRAY[1.0, 2.0, 3.0]::{{ type_vector() }} as embedding
    union all
    select 2 as id, ARRAY[4.0, 5.0, 6.0]::{{ type_vector(3) }} as embedding
)

{{ knn_search(relation='source_data', column='embedding', query_vec='ARRAY[1.0, 1.0, 1.0]', k=5, metric='l2') }}
