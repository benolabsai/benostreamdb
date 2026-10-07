{{ config(
    materialized='table',
    schema='custom_schema'
) }}

-- PostgreSQL-compatible JSON path functions over a text column.

with docs as (
    select 1 as id, '{"a": {"b": [1, 2]}, "c": "text"}' as doc
)

select
    id,
    {{ json_extract_path('doc', "'a'") }} as extracted,
    {{ json_extract_path_text('doc', "'c'") }} as extracted_text,
    {{ json_contains('doc', "'{\"c\": \"text\"}'") }} as contains,
    {{ json_exists('doc', "'a'") }} as exists_key,
    {{ json_typeof('doc') }} as type_of,
    {{ json_path_exists('doc', "'$.a.b'") }} as path_exists,
    {{ json_path_query('doc', "'$.a.b'") }} as path_query
from docs
