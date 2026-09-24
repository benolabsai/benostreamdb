{{ config(
    materialized='incremental',
    partition_by='pt',
    incremental_strategy='insert_overwrite'
) }}

SELECT 1 as id, 'A' as pt
UNION ALL
SELECT 2 as id, 'B' as pt
