{{ config(
    materialized='table',
    schema='custom_schema'
) }}

-- Lexical scoring functions (BM25 / TF-IDF) and the keyword_search macro.

with docs as (
    select 1 as id, 'the cat sat on the mat' as body
    union all
    select 2 as id, 'the dog sat on the log' as body
)

select
    id,
    {{ bm25_score('body', "'cat'") }} as bm25,
    {{ tf_idf('body') }} as tfidf
from docs
