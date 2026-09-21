# HyperStreamDB examples

Everything that demonstrates the engine, in one tree (the old top-level
`demo/` directory was merged in here).

## Flagship: full-site Wikipedia Graph RAG UI

**[`web_ui/`](web_ui/README.md)** — Streamlit app over the **entire English
Wikipedia link graph** (51.8M pages / 383M edges): keyset-paginated SQL browse,
hybrid BM25+dense search, two-level Graph RAG (seed index → CSR expansion →
bitmap-filtered rerank), DRIFT regional communities, and microsecond CSR
traversals. Prepare with `scripts/prepare_demo.py`, then
`streamlit run examples/web_ui/app.py` — full instructions and measured
timings in [`web_ui/README.md`](web_ui/README.md).

## Notebooks & scripts (getting started → RAG → comprehensive guide)

| File | What it covers |
|---|---|
| [`01_basics.py`](01_basics.py) / [`notebooks/01_installation_and_basics.ipynb`](notebooks/01_installation_and_basics.ipynb) | install, create/write, filters, vector search |
| [`02_rag_pipeline.py`](02_rag_pipeline.py) / [`notebooks/02_rag_pipeline.ipynb`](notebooks/02_rag_pipeline.ipynb) | end-to-end RAG: embed → index → hybrid retrieve → answer |
| [`notebooks/03_comprehensive_guide.ipynb`](notebooks/03_comprehensive_guide.ipynb) | feature tour (graphs, quantization, catalogs, streaming) |
| [`notebooks/comprehensive_guide.ipynb`](notebooks/comprehensive_guide.ipynb) | extended guide incl. the presentation table |
| [`run_demos.py`](run_demos.py) | executes the notebooks top-to-bottom (`jupyter nbconvert --execute`) |
| [`rag_hf_demo.py`](rag_hf_demo.py) | RAG demo driven by a Hugging Face model |
| [`verify_explain.py`](verify_explain.py) | `EXPLAIN` / planner-output smoke check |
| [`requirements.txt`](requirements.txt) | extra deps for the notebooks/demos |

## Feature examples

| File | What it covers |
|---|---|
| [`vector_index_types.py`](vector_index_types.py) | HNSW / IVF / PQ / TurboQuant index flavors |
| [`python_distance_api_examples.py`](python_distance_api_examples.py) | distance-metric APIs (L2, cosine, dot…) |
| [`pgvector_python_examples.py`](pgvector_python_examples.py) | pgvector-compatible operators from Python |
| [`pgvector_sql_examples.sql`](pgvector_sql_examples.sql) | pgvector-compatible SQL operators |
| [`catalog_usage.rs`](catalog_usage.rs) | Rust catalog APIs (REST/Glue/Hive/Unity/Nessie) |
| [`configs/`](configs) | catalog TOML examples (`rest`, `glue`, `hive`, `unity`, `nessie`) |
