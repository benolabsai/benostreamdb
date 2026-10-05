# Competitor benchmarks

Run reference engines under the **same resource envelope and parameters** as the
BenoStreamDB harness ([`../ann_benchmarks/run.py`](../ann_benchmarks/run.py)), and
emit the same metrics. See [`docs/BENCHMARKING_PLAN.md`](../../docs/BENCHMARKING_PLAN.md)
for the full plan.

## Runner

[`run_competitor.py`](run_competitor.py) exposes one CLI with pluggable adapters.

### Vector (ANN-Benchmarks datasets)

```bash
# FAISS HNSW
python benchmarks/competitors/run_competitor.py \
  --engine faiss --dataset sift-128-euclidean \
  --cores 8 --ram-gb 16 \
  --m 16 --ef-construction 200 --ef-search 200 \
  --queries 500 --out benchmarks/competitors/results/faiss_sift_20k.json

# hnswlib
python benchmarks/competitors/run_competitor.py \
  --engine hnswlib --dataset sift-128-euclidean --limit 20000 \
  --m 16 --ef-construction 200 --ef-search 200 --queries 500
```

`--m`, `--ef-construction`, `--ef-search`, `--metric`, `--k`, `--limit`, and
`--queries` map 1:1 onto the BenoStreamDB harness flags, so the same configuration
can be run on both sides without translation.

The FAISS adapter picks `METRIC_L2` / `METRIC_INNER_PRODUCT` from `--metric` and
L2-normalizes for cosine. The hnswlib adapter maps cosine → `space="cosine"`.

### Graph (edge list)

```bash
python benchmarks/competitors/run_competitor.py \
  --engine networkx --graph-edges edges.txt --algorithm pagerank
# algorithms: pagerank | connected_components | shortest_path (--source N)
```

## Resource envelope

`--cores` sets `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`; `--ram-gb` sets
`RLIMIT_AS` so an over-budget run fails loudly. The BenoStreamDB side is
constrained by `RAYON_NUM_THREADS`, `BSDB_MAX_CONCURRENCY`, etc. (see the plan,
§5). **Match `--cores`/`--ram-gb` on both sides or the comparison is invalid.**

## Docker: one hardware profile for everyone

[`docker_bench.sh`](docker_bench.sh) runs the whole matrix in Docker with a
single shared envelope applied to **every** participant — the server engines
*and* the runner that executes each client (including BenoStreamDB):

```bash
# CPU baseline (default engines: faiss hnswlib lancedb pgvector
#                              opensearch benostreamdb)
benchmarks/competitors/docker_bench.sh --cpus 8 --mem 16g \
    --dataset sift-128-euclidean --limit 20000 --queries 500

# GPU pass (faiss, benostreamdb — GPU-capable engines only)
benchmarks/competitors/docker_bench.sh --gpu --cpus 8 --mem 16g

# Both passes, tagged device=cpu and device=gpu
benchmarks/competitors/docker_bench.sh --both --cpus 8 --mem 16g
```

* [`docker-compose.bench.yml`](docker-compose.bench.yml) — pgvector, OpenSearch,
  Neo4j+GDS, and the `bench` runner, all under
  `deploy.resources.limits.{cpus,memory}` from `BENCH_CPUS`/`BENCH_MEM`.
* [`docker-compose.bench.gpu.yml`](docker-compose.bench.gpu.yml) — GPU override
  (`gpus: all` + a `bench-gpu` runner from
  [`Dockerfile.bench.gpu`](Dockerfile.bench.gpu), which installs `faiss-gpu-cu12`).
* [`Dockerfile.bench`](Dockerfile.bench) — runner image with the harness and all
  competitor clients (FAISS, hnswlib, LanceDB, DuckDB, psycopg, opensearch-py,
  neo4j). Installs the BenoStreamDB abi3 wheel mounted at `/wheels`.

Results are written to `results/{engine}_{dataset}_{device}.json`, plus
`results/rollup.md` (with a Device column) and `results/hardware_profile.txt`.

### Hardware profile

CPU is the baseline comparison; GPU is opt-in per engine. Each run records the
*container-visible* profile so results are self-describing:

```json
"env": { "cpu_model": "...", "cores": 8, "ram_gb": 16.0,
         "containerized": true, "gpus": ["NVIDIA A100-SXM4-40GB, 40960 MiB"] }
```

The host profile (CPU model, cores, RAM, Docker/Compose versions, GPU list,
envelope) is written to `results/hardware_profile.txt`. GPUs appear only in the
GPU pass; CPU runs report `"gpus": []`.

**GPU-capable engines here:** FAISS (faiss-gpu), BenoStreamDB (wgpu/cuda), and
cuGraph (`cugraph-cu13`). pgvector, LanceDB, OpenSearch, Neo4j (CPU GDS), and
DuckDB are CPU-only, so they are excluded from the GPU pass by default.

> **GPU is slower on small datasets — by design.** On the 20k-vector SIFT run
> the GPU pass is *slower* than CPU for both engines (faiss 6.7k vs 8.8k QPS;
> BenoStreamDB 1.9k vs 1.9k QPS, with a slower build). Fixed PCIe transfer and
> kernel-launch overheads dominate when the data is small, the harness measures
> single-query latency (batch = 1, so there is no GPU parallelism to exploit),
> and HNSW search is a sequential pointer-chasing walk rather than dense math.
> BenoStreamDB's GPU path (`benostream-gpu-ann`) accelerates index
> *construction*, not query. GPU wins at scale: millions of vectors, large
> batch queries, and dense/flat or IVF-PQ search.

## Output

Each run prints a JSON record and, with `--out`, writes `<name>.json` +
`<name>.md` under `results/`, using the schema in the plan (§6): `build_s`,
`index_mb`, `recall_at_k`, `qps`, `p50_ms`, `p99_ms`, plus the envelope and env.

## Server-backed engines

pgvector, OpenSearch, LanceDB, and Neo4j take connection options (or env vars):

| Engine | Option / env | Example |
|---|---|---|
| pgvector | `--dsn` / `PGVECTOR_DSN` | `postgresql://user:pass@localhost/bench` |
| opensearch | `--host` / `ES_URL` | `http://localhost:9200` |
| lancedb | `--path` (default: temp dir) | `/data/lancedb_bench` |
| neo4j | `--uri` / `--user` / `--password` (or `NEO4J_URI/USER/PASSWORD`) | `bolt://localhost:7687` |

DuckDB is embedded: pass `--sql "..."` or `--parquet path.parquet`.

## Adapter status

| Engine | Workload | Installed here | Notes |
|---|---|---|---|
| FAISS | vector | no | `pip install faiss-cpu` |
| hnswlib | vector | no | `pip install hnswlib` |
| pgvector | vector | no | `pip install psycopg[binary]` + Postgres w/ pgvector |
| LanceDB | vector | no | `pip install lancedb` (IVF_PQ) |
| OpenSearch | vector | no | `pip install opensearch-py`; dense_vector HNSW kNN |
| NetworkX | graph | yes | reference (single-threaded) baseline |
| Neo4j | graph | no | `pip install neo4j` + GDS plugin (pagerank/wcc/dijkstra) |
| DuckDB | sql | no | `pip install duckdb`; scan/aggregate + vss |

Every adapter reports `{"available": false, "error": "..."}` when its client
library or server is missing, instead of crashing.

Adding an engine = subclass `VectorAdapter` (implement `build`/`search`/
`index_bytes`), add a branch in `run_graph`, or add a `run_sql` family; then
register it in `vector_adapters()` / the `main()` dispatch.

## Still to add

- **Trino (SQL)** — a Trino competitor adapter (the connector exists, but the
  benchmark runner has no Trino client yet).
- **Spark/Trino Iceberg round-trips** — read a BenoStreamDB-written Iceberg
  table back through Spark/Trino and vice versa, to prove format compatibility.

Each must run under the same envelope and emit the same schema.

**Already covered** (not "to add"):

- **cuGraph** (GPU graph) — `cugraph-cu13` in the GPU runner, `renumber=True` +
  managed memory, verified against NetworkX.
- **ClickHouse** (SQL) — `_sql_clickhouse` materialises the Parquet into a
  `MergeTree` table and runs the shared SQL.
- **Tantivy** (BM25/hybrid) — the `tantivy` adapter lives in `run_competitor.py`,
  and the BEIR lexical/hybrid suite (`benchmarks/beir/`) benchmarks BM25 vs
  Tantivy vs Hybrid RRF.
