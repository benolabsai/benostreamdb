# Benchmarking: how to reproduce every number

Status: runnable methodology. Owner: BenoStreamDB core.
Companion docs: [`BENCHMARKING_PLAN.md`](BENCHMARKING_PLAN.md) (the full plan:
datasets, competitor set, metrics schema), [`RESOURCE_LIMITS.md`](RESOURCE_LIMITS.md)
(memory/CPU knobs), [`CONCURRENCY.md`](CONCURRENCY.md) (writer/reader model).

This document is the **reproducibility contract**: the exact commands, the
environment capture, and the correctness oracles behind every published number.
If a number cannot be regenerated from a command in this file, it does not
belong in the announcement.

---

## 1. The reproducibility contract

Every published result must satisfy all four:

1. **One command.** The result is produced by a committed script, not a REPL
   session. The command is recorded next to the number.
2. **Self-describing environment.** Each JSON record carries an `env` block
   (CPU model, cores, RAM, containerized, GPU list) and the run writes
   `results/hardware_profile.txt` (host CPU/RAM, Docker/Compose versions, GPU
   driver, envelope, dataset, HNSW parameters).
3. **Identical envelope.** Every participant — server engines *and* the runner
   that executes each client, including BenoStreamDB — runs under the same
   `--cpus` / `--mem` cgroup limits. A comparison across different envelopes is
   invalid.
4. **A correctness oracle.** The index must not change the answer. Exact indexes
   are checked against a full scan; ANN indexes are checked for recall. See §7.

---

## 2. Prerequisites

| Requirement | Why |
|---|---|
| Docker Engine + Compose v2.30+ | `gpus: all` in the GPU override needs Compose ≥ 2.30 |
| NVIDIA Container Toolkit | GPU pass only (`--gpu` / `--both`) |
| Python 3.10+ with `maturin[zig]` | `--build-wheel` builds the abi3 manylinux wheel |
| ~10 GB free disk | SIFT HDF5 (~525 MB) + runner images (~9 GB GPU) |

The runner image installs every competitor client, so no host Python packages
are needed for the Docker path.

---

## 3. Quick start

```bash
# CPU baseline: faiss hnswlib lancedb pgvector opensearch benostreamdb
benchmarks/competitors/docker_bench.sh --build-wheel \
    --cpus 8 --mem 16g \
    --dataset sift-128-euclidean --limit 20000 --queries 500

# GPU pass (faiss, benostreamdb, cugraph), same envelope
benchmarks/competitors/docker_bench.sh --gpu --cpus 8 --mem 16g

# Both passes, tagged device=cpu and device=gpu
benchmarks/competitors/docker_bench.sh --both --cpus 8 --mem 16g
```

`--build-wheel` runs `maturin build --release --zig --compatibility
manylinux_2_28 -o dist` first. The manylinux tag matters: the runner image is
`python:3.12-slim` (glibc 2.36), so a host-built wheel linked against a newer
glibc will not load.

Results land in `benchmarks/competitors/results/`:

```
results/{engine}_{dataset}_{device}.json   # one record per engine
results/{engine}_{dataset}_{device}.md     # human-readable sibling
results/rollup.md                          # all engines, one table
results/hardware_profile.txt               # host + envelope + dataset
```

---

## 4. Workload families

The same runner image and envelope are reused; `--workload` selects the family.

### 4.1 Vector (ANN-Benchmarks)

```bash
benchmarks/competitors/docker_bench.sh --build-wheel \
    --cpus 8 --mem 16g \
    --dataset sift-128-euclidean --limit 20000 --queries 500 \
    --m 16 --ef-construction 200 --ef-search 200 --k 10
```

`--m`, `--ef-construction`, `--ef-search`, `--metric`, `--k`, `--limit`, and
`--queries` map 1:1 onto the BenoStreamDB harness flags, so the same
configuration runs on both sides without translation.

**Like-for-like rule.** The competitor adapters return ids only (no payload
fetch). The rollup therefore reports BenoStreamDB's **pure-index** QPS
(`pure_index_qps`), not its full search+row-fetch number, and LanceDB projects
only `id` in its search. Comparing BenoStreamDB's full-search QPS against a
competitor's search-only QPS is not a valid comparison.

### 4.2 Graph

```bash
benchmarks/competitors/docker_bench.sh --workload graph \
    --cpus 8 --mem 16g \
    --graph-engines "networkx neo4j cugraph benostreamdb" \
    --algorithm pagerank \
    --graph-nodes 10000 --graph-edges-count 50000
```

Algorithms: `pagerank`, `connected_components`, `shortest_path` (`--source N`).
A synthetic edge list is generated deterministically (seed 42) and mounted at
`/opt/bench/data/graph_edges.txt`. `networkx` is the single-threaded reference;
`cugraph` is the GPU reference; `neo4j` uses the GDS plugin.

### 4.3 SQL / analytical

```bash
benchmarks/competitors/docker_bench.sh --workload sql \
    --cpus 8 --mem 16g \
    --sql-engines "duckdb datafusion clickhouse benostreamdb" \
    --sql-rows 500000
```

A synthetic Parquet table (`id`, `value`, `category`) is generated and mounted
at `/opt/bench/data/clickbench.parquet`; the query is a group-by aggregate.

### 4.4 Lexical / hybrid (BEIR)

The lexical suite lives in `benchmarks/beir/` and benchmarks BM25 vs Tantivy vs
Hybrid RRF on BEIR datasets (`scifact`, `nfcorpus`, `arguana`). See
[`../benchmarks/beir/`](../benchmarks/beir/).

---

## 5. Resource envelope

`--cpus` sets `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`; `--mem` sets the container
cgroup `deploy.resources.limits.memory`. The BenoStreamDB side is constrained by
`RAYON_NUM_THREADS`, `BSDB_MAX_CONCURRENCY`, and the ingest RAM backpressure
(see [`RESOURCE_LIMITS.md`](RESOURCE_LIMITS.md)).

> **`RLIMIT_AS` is not applied to GPU engines.** `RLIMIT_AS` limits *virtual*
> address space, not resident memory. CUDA/RAPIDS managed memory reserves far
> more virtual address space than physical RAM, so applying the RAM envelope to
> `cugraph` (or any `--device gpu` run) makes the `dlopen` of `libcugraph.so`
> fail with a misleading `cannot open shared object file`. The container cgroup
> already enforces the physical RAM envelope, so `apply_envelope` skips
> `RLIMIT_AS` for GPU engines.

**Match `--cpus`/`--mem` on both sides or the comparison is invalid.**

---

## 6. Output schema

Each record follows the plan's schema (§6 of
[`BENCHMARKING_PLAN.md`](BENCHMARKING_PLAN.md)):

```json
{
  "engine": "benostreamdb",
  "workload": "vector",
  "available": true,
  "build_s": 1.23,
  "index_mb": 4.5,
  "recall_at_k": 0.99,
  "qps": 1931.0,
  "pure_index_qps": 2100.0,
  "p50_ms": 0.4,
  "p99_ms": 0.9,
  "env": { "cpu_model": "...", "cores": 8, "ram_gb": 16.0,
           "containerized": true, "gpus": [] }
}
```

An engine that is not installed or whose server is unreachable emits
`{"available": false, "error": "..."}` instead of crashing, so a partial matrix
still produces a complete rollup.

---

## 7. Correctness oracles

The benchmark numbers are only credible if the index does not change the answer.
These are enforced by tests, not by inspection:

| Index | Oracle | Test |
|---|---|---|
| Scalar bitmap, inverted/BM25, composite | indexed result **identical** to full scan | [`tests/test_differential_index_oracle.rs`](../tests/test_differential_index_oracle.rs) |
| HNSW / IVF / TurboQuant / PQ | exact nearest neighbour appears in indexed top-k (recall) | same file |
| JSON-path | indexed result identical to full scan; unindexed path falls back to scan | [`tests/test_json_path_index.rs`](../tests/test_json_path_index.rs) |
| cuGraph | PageRank matches NetworkX (max abs diff < 1e-5, identical top-5) | `results/cugraph_correct.txt` |

Run them with:

```bash
cargo test --test test_differential_index_oracle
cargo test --test test_json_path_index
```

---

## 8. Regenerating the rollup and report

`docker_bench.sh` writes `results/rollup.md` from the JSON records produced
**during that run** (mtime-filtered by `RUN_START`), so stale results from a
previous dataset or device are never double-counted. The rollup uses
`pure_index_qps` for BenoStreamDB (see §4.1).

The narrative report is [`benchmarks/results/BENCHMARK_REPORT.md`](../benchmarks/results/BENCHMARK_REPORT.md);
`benchmarks/generate_summary.py` renders the summary tables.

---

## 9. How to read the numbers

- **The SIFT comparison is 20,000 vectors.** The dedicated in-memory engines
  (FAISS, hnswlib) are optimised for a narrower problem than an Iceberg-native
  architecture. Do not market the numbers as broad claims. Say:
  *"BenoStreamDB delivers sub-millisecond vector search while maintaining
  transactional Iceberg tables and persistent overlay indexes"* — not
  *"BenoStreamDB is faster than FAISS."*
- **GPU is slower on small datasets — by design.** On the 20k SIFT run the GPU
  pass is slower than CPU for both engines: fixed PCIe transfer and kernel-launch
  overheads dominate, the harness measures single-query latency (batch = 1, so
  there is no GPU parallelism to exploit), and HNSW search is a sequential
  pointer-chasing walk rather than dense math. BenoStreamDB's GPU path
  (`benostream-gpu-ann`) accelerates index *construction*, not query. GPU wins at
  scale: millions of vectors, large batch queries, dense/flat or IVF-PQ search.
- **Overlay indexes trade build cost and storage for query speed and
  reconstructibility.** The report must state where we lose.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `libcugraph.so: cannot open shared object file` | `RLIMIT_AS` applied to a GPU engine | Fixed in `apply_envelope`; rebuild the base runner image (`docker compose ... build bench bench-gpu`) |
| Wheel fails to load on the runner (`GLIBC_2.4x not found`) | host-built wheel linked against newer glibc | use `--build-wheel` (manylinux_2_28) |
| `cudaErrorMemoryAllocation: out of memory` on a 50-series card | cu12 wheels have no sm_120 kernels | the GPU image installs `cugraph-cu13`; `RMM_POOL_SIZE=12GB` leaves headroom for the Wayland compositor |
| GPU image ships an old harness | `bench-gpu` is `FROM bsdb-bench-runner:latest` | `docker_bench.sh` builds `bench` before `bench-gpu` |
| `docker compose run` cannot see the GPU | Compose < 2.30 ignores `gpus: all` | upgrade Compose, or use `docker run --gpus all` |
