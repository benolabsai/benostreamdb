?# Benchmarking: how to reproduce every number

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

Every published result must satisfy all five:

1. **One command.** The result is produced by a committed script, not a REPL
   session. The command is recorded next to the number.
2. **Self-describing environment.** Each JSON record carries an `env` block
   (CPU model, cores, RAM, containerized, GPU list) and the run writes
   `results/hardware_profile.txt` (host CPU/RAM, Docker/Compose versions, GPU
   driver, envelope, dataset, HNSW parameters).
3. **Identical envelope and equivalent configuration.** Every participant —
   server engines *and* the runner that executes each client, including
   BenoStreamDB — runs under the same `--cpus` / `--mem` cgroup limits, against
   the same dataset, query workload, `k`, and index parameters (HNSW
   `m`/`ef_construction`/`ef_search`, etc.), with recall held at parity. A
   comparison across different envelopes *or* different configurations is
   invalid.
4. **A correctness oracle.** The index must not change the answer. Exact indexes
   are checked against a full scan; ANN indexes are checked for recall. See §7.
5. **One machine state, recorded.** See below.

### 1.1 Machine state and session integrity

The machine state is part of the measurement. DIMM timings, a CPU undervolt,
frequency-limit changes (`amd-pstate`/`intel_pstate`, RAPL), and the kernel move
**every engine in a table at once** — competitor rows included — because they all
run on the same host. They do not move everything *equally*: HNSW/IVF query
throughput is latency-bound and moves most, SQL/Parquet scans are bandwidth-bound
and move less, and the discrete-GPU rows in §1b do not move at all. A swapped
memory kit or a new undervolt therefore invalidates the whole comparison, not one
column of it.

The rule that follows: **numbers are comparable only within one session**, and a
session is the set of sections measured in one machine state. A partial re-run
cannot be spliced into an older report.

The suite records the state for you. Before/after each section of a sweep:

```bash
python benchmarks/session_provenance.py capture --section vector \
    --artifact competitors/results/vector_matrix.json
```

That writes `benchmarks/results/session.json` with:

- a **fingerprint** — CPU model, threads, total RAM, DIMM speed/timings
  (`dmidecode -t memory`; needs root, otherwise recorded as unknown rather than
  guessed), governor, frequency driver and limits, RAPL package limit, kernel,
  and the engine version from `Cargo.toml`;
- **calibration probes** — random-gather latency (ns/access), triad bandwidth
  (GB/s), and a fixed fp64 matmul (GFLOPS), sampled once per stamp. The first and
  last sample bracket the sweep, so thermal or power-limit drift *inside* a session
  is visible too;
- a **section → session** map, which `generate_summary.py` enforces.

`generate_summary.py` **refuses to write the report** when stamped sections come
from different sessions (exit 2, listing them). `--allow-mixed-sessions` emits a
historical report that labels them instead. Fingerprints are compared on the
fields that change a number — CPU, memory, kernel, governor/frequency limits, DIMM
config, engine version — never on hostname, PID or timestamps.

Two practical consequences:

- After changing RAM timings or the CPU voltage/frequency curve, re-run the
  **whole** sweep, competitors included. Re-running only BenoStreamDB produces a
  table that cannot be read.
- Compare versions by interleaving builds on the *current* state (A/B/A/B), not
  by diffing a fresh run against stored artifacts from an older state.

Calibration numbers are not results. They exist so a later reader can tell "this
column moved because memory latency moved 12%" from "this column moved because
the code changed".

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

**Like-for-like rule & Iso-Recall evaluation.** 
1. **Index-Only Isolation:** Competitor adapters return vector IDs only (no payload
   fetch). The rollup therefore reports BenoStreamDB's **pure-index** QPS
   (`pure_index_qps`), not its full search+row-fetch number, and LanceDB projects
   only `id` in its search. Comparing BenoStreamDB's full-search QPS against a
   competitor's search-only QPS conflates index traversal with columnar Parquet
   decoding and PyArrow RecordBatch construction (§4.1). Full end-to-end query
   latency is measured separately in §3 and §6.
2. **Iso-Recall Tiering (Option A Parity):** ANN search speed is parameterized by
   beam-search depth (`ef_search`). To prevent an engine with lower recall from
   appearing artificially faster, evaluations are normalized across two standard
   operational tiers: **High-Throughput Tier** ($\text{Recall}@10 \ge 0.95$, candidate
   generation) and **High-Fidelity Tier** ($\text{Recall}@10 \ge 0.99$, top-$k$ ground
   truth). Within a tier, engines perform equivalent exploration work per query;
   cross-tier comparisons are evaluated strictly as $(\text{Recall}, \text{QPS})$ Pareto pairs.

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

### 5.1 NeurIPS BigANN (leaderboard-comparable)

The BigANN ablation runs under the challenge's reference envelope — Azure
Standard_D8lds_v5, **8 vCPU / 16 GiB** — because that envelope determines the
cache budgets, not just the thread count:

```bash
maturin build --release --out dist
docker build -f benchmarks/Dockerfile.neurips -t bsdb-neurips .
DATASET_DIR=/path/to/bigann \
  docker compose -f benchmarks/docker-compose.neurips.yml run --rm neurips
```

[`run_neurips_bigann.py`](../benchmarks/run_neurips_bigann.py) reads the cgroup
limits itself (`--hardware auto`) and derives:

| Knob | Derivation |
| :--- | :--- |
| query threads | the CPU limit (env `--cpus`) |
| `BSDB_CACHE_GB` | half the memory limit |
| preload budget | the same half, capped |
| `BSDB_DISK_CACHE` | off — a local `file://` table is served in place, never copied |

`--hardware d8lds --strict-hardware` pins the reference envelope and **aborts**
if the container is not actually constrained, so an oversized host run cannot
silently produce a number that looks comparable but is not.

The run is staged so a large corpus is ingested once and only the index is
rebuilt per quantizer:

```bash
# ingest once (no index), then 3 reindexes + the Pareto grid, in one process
python benchmarks/run_neurips_bigann.py --dataset 10m --stage both --all-modes --sweep

# 100M/1B: ingest in one job, reuse the table for every later run
python benchmarks/run_neurips_bigann.py --dataset 100m --stage ingest
python benchmarks/run_neurips_bigann.py --dataset 100m --stage query --all-modes --sweep
```

`--sweep` walks `--ef-sweep` × `--n-probe-sweep` (default
`32,64,128,256` × `4,8,16,32`) against the **same warm index**, and prints the
fastest operating point that clears recall ≥ 0.90 / 0.95 / 0.98 — the row the
leaderboard compares against.

Results are written to [`benchmarks/results/neurips_bigann/`](../benchmarks/results/neurips_bigann):

- `<tag>.json` — every operating point with full provenance: dataset paths, the
  ground-truth source, engine version + git revision, the envelope, and the grid.
- `<tag>.md` — the Pareto table and the fastest point clearing each recall bar,
  ready to paste into the paper.
- `bigann_ledger.jsonl` — append-only, one line per run, so numbers can be
  compared across engine versions.

`BenoGraph/paper` reads the results from here rather than keeping a second copy.

Two caveats when reading the ablation:

- `n_probe` must scale with the corpus. The IVF list count is
  `sqrt(n)/5` (63 lists at 1M, 632 at 10M), so `--n-probe 8` is 12% of the lists
  at 1M but 1.3% at 10M — recall is dominated by list pruning, not by the
  quantizer, and all three quantizers return identical neighbours.
- At 8 bits over `uint8` data, SQ8/TQ8/BQ8 are all effectively lossless. The
  quantizers only separate above ~0.90 recall, so quote that region, not the
  coarse-filter end of the curve.

### 5.2 MuSiQue GraphRAG (multi-hop retrieval)

A standalone multi-hop GraphRAG benchmark, separate from the 8-section sweep.
It uses the exact corpus, OpenIE graph, and 1,000-question manifest that
Zilliz/HippoRAG publish, so the comparison is like-for-like.

```bash
# headline: engine-native two-stage retrieval, no hosted LLM
python3 benchmarks/run_musique_parity.py --mode f32 --reranker bge --rebuild \
  --corpus-json benchmarks/musique_corpus.json \
  --triplets-json benchmarks/hippo_openie.json \
  --sample-manifest benchmarks/zilliz_manifest.json

# architecture demo: Passages/Entities/Relations/Mentions tables
python3 benchmarks/run_musique_entity_rag.py --mode f32 --reranker bge --rebuild \
  --corpus-json benchmarks/musique_corpus.json \
  --triplets-json benchmarks/hippo_openie.json \
  --sample-manifest benchmarks/zilliz_manifest.json
```

Headline (1,000 questions, `bge-large-en-v1.5`, f32, BGE cross-encoder):

| Strategy | Micro Recall@5 |
|---|---:|
| Hop-1 vector search | 0.5600 |
| Native Personalized PageRank | 0.5520 |
| **Two-stage + BGE cross-encoder** | **0.5943** |

Zilliz's published Naive RAG baseline is 0.5803; the two-stage row clears it
with the whole loop in-engine (vector search + DataFusion graph traversal +
local cross-encoder), no hosted LLM and no per-token cost.

The three inputs are git-ignored and reproducible; download locations, the
entity-level results, and the full reproduce steps are in
[`benchmarks/MUSIQUE_PARITY.md`](../benchmarks/MUSIQUE_PARITY.md) and
[`benchmarks/MUSIQUE_RESULTS.md`](../benchmarks/MUSIQUE_RESULTS.md).

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

```bash
python benchmarks/session_provenance.py verify          # exit 1 on mixed sessions
python benchmarks/generate_summary.py                   # refuses on mixed sessions
python benchmarks/generate_summary.py --allow-mixed-sessions   # historical report
```

The generator writes the machine-state fingerprint, the calibration envelope and
a per-section session table into the report header, so a reader never has to
guess which state a number came from. Artifacts produced before provenance
stamping have no session: the report labels them accordingly and treats their
mtime table as the only provenance.

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
- **Read the session, not just the section.** Before comparing two numbers, check
  they share a session id in the report header. If they do not, the difference
  includes a machine-state term and is not a code result. Calibration probes
  quantify that term: a `random gather` or `triad` shift of more than ~5 % between
  sessions is enough to explain a visible QPS delta on memory-bound workloads.
- **A `powersave` governor is a confounder.** Check `governor` and the
  `policy`/`hw` frequency range in the header before treating a small delta as
  code. Pin the governor (and record it) for any run whose numbers will be
  published.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `libcugraph.so: cannot open shared object file` | `RLIMIT_AS` applied to a GPU engine | Fixed in `apply_envelope`; rebuild the base runner image (`docker compose ... build bench bench-gpu`) |
| Wheel fails to load on the runner (`GLIBC_2.4x not found`) | host-built wheel linked against newer glibc | use `--build-wheel` (manylinux_2_28) |
| `cudaErrorMemoryAllocation: out of memory` on a 50-series card | cu12 wheels have no sm_120 kernels | the GPU image installs `cugraph-cu13`; `RMM_POOL_SIZE=12GB` leaves headroom for the Wayland compositor |
| GPU image ships an old harness | `bench-gpu` is `FROM bsdb-bench-runner:latest` | `docker_bench.sh` builds `bench` before `bench-gpu` |
| `docker compose run` cannot see the GPU | Compose < 2.30 ignores `gpus: all` | upgrade Compose, or use `docker run --gpus all` |
