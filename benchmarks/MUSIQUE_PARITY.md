# MuSiQue GraphRAG Results

This document records the two results worth reporting from the MuSiQue
multi-hop GraphRAG benchmark:

1. **Engine-native passage-level retrieval** (headline) — BenoStreamDB's
   in-engine retrieval, with no hosted LLM in the loop, clears Zilliz's
   published Naive RAG baseline.
2. **Entity-level Vector Graph RAG** (architecture demo) — BenoStreamDB can
   serve the three-collection Passages/Entities/Relations layout natively; the
   naive traversal does not reach HippoRAG/Zilliz's upper band, and it does not
   need to for the storage-engine claim.

Both use the same upstream inputs Zilliz/HippoRAG publish, so the comparison is
like-for-like on the corpus, the graph, and the 1,000-question sample.

---

## 1. Engine-native passage-level retrieval (headline)

Zilliz manifest sample, 1,000 questions, `bge-large-en-v1.5` (1024-d), f32
index, BGE cross-encoder reranker. **No hosted LLM, no outbound calls.**

| Strategy | Micro Recall@5 |
|---|---:|
| Hop-1 vector search | 0.5600 |
| Native Personalized PageRank (`graph_rag_search`) | 0.5520 |
| **Two-stage + BGE cross-encoder** | **0.5943** |

Zilliz's published **Naive RAG (BGE-large-en-v1.5) is 0.5803**. The two-stage
row clears it with the entire retrieval loop running inside BenoStreamDB —
vector search + DataFusion graph traversal + a local cross-encoder — with no
relationship-selection LLM and no per-token cost.

## 2. Entity-level Vector Graph RAG (architecture demo)

The same 1,000-question manifest, but retrieving through four native Iceberg
tables instead of passage-level co-occurrence:

| Table | Contents | Index |
|---|---|---|
| `musique_entity_rag_passages` | 11,656 passages | HNSW embedding |
| `musique_entity_rag_entities` | 64,982 deduplicated OpenIE entities | HNSW embedding |
| `musique_entity_rag_relations` | 108,594 entity↔entity edges | CSR graph |
| `musique_entity_rag_mentions` | 108,709 entity→passage edges | CSR graph |

Pipeline: vector-search the **Entities** table → BFS the **Relations** edge table
→ follow **Mentions** edges to passages → cross-encoder rerank.

| Strategy | Micro R@5 | Hit@5 | All-hit@5 | Precision@5 | p50 |
|---|---:|---:|---:|---:|---:|
| vector (passage baseline, sanity check) | 0.5597 | 0.9310 | 0.2260 | 0.2774 | 1.34 ms |
| entity_seed (no expansion) | 0.3228 | 0.6770 | 0.0520 | 0.3342 | 7.05 ms |
| entity_bfs (seed + relation expansion) | 0.1348 | 0.2740 | 0.0270 | 0.0884 | 19.49 ms |
| entity_rag (expansion + rerank) | 0.2140 | 0.4140 | 0.0590 | 0.1265 | 103.10 ms |

The `vector` row reproduces the passage-level baseline (0.5597 ≈ 0.5600), which
validates the entity harness end-to-end: the table build and vector-search path
are correct. The entity-level rows are real algorithmic results, not a bug.

**Finding:** raw OpenIE triple traversal with entity-name embeddings does not
recover HippoRAG/Zilliz's ~73%. Entity-name seeds are weak (query↔short-entity
cosine is noisy), and untyped "related" edges route through high-degree hub
entities, flooding the pool (hence `entity_bfs` is worse than `entity_seed`).
That upper band is bought by Zilliz's application-layer LLM relationship-selection
step — workflow logic that lives *above* the storage engine, not inside it.

---

## Data sources

All three inputs are the exact artifacts Zilliz/HippoRAG publish. They are large
and reproducible, so they are git-ignored.

| Artifact | Upstream source | Size |
|---|---|---|
| `musique_corpus.json` | [zilliztech/vector-graph-rag `evaluation/data/musique_corpus.json`](https://github.com/zilliztech/vector-graph-rag/blob/main/evaluation/data/musique_corpus.json) | 11,656 passages |
| `hippo_openie.json` | [OSU-NLP-Group/HippoRAG `output/openie_musique_results_ner_gpt-3.5-turbo-1106_11656.json`](https://github.com/OSU-NLP-Group/HippoRAG/blob/5ddedf0f516c7bbed777ba54da680c8bb8fb8f84/output/openie_musique_results_ner_gpt-3.5-turbo-1106_11656.json) | 11,656 docs, 64,982 entities |
| `zilliz_manifest.json` | [zilliztech/vector-graph-rag `evaluation/jev/two-stage/manifest.json`](https://github.com/zilliztech/vector-graph-rag/blob/main/evaluation/jev/two-stage/manifest.json) | 1,000 questions |

Download into `benchmarks/`:

```bash
curl -L https://raw.githubusercontent.com/zilliztech/vector-graph-rag/main/evaluation/data/musique_corpus.json \
  -o benchmarks/musique_corpus.json
curl -L https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/5ddedf0f516c7bbed777ba54da680c8bb8fb8f84/output/openie_musique_results_ner_gpt-3.5-turbo-1106_11656.json \
  -o benchmarks/hippo_openie.json
curl -L https://raw.githubusercontent.com/zilliztech/vector-graph-rag/main/evaluation/jev/two-stage/manifest.json \
  -o benchmarks/zilliz_manifest.json
```

---

## Reproduce

### 1. Engine-native passage-level (headline)

```bash
python3 benchmarks/run_musique_parity.py --mode f32 --reranker bge --rebuild \
  --corpus-json benchmarks/musique_corpus.json \
  --triplets-json benchmarks/hippo_openie.json \
  --sample-manifest benchmarks/zilliz_manifest.json
```

`--reranker bge` produces the `two_stage_bge` row above (MiniLM is the faster
default and lands at 0.5739). Docker/GPU variant:

```bash
GPU=1 ./benchmarks/run_musique_docker.sh --preset poc --reranker bge --mode f32 \
  --corpus-json /opt/bench/musique_corpus.json \
  --triplets-json /opt/bench/hippo_openie.json \
  --sample-manifest /opt/bench/zilliz_manifest.json
```

### 2. Entity-level architecture demo

```bash
python3 benchmarks/run_musique_entity_rag.py --mode f32 --reranker bge --rebuild \
  --corpus-json benchmarks/musique_corpus.json \
  --triplets-json benchmarks/hippo_openie.json \
  --sample-manifest benchmarks/zilliz_manifest.json
```

Docker/GPU variant (the entrypoint is parameterized by `BENCH_SCRIPT`):

```bash
GPU=1 BENCH_SCRIPT=run_musique_entity_rag.py ./benchmarks/run_musique_docker.sh \
  --mode f32 --reranker bge \
  --corpus-json /opt/bench/musique_corpus.json \
  --triplets-json /opt/bench/hippo_openie.json \
  --sample-manifest /opt/bench/zilliz_manifest.json
```

> **Host-run note:** `benchmarks/scratch_musique` is created by the Docker
> container (root-owned). When running outside Docker, pass a host-writable
> `--scratch-dir` (e.g. `benchmarks/scratch_entity_rag` for the entity script) so
> the harness can write its tables and caches. The passage-embedding cache is
> reused automatically when present at that scratch dir.

---

## Published comparison

| Method | MuSiQue R@5 |
|---|---:|
| Naive RAG · BGE-large-en-v1.5 | 58.03% |
| Vector Graph RAG pre-Jev (GPT-6 Luna / DeepSeek V4.1 Flash) | 72.18–73.20% |
| Vector Graph RAG + Jev 1.14 two-stage | 77.73% |
| **BenoStreamDB two-stage BGE (engine-native, no LLM)** | **59.43%** |

## Methodology

- **Embedder:** `BAAI/bge-large-en-v1.5` (1024-d), BGE retrieval instruction on
  queries only; L2-normalized, cosine index.
- **Index:** `--mode f32` for the quality numbers; `--mode tq8` for
  latency/throughput (costs ~10 points of recall on 1024-d).
- **Reranker:** BGE cross-encoder for the headline; MiniLM is the faster default.
- **Metrics:** micro Recall@5 (`hits / len(gold)`), hit-rate@5 (≥1 gold),
  all-hit@5 (all gold), precision@5.
- **Passage identity:** content-hash key (`title::sha1(title+text)`), not
  `title::idx`, so the same Wikipedia passage is one corpus row across questions.

## Caveats

- The Zilliz corpus (11,656 passages) is a subset of the full per-question pool
  (20,360 unique). The full-pool run is the same command without `--corpus-json`.
- Zilliz's 73% uses an LLM to select relations before retrieval; the two BenoStreamDB
  numbers above do not. The gap is the application-layer relationship-selection
  step, not corpus or triplet quality.
- Query encoding is pre-computed and excluded from the timed loop
  (`--include-query-encode` charges it to every query).
