# MuSiQue Multi-Hop GraphRAG Benchmark & Parity Evaluation

This document details the official results and methodology from the **MuSiQue multi-hop GraphRAG benchmark**:

1. **Engine-Native Passage-Level Retrieval (Headline)**: BenoStreamDB's in-engine retrieval, with **no hosted LLM** in the loop, clears Zilliz's published Naive RAG baseline (**0.5943 vs. 0.5803 Micro Recall@5**).
2. **Entity-Level Vector Graph RAG (Architecture Demo)**: Demonstrates BenoStreamDB natively serving the three-collection Passages/Entities/Relations/Mentions layout directly over Apache Iceberg tables without requiring external vector or graph databases.

Both benchmarks utilize the exact upstream inputs published by Zilliz and HippoRAG, ensuring a like-for-like empirical comparison on the corpus, the knowledge graph, and the 1,000-question evaluation manifest.

---

## 1. Hardware & Resource Envelope

> **Hardware & Memory Distinction (MuSiQue vs. NeurIPS BigANN):**
> - **MuSiQue GraphRAG**: Constrained to **8 vCPUs / 32 GiB RAM** (`--memory=32g` in `benchmarks/docker-compose.musique.yml` and `benchmarks/run_musique_docker.sh`). The 32 GiB memory budget is strictly required because the runner hosts the local in-memory PyTorch runtime for `bge-large-en-v1.5` dense embeddings and the BGE cross-encoder reranker, alongside four active Iceberg tables.
> - **NeurIPS BigANN**: Constrained to **8 vCPUs / 16 GiB RAM** (Azure `Standard_D8lds_v5`). BigANN exercises raw columnar Parquet streaming and HNSW/IVF vector indexing without hosted transformer models.

| Parameter | MuSiQue GraphRAG Specification |
| :--- | :--- |
| **Reference Envelope** | 8 vCPUs, 32 GiB RAM (`deploy.resources.limits.memory: 32G`) |
| **Embedding Model** | `BAAI/bge-large-en-v1.5` (1,024 dimensions, Float32) |
| **Reranker Model** | `BAAI/bge-reranker-large` (Cross-encoder) |
| **Storage Tables** | Native Apache Iceberg (V2/V3) Parquet |
| **Index Overlays** | HNSW (`cosine` metric) + CSR Graph Indexes |
| **Host Execution** | Linux x86_64 / NVIDIA CUDA or Apple Silicon Metal |

---

## 2. Engine-Native Passage-Level Retrieval (Headline)

Evaluated on the official Zilliz manifest sample of 1,000 multi-hop questions using `bge-large-en-v1.5` (1,024 dimensions), Float32 vectors, and BGE cross-encoder reranking. **Zero hosted LLMs, zero outbound API calls, zero per-token cost.**

| Retrieval Strategy | Micro Recall@5 | Description |
|---|---:|---|
| Hop-1 Vector Search | 0.5600 | Direct top-$k$ dense vector search on passage embeddings. |
| Native Personalized PageRank (`graph_rag_search`) | 0.5520 | In-engine PPR traversal across passage co-occurrence graph. |
| **Two-Stage + BGE Cross-Encoder** | **0.5943** | Vector search + DataFusion graph expansion + local cross-encoder. |

**Comparison Baseline:**  
Zilliz's published **Naive RAG baseline (BGE-large-en-v1.5) is 0.5803**.  
BenoStreamDB's two-stage pipeline scores **0.5943 Micro Recall@5 (+1.40 points)** with the entire retrieval cycle executing in-process inside the database engine.

---

## 3. Entity-Level Vector Graph RAG (Architecture Demo)

The same 1,000-question manifest evaluated across four native Apache Iceberg tables rather than passage-level co-occurrence:

| Table Name | Contents & Schema | Index Type |
|---|---|---|
| `musique_entity_rag_passages` | 11,656 passages (`id`, `doc_key`, `title`, `text`, `embedding`) | HNSW vector embedding |
| `musique_entity_rag_entities` | 64,982 deduplicated OpenIE entities (`id`, `name`, `embedding`) | HNSW vector embedding |
| `musique_entity_rag_relations` | 108,594 entity↔entity edges (`source`, `target`, `relation`, `weight`) | CSR graph index |
| `musique_entity_rag_mentions` | 108,709 entity→passage edges (`source`, `target`, `relation`, `weight`) | CSR graph index |

### Retrieval Traversal Pipeline
1. Vector-search the **Entities** table for query entity seeds.
2. Breadth-First Search (BFS) the **Relations** edge table to discover related entities.
3. Follow **Mentions** edges from discovered entities to candidate passages.
4. Cross-encoder rerank the candidate passages.

| Strategy | Micro R@5 | Hit@5 | All-hit@5 | Precision@5 | Latency (p50) |
|---|---:|---:|---:|---:|---:|
| `vector` (passage baseline sanity check) | 0.5597 | 0.9310 | 0.2260 | 0.2774 | 1.34 ms |
| `entity_seed` (no relation expansion) | 0.3228 | 0.6770 | 0.0520 | 0.3342 | 7.05 ms |
| `entity_bfs` (seed + relation expansion) | 0.1348 | 0.2740 | 0.0270 | 0.0884 | 19.49 ms |
| `entity_rag` (expansion + BGE rerank) | 0.2140 | 0.4140 | 0.0590 | 0.1265 | 103.10 ms |

### Systems & Algorithmic Findings
1. **Passage Baseline Verification:** The `vector` row reproduces the passage baseline (`0.5597 ≈ 0.5600`), verifying the underlying Iceberg table build, Arrow FFI, and HNSW vector index paths.
2. **The LLM Upper-Band Trade-off:** Naive OpenIE triple traversal without relationship pruning does not match HippoRAG/Zilliz's ~73% upper band. Entity-name embeddings are short and noisy, and untyped edges cause hub-entity flooding. The upper band reported by HippoRAG requires an application-layer LLM relationship-selection filter—workflow logic that lives *above* the storage engine, not inside the database.

---

## 4. Upstream Data Sources

All inputs are identical to the published Zilliz and HippoRAG artifacts:

| Artifact | Source Repository / Path | Volume |
|---|---|---|
| `musique_corpus.json` | [zilliztech/vector-graph-rag](https://github.com/zilliztech/vector-graph-rag/blob/main/evaluation/data/musique_corpus.json) | 11,656 passages |
| `hippo_openie.json` | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG/blob/5ddedf0f516c7bbed777ba54da680c8bb8fb8f84/output/openie_musique_results_ner_gpt-3.5-turbo-1106_11656.json) | 64,982 entities |
| `zilliz_manifest.json` | [zilliztech/vector-graph-rag](https://github.com/zilliztech/vector-graph-rag/blob/main/evaluation/jev/two-stage/manifest.json) | 1,000 questions |

Download commands:

```bash
curl -L https://raw.githubusercontent.com/zilliztech/vector-graph-rag/main/evaluation/data/musique_corpus.json \
  -o benchmarks/musique_corpus.json

curl -L https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/5ddedf0f516c7bbed777ba54da680c8bb8fb8f84/output/openie_musique_results_ner_gpt-3.5-turbo-1106_11656.json \
  -o benchmarks/hippo_openie.json

curl -L https://raw.githubusercontent.com/zilliztech/vector-graph-rag/main/evaluation/jev/two-stage/manifest.json \
  -o benchmarks/zilliz_manifest.json
```

---

## 5. Reproduction Harness

### 1. Engine-Native Passage-Level Retrieval (Headline, 0.5943)

```bash
python3 benchmarks/run_musique_parity.py --mode f32 --reranker bge --rebuild \
  --corpus-json benchmarks/musique_corpus.json \
  --triplets-json benchmarks/hippo_openie.json \
  --sample-manifest benchmarks/zilliz_manifest.json
```

### 2. Entity-Level Architecture Demo

```bash
python3 benchmarks/run_musique_entity_rag.py --mode f32 --reranker bge --rebuild \
  --corpus-json benchmarks/musique_corpus.json \
  --triplets-json benchmarks/hippo_openie.json \
  --sample-manifest benchmarks/zilliz_manifest.json
```

### 3. Docker Containerized Reproduction (32 GiB Enforced)

```bash
# Build the benchmark image
docker build -f benchmarks/Dockerfile.musique -t bsdb-musique .

# Execute under strict cgroup limits (32G memory)
docker compose -f benchmarks/docker-compose.musique.yml run --rm musique
```
