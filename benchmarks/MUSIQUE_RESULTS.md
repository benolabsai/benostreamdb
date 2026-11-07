# MuSiQue GraphRAG — Reported Results

Two results worth reporting from the MuSiQue multi-hop GraphRAG benchmark.
Both use the same upstream inputs Zilliz/HippoRAG publish, on the identical
1,000-question manifest sample.

## 1. Engine-native passage-level retrieval (headline)

BenoStreamDB's in-engine retrieval, with **no hosted LLM and no outbound calls**,
clears Zilliz's published Naive RAG baseline.

| Strategy | Micro Recall@5 |
|---|---:|
| Hop-1 vector search | 0.5600 |
| Native Personalized PageRank (`graph_rag_search`) | 0.5520 |
| **Two-stage + BGE cross-encoder** | **0.5943** |

Zilliz **Naive RAG (BGE-large-en-v1.5): 0.5803**.

## 2. Entity-level Vector Graph RAG (architecture demo)

Four native Iceberg tables serve the three-collection layout:

| Table | Contents | Index |
|---|---|---|
| `passages` | 11,656 passages | HNSW embedding |
| `entities` | 64,982 deduplicated OpenIE entities | HNSW embedding |
| `relations` | 108,594 entity↔entity edges | CSR graph |
| `mentions` | 108,709 entity→passage edges | CSR graph |

Pipeline: vector-search **Entities** → BFS **Relations** → follow **Mentions** →
cross-encoder rerank.

| Strategy | Micro R@5 | Hit@5 | All-hit@5 | Precision@5 |
|---|---:|---:|---:|---:|
| vector (baseline sanity check) | 0.5597 | 0.9310 | 0.2260 | 0.2774 |
| entity_seed (no expansion) | 0.3228 | 0.6770 | 0.0520 | 0.3342 |
| entity_bfs (seed + expansion) | 0.1348 | 0.2740 | 0.0270 | 0.0884 |
| entity_rag (expansion + rerank) | 0.2140 | 0.4140 | 0.0590 | 0.1265 |

**Finding:** naive OpenIE triple traversal does not reach HippoRAG/Zilliz's
~73%. That upper band comes from their application-layer LLM
relationship-selection step — workflow logic that lives *above* the storage
engine, not inside it.

## Reproduce

```bash
# 1. Engine-native passage-level (headline, 0.5943)
python3 benchmarks/run_musique_parity.py --mode f32 --reranker bge --rebuild \
  --corpus-json benchmarks/musique_corpus.json \
  --triplets-json benchmarks/hippo_openie.json \
  --sample-manifest benchmarks/zilliz_manifest.json

# 2. Entity-level architecture demo
python3 benchmarks/run_musique_entity_rag.py --mode f32 --reranker bge --rebuild \
  --corpus-json benchmarks/musique_corpus.json \
  --triplets-json benchmarks/hippo_openie.json \
  --sample-manifest benchmarks/zilliz_manifest.json
```

Host runs need a host-writable `--scratch-dir` (`benchmarks/scratch_musique` is
root-owned by the Docker container). Docker/GPU variants and full data-source
provenance are documented in [`MUSIQUE_PARITY.md`](MUSIQUE_PARITY.md).

## Positioning

- **Headline:** engine-native two-stage BGE **0.5943 > Zilliz Naive RAG 0.5803**,
  zero hosted LLM, no per-token cost.
- **Secondary:** BenoStreamDB natively serves the three-collection Vector Graph
  RAG layout; the ~73% band is application-layer LLM relationship selection, not
  the storage engine.
