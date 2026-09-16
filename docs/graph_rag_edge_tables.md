# Graph RAG & Iceberg Edge Tables Guide

HyperStreamDB provides native graph analytics and edge-level semantic search directly over Apache Iceberg edge tables. This architecture eliminates the need to run separate vector databases (e.g. Pinecone/Milvus) alongside dedicated graph databases (e.g. Neo4j) or Spark GraphX clusters.

---

## 1. Standard Edge Table Schema Convention

An edge table in HyperStreamDB represents relationships between entities. To guarantee high performance across graph analytics UDFs and vector search, edge tables follow this standard convention:

| Column Name | Arrow / Iceberg Data Type | Nullable | Description |
|---|---|---|---|
| `source` | `UInt64` (or `String`) | **No** | Origin node / entity identifier. Using integer IDs provides maximum speed in graph traversals. |
| `target` | `UInt64` (or `String`) | **No** | Destination node / entity identifier. |
| `relation` | `Utf8` / `String` | Yes | Relationship predicate (e.g. `'cites'`, `'invested_in'`, `'acquired'`, `'mentions'`). |
| `weight` | `Float64` / `Float32` | **No** | Edge weight / strength (defaults to `1.0`). |
| `embedding` | `FixedSizeList<Float32, dim>` | Yes | Dense semantic embedding vector describing the relationship, interaction context, or anchor text. |

### Creating an Edge Table

You can create a standardized edge table with automated indexing using `Table.create_edge_table`:

```python
import hyperstreamdb as hs

# Create an edge table with automated sidecar index generation
edge_table = hs.Table.create_edge_table(
    uri="file:///data/knowledge_graph/edges",
    node_id_type="uint64",          # 'uint64' (optimal for Graph UDFs) or 'string'
    embedding_dim=384,               # Vector dimension for edge semantics
    with_relation=True,              # Include 'relation' string column
    with_weight=True,                # Include 'weight' float column
    index_endpoints=True,            # Auto-generate Roaring Bitmap sidecars on source & target
    index_embedding=True,            # Auto-generate HNSW index on embedding column
    partition_by_relation=False      # Optional: partition Iceberg data files by relation type
)
```

---

## 2. Automated Sidecar Indexing

Graph traversals and vector searches require distinct physical acceleration structures:

### A. Endpoint Roaring Bitmaps (`source` and `target`)
- When `index_endpoints=True` is specified, HyperStreamDB automatically configures **Roaring Bitmap sidecar indexes** on the `source` and `target` columns.
- **Why**: Allows $O(1)$ neighborhood lookups and instantaneous candidate filtering during breadth-first search (BFS) and induced subgraph extraction without table scanning.

### B. HNSW Vector Index on `embedding`
- When `index_embedding=True` is specified and an `embedding` column exists, HyperStreamDB automatically configures an **HNSW vector index** on the relationship vectors.
- **Why**: Enables semantic similarity search directly over relationships (e.g. finding relationships describing "mergers and acquisitions" or "patent disputes").

---

## 3. NetworkX Interoperability

HyperStreamDB provides complete bi-directional interoperability with NetworkX, supporting all graph types: `Graph`, `DiGraph`, `MultiGraph`, and `MultiDiGraph`.

### Ingesting from NetworkX (`Table.from_networkx`)

```python
import networkx as nx
import hyperstreamdb as hs

# Build or load a graph in NetworkX
G = nx.DiGraph()
G.add_edge(1, 2, relation="cites", weight=2.5, paper_id="arXiv:2401.001")
G.add_edge(2, 3, relation="co_author", weight=1.0)
G.add_edge(3, 1, relation="mentions", weight=0.8)

# Ingest directly into an Iceberg edge table with sidecar indexes
table = hs.Table.from_networkx(
    uri="file:///data/graphs/citation_graph",
    graph=G,
    index_endpoints=True
)
table.commit()
```

### Exporting to NetworkX (`table.to_networkx`)

You can export any HyperStreamDB edge table back into a NetworkX graph while preserving all edge attributes and multi-edges:

```python
# Export to a directed NetworkX graph
G_out = table.to_networkx(directed=True, multigraph=False)

assert G_out.has_edge(1, 2)
assert G_out[1][2]["relation"] == "cites"
assert G_out[1][2]["weight"] == 2.5
```

---

## 4. End-to-End Hybrid Graph RAG Workflow

Graph RAG combines semantic vector search with structural graph context to retrieve coherent subgraphs rather than isolated text chunks.

### Step 1: Semantic Edge Search
Find the most relevant edges in vector space matching the user's question:

```python
query_vector = [0.12, -0.45, 0.88, ...] # 384-dimensional query embedding

# Search edge embeddings for relevant relationships
matched_edges = table.search("embedding", query_vector, k=5)
# matched_edges contains: source, target, relation, weight, distance
```

### Step 2: Seed Node Extraction
Extract the endpoint nodes of the top semantic relationships:

```python
seed_nodes = list(set(matched_edges["source"]).union(set(matched_edges["target"])))
```

### Step 3: Multi-Hop Subgraph Extraction
Extract the multi-hop induced subgraph surrounding the semantic seeds:

```python
# Extract 2-hop induced subgraph starting from seed nodes
subgraph_edges = table.subgraph(seed_nodes, hops=2, directed=False).to_pandas()
```

### Step 4: Personalized PageRank Grounding
Rank nodes in the extracted context by their structural relevance to the seed entities:

```python
# Compute Personalized PageRank with teleportation biased toward the seed entities
pagerank_scores = table.personalized_pagerank(
    seeds=seed_nodes,
    alpha=0.85,
    max_iter=30
).to_pandas()
```

### Step 5: Connecting Paths for Multi-Entity Reasoning
If the query mentions two or more distinct entities, find all connecting paths between them:

```python
paths = table.connecting_paths(
    seeds=[entity_a_id, entity_b_id],
    max_depth=3,
    directed=False
).to_pandas()
```

---

## 5. Iceberg Partitioning & Optimization Best Practices

1. **Node ID Types**:
   - Prefer `uint64` for `source` and `target` when performance is critical. Integer comparisons in DataFusion graph accumulators avoid string allocation overhead.
   - For heterogeneous entity graphs (e.g. `user:101`, `product:502`), map string entity IDs to 64-bit integer hashes or maintain a dictionary mapping table.

2. **Partitioning by Relation**:
   - If your graph has millions of edges with distinct predicates (`partition_by_relation=True`), HyperStreamDB partitions data files into directories by `relation`.
   - Queries filtering on specific edge types (e.g. `WHERE relation = 'cites'`) prune irrelevant data files at the manifest level before scanning Parquet files.

3. **Combined BM25 + Vector Search**:
   - For edge tables with descriptive text in `relation` or `summary` columns, combine lexical matching with vector search using the REST gateway (`port 9200`), DataFusion SQL, or `table.hybrid_search(...)`.

---

## 6. End-to-End Graph RAG Search & Community Summarization

HyperStreamDB provides full native pipeline integration connecting vector document retrieval, knowledge graph traversal, and topological relevance scoring:

### Local Graph RAG Search
Local search finds seed entities via vector search, expands the neighborhood via an induced multi-hop subgraph, ranks entities using Personalized PageRank (PPR), and formats the context into prompt-ready markdown for an LLM:

```python
# Query document table using embedding vector + edge table graph topology
result = doc_table.graph_rag_search(
    query=query_embedding,       # List[float] dense embedding
    edge_table=edge_table,       # Edge table containing relational edges
    mode="local",                # 'local' or 'global'
    hops=2,                      # 2-hop induced subgraph
    top_k=5,                     # Initial seed count
    alpha=0.85,                  # Personalized PageRank restart probability
    allowed_relations=["invested_in", "supplies"],  # Filter out hub noise like 'mentioned_in'
    search_edges=True            # Dual Vector-Graph RAG: search edge embeddings in parallel
)

# Inspect retrieved entities (ranked by PageRank score) and induced edges
print(result.nodes)   # pandas.DataFrame with columns: id, title, content, pagerank
print(result.edges)   # pandas.DataFrame with columns: source, target, weight

# Format prompt-ready markdown for direct LLM injection
llm_context = result.format_context(max_tokens=2000)
```

### HippoRAG-Style Continuous Seed-Weighted Personalized PageRank
Unlike standard uniform seed PPR where teleportation probability is split uniformly across seeds ($p_0(v) = 1/|S|$), HyperStreamDB supports HippoRAG-style continuous similarity weighting ($p_0(v) \propto w(v)$):

```python
# Pass continuous relevance weights from vector similarity scores or reciprocal distances
pagerank_scores = edge_table.personalized_pagerank(
    seeds=[10, 20],
    seed_weights=[0.92, 0.15],   # Modulates teleportation mass toward node 10
    damping=0.85,
    iterations=30
).to_pandas()
```

### Predicate & Relation Pushdown in Subgraphs
Eliminate high-degree hub noise (e.g. `'mentioned_in'`, `'contains'`) by restricting traversal to high-signal predicates:

```python
# Extract 2-hop induced subgraph traversing ONLY 'founded' and 'invested_in' edges
subgraph_df = edge_table.subgraph(
    seeds=[1, 2],
    hops=2,
    allowed_relations=["founded", "invested_in"]
).to_pandas()
```

### Dual Vector-Graph RAG (Semantic Edge Search)
Query relationship edge embeddings in parallel with entity doc embeddings to uncover relation-phrase seeds (e.g. "patent licensing agreement between chipmakers"):

```python
result = doc_table.graph_rag_search(
    query=query_embedding,
    edge_table=edge_table,
    mode="local",
    search_edges=True,            # Searches edge embeddings simultaneously
    edge_vector_column="embedding"
)
```

### Hierarchical Community Summarization Workflow (Global Search)
Generate multi-level community trees across Louvain resolution pyramids (`level`, `parent_community_id`) for Microsoft GraphRAG parity:

```python
# Materialize multi-level hierarchical community pyramid as an Iceberg table
comm_table = edge_table.summarize_communities(
    doc_table=doc_table,
    target_uri="file:///tmp/hierarchical_communities",
    hierarchical=True,
    resolutions=[0.5, 1.0, 2.0],  # Macro (L0) -> Meso (L1) -> Micro (L2)
    top_entities_per_comm=5
)

# Global Graph RAG search over hierarchical community tree
result = doc_table.graph_rag_search(
    query=query_embedding,
    edge_table=edge_table,
    mode="global",
    community_table=comm_table,
    top_k=3
)

print(result.format_context())
```

### Entity Equivalence Resolution (`Table.resolve_entities`)
Automatically resolve synonyms, aliases, and entity duplicates across `same_as` edges using transitive equivalence closure:

```python
# Compute transitive alias closure over 'same_as' edges
mapping = edge_table.resolve_entities(relation="same_as")
# Returns: {10: 10, 20: 10, 30: 10, 40: 40, 50: 40}
```

### Reciprocal Rank Fusion (RRF) Hybrid Search
Combine dense vector search and BM25 text keyword search into a fused ranked result:

```python
results = doc_table.hybrid_search(
    text_column="content",
    query_text="semiconductor GPU foundry",
    vector_column="embedding",
    query_vector=query_embedding,
    k=10,
    rrf_k=60
)
```

---

## 7. Analytics Engineering with dbt (`dbt-hyperstreamdb`)

The official `dbt-hyperstreamdb` adapter plugin provides native Jinja macros for all Graph UDFs, allowing analytics engineers to model, transform, and materialize graph analytics tables into Apache Iceberg directly within dbt projects:

```sql
-- models/entity_pagerank.sql
{{ config(materialized='table') }}

{{ pagerank(ref('knowledge_graph_edges'), damping=0.85, iterations=30) }}
```

### Supported dbt Graph Macros:

| Macro | Description |
|---|---|
| `{{ pagerank(ref('edges'), damping=0.85, iterations=30) }}` | Iterative PageRank node importance scores |
| `{{ personalized_pagerank(ref('edges'), seeds=[...], seed_weights=[...]) }}` | HippoRAG-style continuous seed-weighted PPR |
| `{{ community_detect(ref('edges'), algorithm='louvain', resolution=1.0) }}` | Louvain or Label Propagation community clusters |
| `{{ graph_neighbors(ref('edges'), entity_id, hops=2) }}` | N-hop neighborhood expansion |
| `{{ subgraph(ref('edges'), seeds=[...], hops=1) }}` | Multi-hop induced subgraph extraction |
| `{{ connecting_paths(ref('edges'), seeds=[...]) }}` | Pairwise shortest connecting paths between seeds |
| `{{ shortest_path(ref('edges'), start, end) }}` | Shortest path node sequences |
| `{{ connected_components(ref('edges'), directed=false) }}` | Weakly / strongly connected components |
| `{{ degree_centrality(ref('edges')) }}` | In/out/total degree centrality distribution |
| `{{ node_similarity(ref('edges'), node_a, node_b, method='jaccard') }}` | Link prediction scores (Jaccard, Adamic-Adar, etc.) |
| `{{ topological_sort(ref('edges')) }}` | Directed acyclic graph execution ordering |


