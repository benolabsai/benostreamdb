or # 06. Roadmap & Execution Milestones

**Status:** Technical & Commercial Execution Roadmap  
**Target:** 2026–2027  
**Alignment:** Directly synchronized with `ROADMAP.md` (Phases 1–12)  

---

## 1. Phased Executive Timeline (Current Position: Late Q3 2026)

BenoStreamDB development began in **December 2025**. Over the past 9 months through late **Q3 2026**, the core engine foundation (Phases 1–8) was completed, tested, and released (v0.7.0).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      REAL-WORLD EXECUTION TIMELINE                          │
├──────────────┬───────────────────────────────┬──────────────────────────────┤
│ TIMELINE     │ STATUS & KEY MILESTONES       │ DELIVERABLES                 │
├──────────────┼───────────────────────────────┼──────────────────────────────┤
│ Dec 2025 –   │ ✅ COMPLETED FOUNDATION       │ • 71,880 Lines of Pure Rust  │
│ Aug 2026     │ • Core Engine v0.7.0 Release  │ • Apache Iceberg V2/V3 Engine│
│ (Q1–Q2 2026) │ • 3 Wire Gateways Live        │ • Flight SQL, 9200, 6333     │
│              │ • TurboQuant (TQ4/TQ8) Kernels│ • RoaringBitmap Sidecar .idx │
├──────────────┼───────────────────────────────┼──────────────────────────────┤
│ Q3 2026      │ ⏳ CURRENT FOCUS: v0.8.0      │ • BenoStreamDB v0.8.0 Launch│
│ (Sept 2026)  │ • v0.8.0 Native Graph RAG     │ • Graph Edge Tables & Indexes│
│              │ • DataFusion Graph UDFs       │ • PageRank & Louvain UDFs    │
│              │ • Self-paced engineering      │ • GRAPH_RAG_SEARCH Engine    │
│              │ • (Off PhD Fall '26: TBI Rec) │ • Prerequisite for EDGAR     │
├──────────────┼───────────────────────────────┼──────────────────────────────┤
│ Q4 2026      │ 🎯 NEXT QUARTER               │ • Public PyPI & GTM Launch   │
│ (Oct–Dec)    │ • Phase 12: Public Release    │ • Sub-2 TB S3 2014–26 Corpus │
│              │ • 2014–2026 S3 Backfill Done  │ • First Beta Quant Desks     │
│              │ • Self-paced engine prep      │ • Turn-key data for Jan '27  │
├──────────────┼───────────────────────────────┼──────────────────────────────┤
│ Q1 2027      │ 🎓 RESUME PHD & EXPANSION     │ • Formal Return to PhD (Jan) │
│ (Jan–Mar)    │ • Resume PhD Research         │ • Carhart + ISF Research Run │
│              │ • Enterprise Security Tier    │ • Sidecar-level RLS & CMEK   │
│              │ • Customer-Funded Add-on      │ • First Enterprise Logos     │
├──────────────┼───────────────────────────────┼──────────────────────────────┤
│ Q2 2027+     │ 🌐 CLOUD SCALE                │ • "BenoStream Cloud" Control│
│              │ • Managed Serverless Offering │ • Series A Institutional Cap │
└──────────────┴───────────────────────────────┴──────────────────────────────┘
```

---

## 2. Completed Foundation (Phases 1–8) ✅ Verified in Code

All foundational data systems capabilities are verified and operating:
* **Phase 1: Real-World Dataset Benchmarks**: NYC Taxi (753k rows/s ingest, 85ms selective query), Wikipedia 100k docs (14ms projected scalar search), 100k 768D BERT embeddings.
* **Phase 2: Nessie REST Catalog Integration**: Git-like table branching and multi-table transactions.
* **Phase 3 & 3.5: Performance & Native DataFusion SQL Engine**: MoR/CoW deletion vectors, partition pruning, Index Nested Loop Joins, and pgvector operators (`<->`, `<=>`, `<#>`).
* **Phase 4.5: Multi-Catalog Abstraction**: REST, AWS Glue, Hive Metastore, and Databricks Unity catalogs.
* **Phase 5: Connectors & Distributed Analytics**: Spark DataSource V2, Trino SPI connector, and split-level byte-range parallelism.
* **Phase 6: Operational Tooling & Observability**: `bsdb` CLI REPL, `tracing-opentelemetry`, Prometheus metrics exporter (`/metrics`).
* **Phase 6.5: Ecosystem Gateways**: Dual search gateway (`benostreamdb-search` on Ports 9200 & 6333), Arrow Flight SQL gateway (`benostreamdb-flight` on Port 50051), and `dbt-benostreamdb` adapter.
* **Phase 7: Cloud-Agnostic Concurrency & Durability**: `FileBasedLock` using object storage CAS (`PutMode::Create`), OCC snapshot swaps with retries, and chaos testing (`tests/test_chaos.rs`).
* **Phase 8: Documentation & Packaging Suite**: Sphinx / ReadTheDocs configuration in `docs/` covering SQL, Python, Iceberg V2/V3, GPU, and Concurrency. `maturin` / `pyo3` Python wheel compilation is fully functional (CPython 3.10–3.14 `abi3` wheels generated for v0.6.0 and earlier; unmarketed baseline).

---

## 3. Active & Upcoming Roadmap Backlog (Phases 9–12)

The active development backlog directly maps the core engine milestones into commercial capabilities:

### Phase 9: Resource-Constrained Vector Benchmarking (4 GB RAM Matrix) ⏳ ACTIVE
* **Objective**: Prove sustained vector ingestion and sub-second hybrid query latency under strict container limits (`docker run --memory=4g --cpus=4`).
* **Commercial Value**: Directly undercuts OpenSearch and LanceDB by proving BenoStreamDB operates in production under 4 GB RAM without JVM crashes or OOM kills.
* **Concrete Tasks**:
  - [ ] Reproducible Docker benchmark harness comparing BenoStreamDB (TQ8/TQ4) vs OpenSearch 2.x/3.x and LanceDB.
  - [ ] Automated measurement of RSS memory ceilings, ingest throughput (vectors/sec), and p95/p99 query latency.
  - [ ] Formal LRU cache budget configuration guide (`BENOSTREAM_CACHE_CAP_BYTES`) ensuring predictable memory bounds on edge/container hosts.

### Phase 10: Streaming Commit, Delete Lifecycle & Concurrency Verification ⏳ ACTIVE
* **Objective**: Formally verify HNSW index overlay stability across immutable Iceberg snapshot commits, partition splits, and position/equality deletes.
* **Commercial Value**: Guarantees enterprise ACID compliance and GDPR/CCPA hard deletion support directly on object storage without rewriting entire Parquet tables.
* **Concrete Tasks**:
  - [ ] Integration test suite for Iceberg V2 position delete masking in vector graph scans (`tests/verify_mor_vector_deletes.rs`).
  - [ ] Incremental sidecar index append vs. compaction coordination under concurrent streaming writes.
  - [ ] Architecture documentation detailing the interaction between persistent HNSW overlays and Iceberg transaction manifests.

### Phase 11: High-Cardinality Metadata & Temporal Partitioning Validation (SEC EDGAR & EdgarStreamDB) ⏳ ACTIVE
* **Re-scope Note (2026-09-22)**: The raw *scale-testing* objective is now satisfied by the full-site Wikipedia Graph RAG demo (51.8M pages / 383M edges, HNSW-TQ8 + BM25, CSR graph, two-level Graph RAG) — a larger structured dataset than 12 years of SEC EDGAR. Phase 11 is narrowed to the access patterns the demo does not cover, and retained as the vertical commercial lighthouse + PhD research platform.
* **Objective**: Validate **high-cardinality metadata pre-filtering** (CIK, SIC, filing dates) + RoaringBitmap/TQ8 HNSW under a 4 GB node, and hybrid scalar-vector queries under **high data skew and temporal partitioning** — on 12+ years of SEC EDGAR filings (Form 4 XML insider transactions, 10-K/10-Q text, 13F holdings).
* **Commercial Value**: Powers EdgarStreamDB dogfooding and EdgarStream Wagtail frontend; delivers immediate cash flow from quants ($500–$1,500/mo) and institutional funds ($30k–$50k/yr). Also the empirical platform for the Carhart + ISF dissertation research.
* **Concrete Tasks**:
  - [ ] Reference benchmark implementation in `examples/sec_edgar_scale_test.md`.
  - [ ] Zero-copy PyTorch tensor feeding via Arrow Flight SQL gateway for deep learning feature pipelines.
  - [ ] End-to-end verification of hybrid scalar-vector queries under high data skew and temporal partitioning.
  - [ ] High-cardinality scalar pre-filtering benchmark (CIK / SIC / filing-date predicates) against RoaringBitmap sidecars.

### Phase 12: Public PyPI Release, GTM Marketing & AI Frameworks ⏳ PLANNED
* **Objective**: Transition existing private `maturin` wheel pipeline into public distribution on PyPI and drive developer mindshare. *(Note: Linux `x86_64` CPython 3.10–3.14 wheels already compile cleanly in-repo; this milestone focuses on public release and marketing).*
* **Commercial Value**: Top-of-funnel developer adoption to drive community installations $\rightarrow$ Team tier conversion.
* **Concrete Tasks**:
  - [ ] Publish official v0.8.0 binary wheels on PyPI (`pip install benostreamdb`) for Linux (x86_64) and macOS (Apple Silicon via personal Mac).
  - [ ] Developer marketing push across Hacker News, r/dataengineering, and DuckDB/Iceberg developer communities.
  - [ ] Official LangChain vector store integration (`BenoStreamVectorStore`).
  - [ ] Official LlamaIndex vector store integration (`BenoStreamIndexStore`).

---

## 4. Detailed Engineering Backlog (From `ROADMAP.md`)

### 4.1 Connector & Pushdown Enhancements
- [ ] **Out-of-Core Index Ingestion**: Rework HNSW and inverted index building to use out-of-core (on-disk) processing and incremental batching. Allows ingesting terabytes of data directly via the core Rust library without OOM errors, while maintaining Spark distributed ingestion support. (Target: v0.8.0)
- [ ] **HNSW Hot Cache Optimization**: Update `IndexFileCache` to store fully deserialized `Arc<Hnsw>` graphs in memory rather than raw `Vec<u8>` bytes. Eliminates per-query deserialization overhead and brings kNN latency down to ~3–5ms (on par with OpenSearch). (Target: v0.8.0)
- [ ] **Trino Connector Sidecar Pushdown**: Enhance `trino-benostream` SPI implementation to evaluate filter predicates directly against sidecar `.hnsw` and `.idx` files before scanning parquet splits.
- [ ] **Micro-Batch Streaming Ingest Buffer**: Native 5–30s Iceberg snapshot buffer for streaming ingestion from Kafka and Kinesis.

### 4.2 Advanced Search & Query Features
- [ ] **Zero-Copy Arrow IPC Vector Index**: Completely rewrite internal HNSW graph implementation to traverse columnar Apache Arrow IPC structures instead of Rust heap pointers. Enables true zero-copy memory mapping and native ecosystem interoperability with Spark/Trino for vector search. (Target: v0.8.0)
- [ ] **Async Ingest Memory Buffer & WAL**: Re-architect `_bulk` ingestion to buffer documents in memory and flush asynchronously via a Write-Ahead Log (WAL), removing the synchronous disk fsync bottleneck. (Target: v0.8.0)
- [x] **TurboQuant™ Core Quantization**: Built-in scalar quantization (TQ4 / TQ8 with Fast Walsh-Hadamard Transform) for 4x memory compression in core open-source engine. ✅ (v0.7.0)
- [x] **Composite Scalar Indexes**: Multi-column composite roaring bitmaps for frequent multi-column filter queries. ✅ (v0.7.0)
- [x] **Multi-Vector Search**: Query planner and scoring coordination to search and rank across multiple embedding columns simultaneously using Reciprocal Rank Fusion (RRF). ✅ (v0.7.0)

### 4.3 Graph RAG & Lakehouse Graph Analytics [Community / Free]
Native graph analytics on Iceberg edge tables with sidecar index acceleration, replacing Neo4j + Pinecone combos:
- [ ] **Standard Edge Table Layout**: Define standard Iceberg edge table schema (`source_id`, `target_id`, `relation`, `weight`, `embeddings`).
- [ ] **Sidecar Indexes for Edges**: Auto-generate sidecar indexes on `source_id` and `target_id` columns (Roaring Bitmap) for O(1) edge lookups.
- [ ] **Graph SQL Functions (DataFusion UDFs)**:
  - `PAGERANK(edge_table, damping, max_iterations, tolerance)`: Iterative PageRank over edge table.
  - `COMMUNITY_DETECT(edge_table, algorithm, resolution)`: Louvain / Label Propagation community detection.
  - `GRAPH_NEIGHBORS(entity_id, edge_table, hops, direction)`: 1–N hop neighborhood retrieval.
  - `NODE_SIMILARITY(node_a, node_b, edge_table, method)`: Jaccard and overlap similarity via sidecar bitmap intersection.
  - `CONNECTED_COMPONENTS(edge_table)`: Component labeling via iterative label propagation.
  - `DEGREE_CENTRALITY(edge_table, direction)`: In-degree, out-degree, and total degree aggregation.
- [ ] **Graph RAG Pipeline Integration**: `GRAPH_RAG_SEARCH(query_embedding, edge_table, doc_table, mode, community_col)` combined graph + vector search (local and global modes).
- [ ] **Python Graph API**: `table.pagerank()`, `table.communities()`, `table.graph_neighbors()`, `table.to_networkx()`.
- [ ] **dbt Macros (`dbt-benostreamdb`)**: `{{ pagerank() }}`, `{{ community_detect() }}`, `{{ graph_neighbors() }}`.
- [ ] **Search Gateway Graph Endpoints**: Qdrant Port 6333 `/points/search` with `graph_filter`; OpenSearch Port 9200 `_search` DSL with `graph_neighbors`.

### 4.4 Packaging, Hardware & CI
- [ ] **Universal GPU PyPI Wheel**: Distribute a single universal Python wheel leveraging `cudarc` runtime dynamic loading (`libcuda.so`) and WGPU across Linux and macOS.
- [ ] **GitHub Actions CUDA CI**: Automated CUDA build and test pipeline with `nvidia/cuda` Docker containers.

### 4.5 Codebase Intelligence & Model Context Protocol (MCP) Server
- [ ] **MCP Server Implementation (`benostream-mcp`)**: Standard Model Context Protocol (JSON-RPC over stdio and SSE) exposing tools: `code_search`, `find_symbol`, `get_context`, `code_graph`.
- [ ] **AST Semantic Chunking**: Tree-sitter integration for Rust, Python, TS/JS, Go, Java, C++.
- [ ] **Git-Diff Incremental CI Indexer**: CLI command `benostream index --diff-since <ref>` with incremental Parquet & overlay appends.
- [ ] **Official GitHub Action**: `benostreamdb/index-action@v1`.

### 4.6 Commercial Enterprise Gated Capabilities [Paid Tiers]
- [ ] **Row-Level Security (RLS) & Multi-Tenancy**: Sidecar-level tenant bitmap isolation (`.idx` intersection before reading Parquet).
- [ ] **Dynamic Column Masking**: Role-based PII redaction on query and vector results.
- [ ] **Customer-Managed Encryption Keys (CMEK)**: Envelope encryption for sidecar index files via AWS KMS, GCP KMS, or HashiCorp Vault.
- [ ] **Cryptographic Audit Logging**: Tamper-evident hash chain recording queries across all three protocols (9200, 6333, 50051).
- [ ] **SIEM Telemetry Export**: Native connector export to Splunk, Datadog, and AWS CloudWatch.
- [ ] **Cross-Catalog Governance Propagation**: Unified RLS policies and audit synchronization across Polaris, Unity, and Glue catalogs.
- [ ] **Fused SIMD & Tensor Core Kernels**: Hand-crafted AVX-512, ARM SVE, and Hopper/Blackwell FP8/FP4 fused kernels.
- [ ] **GPUDirect Storage (GDS) Bypass**: Direct NVMe/S3 local cache streaming to GPU VRAM, bypassing host CPU/PCIe bottleneck.
- [ ] **Sidecar Lifecycle Manager**: Autonomous 3-format coordinated compaction (Iceberg manifests + Parquet bin-packing + HNSW/Bitmap sidecars) with cost-aware S3 scheduling and recall drift rebalancing.

---

## 5. Key Execution Risks & Mitigations

1. **Risk: Mechanical Disk Thrashing during Backfill (14TB Spinning Disk)**
   - *Mitigation*: Strictly enforce single-threaded sequential readahead in the Rust ingestion engine. Decouple disk I/O from CPU parsing using in-memory channels. Never run recursive directory traversals across raw directories on `/dev/sda1`.
2. **Risk: SEC XBRL / XML Schema Drift**
   - *Mitigation*: Anchoring to **2014–Present** ensures universal XBRL compliance. Rust parsers use robust fallback matchers for non-standard XML tags.
3. **Risk: Brand Confusion between Database Engine and Financial Application**
   - *Mitigation*: Maintain **2 separate repositories**. `benostreamdb` remains a pristine, horizontal, Apache 2.0 database engine. `edgarstream` houses the domain-specific ingestion workers and Wagtail frontend.
4. **Risk: Legacy IP & Post-Employment Restrictive Covenants (Verity LLC / TMX Group)**
   - *Mitigation*: 
     - **Clean-Room Hardware Discipline**: 100% of proprietary core engine development executed on personal Linux workstations (`/home/ralbright/projects/benostreamdb`); dedicated second-hand Mac for macOS/Metal builds; zero corporate laptop touchpoints.
     - **Personal Identity Integrity**: 100% of Git commit history signed strictly under personal identity (`rla3rd <rla3rd@gmail.com>`).
     - **Horizontal Domain Insulation**: BenoStreamDB is general database infrastructure (unencumbered); quantitative models ring-fenced under explicit "stat-arb" prior inventions carve-out.
     - **Commercial Prudence**: No direct commercial pitches or outreach to TMX Group or former affiliates until 2-year post-employment restrictive periods have cleanly lapsed and been formally cleared by employment counsel.
     - **Open-Source Safe Harbor**: Local Apple Silicon Metal GPU acceleration maintained strictly in the **Apache 2.0 Open Source Community Tier**.

---

## 6. Series A Fundraising Strategy & Execution Playbook (Target: Mid/Late 2027)

A disciplined capital strategy is critical to maximizing founder equity, protecting health, and avoiding predatory venture terms:

### 6.1 The Non-Negotiable Principle: Zero Fundraising in Fall 2026
* **Health & Cognitive Protection**: Running an intense 40-meeting VC roadshow while recovering from a TBI is counterproductive and introduces dangerous stress.
* **Never Sell at the Bottom**: In Fall 2026, BenoStreamDB is an unmarketed engine. VCs would treat it as an unproven pre-seed experiment and demand 25%–30% equity for minimal capital.
* **Zero Payroll Burn Advantage**: The monthly infrastructure burn is negligible (<$100/mo). Because there is no burn clock forcing a round, time works entirely in the founder's favor.

### 6.2 Target Series A Metrics ($10M–$15M Round @ $50M–$100M+ Valuation)
To command tier-1 Silicon Valley term sheets in 2027, the company targets hitting at least one (ideally both) of the standard database infrastructure hurdles:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THE TWO SERIES A ENTRY TRIGGERS                       │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ TRIGGER 1: THE COMMERCIAL HURDLE     │ TRIGGER 2: THE OPEN-SOURCE HURDLE    │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • $500k – $1M in ARR Run-Rate        │ • 3,000 – 5,000+ GitHub Stars        │
│ • 5 to 10 paying institutional desks │ • 20,000+ monthly PyPI downloads     │
│ • High Net Revenue Retention (>120%) │ • Organic developer buzz on HN / X   │
│ • Proof that hedge funds/enterprises │ • 2–3 recognizable tech logos using  │
│   expand contract values             │   it for internal Iceberg vector RAG │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### 6.3 The 4-Stage Fundraising Execution Timeline

```
Stage 1: Health, Recovery & Turn-Key Prep (Fall 2026: Sept – Dec 2026)
├── PRIORITY: Zero stress, TBI healing, self-paced systems engineering.
├── Complete v0.8.0 Graph RAG and the sequential EDGAR ingestion engine.
└── Have the entire 12-year lakehouse ready before returning to university.
    [Fundraising Action: ZERO investor outreach. Total silence.]

Stage 2: Organic Proof & Academic Release (Q1 2027: Jan – April 2027)
├── Resume PhD in January with a turn-key empirical research platform.
├── Publish the Python wheel publicly on PyPI (`pip install benostreamdb`).
├── Publish the benchmark: "12 Years of SEC Filings on S3 under 4 GB RAM."
├── Sign first 3 to 5 quant beta desks ($50k–$150k early ARR).
└── [Fundraising Action: Zero formal pitching. If inbound VCs reach out, 
    politely reply: "Focusing on product and academic paper; will talk mid-2027."]

Stage 3: The Informal "Warm-Up" (Q2 2027: May – June 2027)
├── The academic paper (Carhart + ISF) is submitted/circulating.
├── Community adoption is growing organically on GitHub and PyPI.
└── [Fundraising Action: Have 4-5 casual 20-minute intro coffees with 
    top-tier data infrastructure partners (Amplify Partners, Index Ventures, 
    Bessemer, Felicis, Redpoint). Frame it as: "Not raising yet, just building 
    relationships for our Series A later this year."]

Stage 4: The Series A Strike Zone (Late Q2 / Q3 2027: July – October 2027)
├── Sitting on $500k+ ARR or 5,000+ GitHub stars.
├── Cognitive energy is 100% restored, and PhD research is running smoothly.
└── [Fundraising Action: Launch a tight 3-week fundraising process. 
    Raise $10M–$15M on a $60M–$80M valuation to build "BenoStream Cloud" 
    and hire the first 5 systems and enterprise sales engineers.]
```
