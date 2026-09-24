# BenoStreamDB Business Plan & Architecture Directory

**Status:** Executive Architecture & Strategic Plan  
**Target Horizon:** 2026–2027  
**Git Status:** Gitignored (Proprietary / Internal)  
**Last Updated:** 2026-09-13  

---

## Directory Navigation

This directory contains the modularized business plan, commercialization strategy, competitive intelligence, and technical ingestion architecture for **BenoStreamDB**, its dogfooding engine **EdgarStreamDB**, and its showcase frontend **EdgarStream**:

1. **[01. Market & Competitive Landscape](file:///home/ralbright/projects/benostreamdb/business_plan/01_market_and_competitive_landscape.md)**
   - The $30B+ horizontal lakehouse index & search TAM.
   - Real-world incumbent teardown: **InsiderScore / VerityData (acquired by TMX Group in Canada)** ($20M ARR, 80 employees, M&A precedent, 90%+ cost mitigation).
   - The Four Threat Vectors (Disk-adjacent, Dedicated Vector Silos, Search Incumbents, Cloud Giants).
   - Deep teardown: **BenoStreamDB vs. LanceDB ("The Format Trap")**.
   - Master Competitive Moat Matrix.
   - **Legal Firebreak & IP Clean-Room Governance**: Stat-arb carve-out, horizontal engine insulation, and non-compete risk mitigation.

2. **[02. Commercialization & Monetization](file:///home/ralbright/projects/benostreamdb/business_plan/02_commercialization_and_monetization.md)** (and **[COMMERCIALIZATION_STRATEGY.md](file:///home/ralbright/projects/benostreamdb/business_plan/COMMERCIALIZATION_STRATEGY.md)**)
   - September 2026 Feature Viability Matrix (Auditing free vs paid).
   - Tiered Packaging (Community, Team, Enterprise).
   - Open-Core & Dual-Licensing (Apache 2.0 vs Enterprise Source-Available).
   - The **2014–Present** Modern Horizon + Free **Tiingo** BYOK connector.
   - Customer-funded historical backfill (**2004–2013** Enterprise upsell at $35k–$50k ARR).
   - **Strategic Exit & Corporate Licensing**: The TMX Group / TMX Datalinx monetization vector.

3. **[03. Financial Model & Unit Economics](file:///home/ralbright/projects/benostreamdb/business_plan/03_financial_model_and_unit_economics.md)**
   - 5-Year Revenue & Expense Projections (Year 1 to Year 5).
   - Real-world legacy production cost teardown ($20k/mo legacy vs $1,450/mo BenoStreamDB stack, saving $222,600/year).
   - Solo-founder bootstrap economics and PhD non-dilutive funding model.

4. **[04. Technical Architecture & Ingestion Strategy](file:///home/ralbright/projects/benostreamdb/business_plan/04_technical_architecture_and_ingestion.md)**
   - **Ingestion Engine**: Native Rust binary for high-throughput sequential streaming from the 14TB spinning disk (`/dev/sda1`, 6.5 TB used).
   - **Raw Zstandard Format**: Ingesting raw `.zst` files (NOT Parquet) by form type via in-memory `<SEC-HEADER>` sniffing and zero-copy parsing (`quick-xml`, `lol-html`).
   - **Spinning Disk Head Seek Elimination**: Single sequential readahead worker feeding in-memory Rayon thread pools to prevent HDD thrashing.
   - **Wagtail Frontend Integration**: Python bindings (`benostreamdb-python`) and Arrow Flight (Port 50051) for sub-millisecond analytical & RAG queries; keeping PostgreSQL lean (<20 GB) for Wagtail CMS/auth with zero text/vector bloat.

5. **[05. SEC EDGAR Taxonomy & Graph RAG](file:///home/ralbright/projects/benostreamdb/business_plan/05_sec_edgar_taxonomy_and_graph_rag.md)**
   - Complete SEC EDGAR Form Taxonomy (~160+ form types across 10 categories).
   - Master Graph RAG Topology (Institutional, Corporate, Governance, Semantic Lakehouse layers).
   - Physical Iceberg storage layout on S3 (`edgar.filing_chunks`, `edgar.transactions_and_holdings`, `edgar.graph_edges`).

6. **[06. Roadmap & Execution Milestones](file:///home/ralbright/projects/benostreamdb/business_plan/06_roadmap_and_milestones.md)**
   - Phased Real-World Execution Timeline (Dec 2025 – 2027).
   - Key enterprise risks and technical mitigations (legacy IP, non-compete, hardware clean-room governance).
   - **Series A Fundraising Playbook**: 4-stage execution timeline, target metrics ($10M–$15M on $50M–$100M+ valuation), and Mid/Late 2027 strike zone.

---

## Executive Overview: The Three-Tier Product Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   BENOSTREAMDB (Horizontal Data Infrastructure)            │
│  • Primary Product & Commercial Focus ($30B+ TAM across all industries)     │
│  • Apache Iceberg V2/V3 + RoaringBitmap + HNSW/TQ8 Overlays                 │
│  • Multi-Protocol Gateways: OpenSearch DSL, Qdrant wire, Arrow Flight SQL   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Real-World Scale Lab
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 EDGARSTREAMDB (Internal Scale Lab & Showcase)               │
│  • Dogfooding proving ground: 10+ years of SEC EDGAR (Form 4, 10-K/Q)       │
│  • Demonstrates hybrid vector + high-cardinality pre-filtering under 4GB RAM │
│  • Powers PhD dissertation (Carhart + Insider Sentiment Factor ISF)         │
│  • Opportunistic Revenue: Packaged data appliance for quants ($500–$1.5k/mo)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Downstream UI
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EdgarStream (github.com/rla3rd/edgarstream)               │
│  • Independent Open-Source Application & Financial Intelligence Terminal    │
│  • Built on Wagtail (Python / Django) for clean admin, CMS, and analyst UI  │
│  • Connects directly to EdgarStreamDB via Arrow Flight & Python SDK         │
│  • Showcases live Graph RAG, interactive filing search & insider analytics  │
```

---

## Strategic Principle: BenoStreamDB is Where the Real Money at Scale Lives (Not EDGAR)

* **Horizontal Infrastructure vs. Vertical Niche**:
  * **EDGAR applications** (e.g. InsiderScore, Sentieo) are vertical niche tools capped at a ~$50M TAM with 3x–6x ARR valuation multiples.
  * **BenoStreamDB** is horizontal data infrastructure addressing a **$30B+ TAM** (competing with Snowflake, Databricks, Elastic, and LanceDB) with **15x–30x+ ARR valuation multiples**.
* **The Role of EdgarStreamDB**:
  * EdgarStreamDB is not the final commercial capstone; it is the **ultimate "Lighthouse" proof of concept**.
  * By indexing 12+ years of brutal, high-skew SEC filings under a 4 GB RAM ceiling for $45/mo on S3, we establish the definitive real-world benchmark that closes Fortune 500 enterprise infrastructure deals.
