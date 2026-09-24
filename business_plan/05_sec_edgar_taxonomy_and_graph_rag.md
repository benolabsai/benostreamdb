# 05. SEC EDGAR Taxonomy & Graph RAG Architecture

**Status:** Technical Specification & Form Taxonomy  
**Target Dataset:** SEC EDGAR (2014–Present, 12-Year Horizon)  
**Storage Target:** Apache Iceberg on AWS S3 with BenoStreamDB Sidecars  

---

## 1. SEC EDGAR Form Taxonomy (The 10 Master Categories)

The SEC EDGAR system spans over 160 distinct submission form types. The table below outlines how each category is ingested by the Rust ingestion worker and mapped into Iceberg Parquet tables:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               MASTER SEC EDGAR FORM TAXONOMY (10 CATEGORIES)                            │
├──────────────────────┬────────────────────────────────┬──────────────────────┬──────────────────────────┤
│ CATEGORY             │ KEY SEC FORM TYPES             │ RAW FORMAT (ZST)     │ EXTRACTED ENTITIES       │
├──────────────────────┼────────────────────────────────┼──────────────────────┼──────────────────────────┤
│ A. Periodic Financial│ 10-K, 10-K/A, 10-Q, 10-Q/A,    │ HTML + Inline XBRL   │ Balance Sheet, Income,   │
│    Disclosures       │ 20-F, 40-F, 6-K                │ (iXBRL)              │ Cash Flows, MD&A, Risks  │
├──────────────────────┼────────────────────────────────┼──────────────────────┼──────────────────────────┤
│ B. Material Current  │ 8-K, 8-K/A, 6-K                │ HTML + Inline XBRL   │ Item 1.01 Material Agr., │
│    Events            │                                │                      │ Item 2.02 Earnings, CEO  │
├──────────────────────┼────────────────────────────────┼──────────────────────┼──────────────────────────┤
│ C. Insider Trading & │ Form 3, 3/A, Form 4, 4/A,      │ Structured XML       │ Insider CIK, Shares,     │
│    Beneficial Owners │ Form 5, 5/A, Form 144          │ (SEC Form 4 XSD)     │ Price, Trans Code, 10b51 │
├──────────────────────┼────────────────────────────────┼──────────────────────┼──────────────────────────┤
│ D. Institutional     │ 13F-HR, 13F-NT, SC 13D, 13D/A, │ Structured XML       │ Institutional Manager,   │
│    Holdings & Stakes │ SC 13G, 13G/A, 13H             │ + HTML               │ Portfolio Position, %    │
├──────────────────────┼────────────────────────────────┼──────────────────────┼──────────────────────────┤
│ E. Proxy Statements  │ DEF 14A, DEFA14A, DEFM14A,     │ HTML + iXBRL         │ Board Nominees, Exec     │
│    & Governance      │ PRE 14A, PREC14A, DEF 14C      │                      │ Compensation, Peer Group │
├──────────────────────┼────────────────────────────────┼──────────────────────┼──────────────────────────┤
│ F. Capital Raising & │ S-1, S-3, S-3ASR, S-4, S-8,    │ HTML + iXBRL         │ Offering Size, Syndicate,│
│    Offerings         │ F-1, F-3, F-4, 424B1–424B8, FWP│                      │ Dilution, Conversion Px  │
├──────────────────────┼────────────────────────────────┼──────────────────────┼──────────────────────────┤
│ G. Tender Offers &   │ SC TO-T, SC TO-I, SC 13E3,     │ HTML                 │ Tender Offer Price, Target│
│    M&A Buyouts       │ SC 14D9, 14D9/A                │                      │ Board Recommendation     │
├──────────────────────┼────────────────────────────────┼──────────────────────┼──────────────────────────┤
│ H. Private Offerings │ Form D, Form D/A, Form C,      │ Structured XML       │ Unregistered Capital,    │
│    & Crowdfunding    │ 1-A, 1-K, 1-SA, 1-U            │ (Form D XSD)         │ Minimum Investment, 506b │
├──────────────────────┼────────────────────────────────┼──────────────────────┼──────────────────────────┤
│ I. Funds & ETFs      │ N-PORT, N-PORT/P, N-CEN,       │ Structured XML       │ Fund Portfolio Holdings, │
│    (1940 Act)        │ N-MFP, N-MFP2, N-CSR, N-PX     │ + HTML               │ Swap Counterparties      │
├──────────────────────┼────────────────────────────────┼──────────────────────┼──────────────────────────┤
│ J. Market Entities & │ BD, BDW, TA-1, TA-2, TA-W,     │ Structured XML       │ Transfer Agent Ledger,   │
│    Securitization    │ ABS-EE, ABS-15G                │ + HTML               │ Loan-Level Default Data  │
└──────────────────────┴────────────────────────────────┴──────────────────────┴──────────────────────────┘
```

---

## 2. The Master Graph RAG Topology

In this architecture, filings are decomposed into a unified property graph with semantic vector overlays:

```mermaid
flowchart TD
    subgraph Institutions ["Institutional Layer (13F / 13D / N-PORT)"]
        IM["Institutional Manager<br>(e.g. Berkshire, Citadel)"]
        FD["Fund / Series / ETF<br>(N-PORT Portfolio)"]
    end

    subgraph Corporate ["Corporate Layer (10-K / 8-K / S-1)"]
        CO["Company / Issuer<br>(CIK, Ticker, SIC, State)"]
        SUB["Subsidiary Entity<br>(EX-21.1)"]
        EV["Material Event<br>(Form 8-K Items)"]
        OFF["Capital Offering<br>(Form D / S-3)"]
    end

    subgraph Governance ["Insider & Governance Layer (Form 4 / DEF 14A)"]
        PE["Person / Insider<br>(CIK, Name)"]
    end

    subgraph Semantic ["Semantic Lakehouse Layer (BenoStreamDB Sidecars)"]
        FL["Filing Record<br>(Accession Number, Date)"]
        FC["Filing Text Chunk<br>(Item 1, 1A, 7 + TQ8 Vector)"]
        FF["Financial Fact<br>(XBRL Tag, Value, Period)"]
    end

    IM -->|HOLDS_STAKE| CO
    FD -->|HOLDS_ASSET| CO
    PE -->|TRANSACTED<br>(Form 4 / 144)| CO
    PE -->|HOLDS_ROLE<br>(CEO / CFO)| CO
    PE -->|BOARD_MEMBER_OF<br>(DEF 14A)| CO
    CO -->|HAS_SUBSIDIARY| SUB
    CO -->|DISCLOSED_EVENT| EV
    CO -->|ISSUED_OFFERING| OFF
    CO -->|FILED| FL
    FL -->|CONTAINS_CHUNK| FC
    FL -->|REPORTS_FACT| FF
```

---

## 3. The Structural Paradigm Shift: Legacy Relational vs. Graph RAG Lakehouse

Adopting **Graph RAG fundamentally alters the table layout**, collapsing 30–40 fragmented, form-specific relational tables into **4 unified lakehouse structures**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE TABLE LAYOUT ARCHITECTURAL SHIFT                     │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ LEGACY RELATIONAL / FORM-CENTRIC     │ GRAPH RAG LAKEHOUSE / ENTITY-CENTRIC │
│ (EdgarStream 2018 / InsiderScore)      │ (BenoStreamDB + Apache Iceberg)     │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • 40+ disparate tables:              │ • Only 4 unified Iceberg tables:     │
│   - filing_documents (HTML blobs)    │   1. edgar.entities (All nodes)      │
│   - form4_transactions               │   2. edgar.graph_edges (Adjacency)   │
│   - form4_derivatives                │   3. edgar.filing_chunks (Vectors)   │
│   - form13f_holdings                 │   4. edgar.graph_communities (RAG)   │
│   - reporting_owners                 │ • Unified (Node, Edge, Chunk) stream │
│ • Expensive 6-table SQL JOINs        │ • RoaringBitmap O(1) edge traversal  │
│ • Blind to cross-board syndicates    │ • Native PageRank & 2-hop traversal  │
│ • Text trapped in database disk      │ • Storage on S3; 90% storage savings │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 4. The 4 Unified Graph RAG Iceberg Table Schemas

Instead of maintaining brittle schemas for every SEC form variation, the Rust ingestion engine transforms all filings into four foundational tables:

### Table 1: `edgar.entities` (The Unified Node Store)
Represents all physical, corporate, and semantic entities discovered across all filings:
```sql
CREATE TABLE edgar.entities (
    entity_id           VARCHAR,        -- e.g. 'PERSON:0001234567', 'COMPANY:0000320193', 'FUND:0001067983'
    entity_type         VARCHAR,        -- 'Person', 'Issuer', 'InstitutionalManager', 'Fund', 'MaterialEvent'
    primary_name        VARCHAR,        -- 'Tim Cook', 'Apple Inc.', 'Berkshire Hathaway'
    cik                 VARCHAR,        -- Central Index Key (if applicable)
    ticker              VARCHAR,        -- Trading symbol (if issuer)
    sic_code            VARCHAR,        -- Standard Industrial Classification
    properties          STRING,         -- JSON payload with entity-specific attributes
    description_vector  FIXED_SIZE_LIST(FLOAT32, 768) -- Optional semantic embedding for entity matching
) USING iceberg;
```

### Table 2: `edgar.graph_edges` (The High-Throughput Adjacency Matrix)
Represents all directed, typed, and temporal interactions between entities:
```sql
CREATE TABLE edgar.graph_edges (
    edge_id             VARCHAR,        -- UUID
    source_entity_id    VARCHAR,        -- e.g. 'PERSON:0001234567'
    target_entity_id    VARCHAR,        -- e.g. 'COMPANY:0000320193'
    relation_type       VARCHAR,        -- 'TRANSACTED_IN', 'HOLDS_ROLE', 'BOARD_MEMBER_OF', 'HOLDS_STAKE'
    weight              DOUBLE,         -- Transaction value ($), or ownership percentage, or edge strength
    transaction_code    VARCHAR,        -- 'P' (Purchase), 'S' (Sale), 'M' (Option), 'A' (Grant)
    shares              DOUBLE,         -- Number of shares involved
    price_per_share     DOUBLE,         -- Execution price
    is_10b5_1           BOOLEAN,        -- Rule 10b5-1 pre-scheduled trading plan flag
    filing_date         DATE,           -- SEC filing date
    valid_from          DATE,           -- Effective date of relationship
    valid_to            DATE,           -- Expiration / termination date (NULL if active)
    properties          STRING          -- JSON payload for form-specific footnotes & metadata
) USING iceberg
PARTITIONED BY (days(filing_date), relation_type);
```
* **Sidecar Indexes**: RoaringBitmap `.idx` sidecars on `source_entity_id` and `target_entity_id`. This allows BenoStreamDB to compute **multi-hop graph neighborhoods in <5ms** directly in-engine without requiring Neo4j.

### Table 3: `edgar.filing_chunks` (The Semantic Hybrid Vector & Text Store)
Stores segmented narrative text from 10-K, 10-Q, 8-K, S-1, and DEF 14A filings:
```sql
CREATE TABLE edgar.filing_chunks (
    chunk_id            VARCHAR,        -- e.g. '0000320193-24-000106_item1a_c04'
    accession_number    VARCHAR,        -- SEC filing identifier
    entity_id           VARCHAR,        -- Parent issuer/company node ID
    form_type           VARCHAR,        -- '10-K', '10-Q', '8-K', 'DEF 14A'
    item_section        VARCHAR,        -- 'Item 1A. Risk Factors', 'Item 7. MD&A'
    breadcrumb_path     VARCHAR,        -- Hierarchical markdown header path
    chunk_text          VARCHAR,        -- Cleaned markdown narrative
    vector_embedding    FIXED_SIZE_LIST(FLOAT32, 768), -- TurboQuant TQ8 compressed embedding
    mentioned_entities  LIST(VARCHAR)   -- Array of entity IDs referenced in this chunk
) USING iceberg
PARTITIONED BY (form_type, bucket(16, entity_id));
```
* **Sidecar Indexes**: TurboQuant TQ8 HNSW graph (`.hnsw`), BM25 inverted index (`.inv.parquet`), and RoaringBitmap on `entity_id` and `item_section`.

### Table 4: `edgar.graph_communities` (Hierarchical Summaries for Global RAG)
Powers high-level, multi-document synthesis (e.g. *"What are the common supply-chain bottlenecks across semiconductor companies with director cluster sales?"*):
```sql
CREATE TABLE edgar.graph_communities (
    community_id        VARCHAR,        -- e.g. 'COMM_L2_482'
    level               INTEGER,        -- Hierarchy level (0=raw, 1=local, 2=global)
    title               VARCHAR,        -- e.g. 'Semiconductor Executive Syndicate'
    member_entity_ids   LIST(VARCHAR),  -- List of entities in this cluster
    community_summary   VARCHAR,        -- LLM-generated comprehensive cluster summary
    summary_vector      FIXED_SIZE_LIST(FLOAT32, 768) -- Embedding for global thematic routing
) USING iceberg;
```

---

## 5. How Graph RAG Radically Simplifies the Rust Ingestion Port

In traditional architectures, the ingestion worker has to map each filing into dozens of distinct table schemas with foreign keys.

Under the Graph RAG lakehouse layout, **the Rust ingestion engine only ever emits three primitive types**:
```rust
pub enum EdgarGraphEvent {
    Node(EntityNode),
    Edge(GraphEdge),
    Chunk(FilingChunk),
}
```

### Form Ingestion Examples:
1. **Form 4 Ingestion**:
   - Emits 1 Node for the Reporting Owner (`PERSON:CIK`).
   - Emits 1 Node for the Issuer (`COMPANY:CIK`).
   - Emits 1 Edge: `(PERSON) -[HOLDS_ROLE {title: "CFO"}]-> (COMPANY)`.
   - Emits 1 Edge: `(PERSON) -[TRANSACTED_IN {shares: 10000, price: 152.0, code: "P"}]-> (COMPANY)`.
2. **Form 13F Ingestion**:
   - Emits 1 Node for the Institutional Manager (`INSTITUTION:CIK`).
   - Emits N Edges: `(INSTITUTION) -[HOLDS_STAKE {cusip: "037833100", value_k: 4500000}]-> (COMPANY)`.
3. **Form 10-K Ingestion**:
   - Emits N Chunks into `filing_chunks`.
   - Emits directed Edges linking `(COMPANY) -[CONTAINS_CHUNK]-> (CHUNK)`.
   - Entity linking extracts cross-company references: `(CHUNK) -[DISCLOSES_RISK_REGARDING]-> (SUPPLIER_COMPANY)`.

This unified model means adding a new SEC form type to the Rust ingestor takes **hours, not weeks**, because every form simply produces standard nodes and edges.

---

## 6. Form 4 Transaction Aggregation Engine (The Institutional Standard)

In raw Form 4 XML filings, a single executive trading decision (such as a 10b5-1 sale of 50,000 shares) is frequently reported across **10 to 30 individual execution lots** (e.g. 500 shares @ $185.10, 1,200 shares @ $185.12). 

Displaying raw unaggregated fills floods UI screens with noise and breaks quantitative event studies by counting a single economic decision as 25 separate trades.

### 6.1 The Institutional Aggregation Grain
All raw Form 4 lines within a submission are pre-aggregated in-memory during Rust ingestion at the **Economic Trade Event** grain:
* **Grouping Dimensions**: `(company_cik, ticker, insider_cik, insider_name, filing_date, transaction_code, transaction_date)`
* **Aggregated Metrics Computed**:
  - $\text{Total Shares} = \sum \text{shares}_i$
  - $\text{Total Value} = \sum (\text{shares}_i \times \text{price}_i)$
  - $\text{VWAP} = \frac{\text{Total Value}}{\text{Total Shares}}$ (Volume Weighted Average Price)
  - $\text{Price Min} = \min(\text{price}_i)$, $\text{Price Max} = \max(\text{price}_i)$
  - $\text{Raw Fill Count} = \text{COUNT}(*)$ (e.g. 24 execution lots merged into 1 event)
  - $\text{Is 10b5-1} = \bigvee \text{is\_10b5\_1}_i$ (True if any lot was marked 10b5-1)
  - $\text{Post-Trade Shares Held} = \text{Latest reported balance}$

### 6.2 Two-Tier Storage Architecture in Iceberg
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: edgar.graph_edges (The Aggregated Economic Trade Events)           │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Grain: (Company, Insider, FilingDate, Code, TransDate)                   │
│  • Edge: (PERSON:Cook) -[TRANSACTED {shares: 50000, vwap: 185.14}]-> (AAPL) │
│  • Powers: Wagtail UI, Graph RAG, PageRank, Quant Event Studies & Alpha     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ 1-to-N Link (edge_id)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 2: edgar.form4_raw_fills (The Underlying Execution Lots)              │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Stores every single raw line from Table I & Table II of Form 4 XML        │
│  • Preserves exact broker execution lots, raw prices, and specific footnotes│
│  • Displayed only when an analyst clicks "Inspect N Execution Fills" in UI  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Data Quality, Automated Anomaly Detection & Wagtail Curation Overrides

Data errors in SEC filings are inevitable (filer typos, inverted dates, ambiguous footnote codes). To ensure institutional reliability without database bloat, the platform separates **automated error detection** from **human-in-the-loop curation**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: AUTOMATED ANOMALY DETECTION (BenoStreamDB SQL Engine)             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Runs daily across normalized Iceberg tables + Tiingo EOD Prices:           │
│  • Price Outlier Check: Reported Price > 5x or < 0.2x Tiingo Market Close   │
│  • Share Count Check: Shares Traded > Company Float / Shares Outstanding    │
│  • Math Reconciliation: Post-Shares != Prior Shares - Traded Shares         │
│  • Temporal Inversion: Transaction Date > Filing Date                       │
│  Flagged events are populated into: edgar.anomalies_queue                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: WAGTAIL CURATION & REVIEW WORKBENCH (Human-in-the-Loop UI)         │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Wagtail Admin displays side-by-side:                                     │
│    - Flagged transaction + Raw Form 4 XML snippet                           │
│    - Tiingo closing price for that trading day                              │
│    - Suggested correction (e.g. $18,500.00 -> $185.00 decimal shift)        │
│  • Analyst reviews and clicks [ Approve & Apply Override ]                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: SPARSE OVERLAY TABLE (edgar.curation_overrides)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Exception-only table holding ONLY the 0.1%–0.5% curated corrections      │
│  • 99.5% of clean filings bypass the override table via RoaringBitmap (.idx)│
│  • Zero mutations to the 6.5 TB historical Parquet files (Immutable Audit)  │
│  • Wagtail & EdgarStream views instantly return clean, corrected numbers      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Eliminating the "Regex Trap" across SEC Forms

Legacy systems (including 2018 EdgarStream) suffered from thousands of lines of fragile, unmaintainable regular expressions trying to match raw HTML:

1. **Structured Forms (Forms 3, 4, 5, 13F, Form D, N-PORT)**:
   - **Zero Regex**: Strict XML schemas (XSD v5.5) parsed deterministically with streaming Rust `quick-xml`.
2. **Financial Disclosures (10-K, 10-Q Financial Statements)**:
   - **Zero Regex**: Parsed directly from standard Inline XBRL (`iXBRL`) tags (`<ix:nonFraction>`).
3. **Narrative Text (10-K Item 1A Risks, 8-K Items, DEF 14A Proxies)**:
   - **DOM Normalization (`lol-html`)**: Rather than matching raw HTML with complex regexes, the parser strips fonts, converts `&nbsp;`, and normalizes text lines before evaluating header boundaries.
   - **iXBRL Continuation Anchors**: Modern 10-Ks tag narrative blocks with standard SEC anchors.
   - **Small Language Model Fallback**: If a table-of-contents layout is non-standard (<1% of filings), an embedded local SLM (e.g. Qwen2.5-3B) identifies the starting page, eliminating manual regex maintenance.

---

## 9. EDGAR-Native Security Master & OpenFIGI Symbology Integration

A core competitive advantage over legacy platforms (such as InsiderScore, which spends $50,000–$150,000+/year licensing Morningstar Global Equities data) is constructing an **authoritative, self-updating Security Master directly from SEC EDGAR filings and OpenFIGI**, eliminating third-party symbology vendor dependencies.

### 9.1 Leveraging Existing EdgarStream Assets
The codebase in `edgarstream` already possesses foundational symbology and security master assets that are integrated into this architecture:
1. **`edgarstream/processes/symbology.py`**:
   - Implements `OpenFIGIClient.resolve_batch()`: Resolves batches of CUSIP, ISIN, and Ticker identifiers into global Financial Instrument Global Identifiers (FIGI) and exchange codes with rate-limit backoff.
   - Implements `enrich_company_model()`: Takes company records and enriches them with FIGI, canonical ticker, and exchange code.
2. **`edgarstream/models.py` (`Company` & `CompanyInfo` Models)**:
   - `Company` already models: `cik`, `cik_name`, `ticker`, `exchange`, `sic_code`, `figi` (OpenFIGI identifier), `cusip`, `isin`, `is_active`, `is_human`.
   - `CompanyInfo` already models: `tickers` (ArrayField), `exchanges` (ArrayField), `former_names` (JSONField for historical name tracking), `industry`, `sic_description`, `state_of_incorporation`, `fiscal_year_end`.

### 9.2 The EDGAR-Native Single Source of Truth
Instead of paying Morningstar, the Security Master is synthesized directly from statutory SEC disclosures:
* **Baseline Directory**: Bootstrapped from the official SEC EDGAR endpoint `company_tickers_exchange.json` (maps all active CIKs, tickers, and primary exchanges).
* **CUSIP & Share Class Master**: Extracted from quarterly **Form 13F-HR** institutional information tables (`<cusip>`, `<nameOfIssuer>`, `<titleOfClass>`) and official SEC Section 13(f) security lists.
* **Real-Time Ticker & Corporate Changes**:
  - **Form 8-K Item 5.03**: Captures legal ticker changes (e.g., FB $\rightarrow$ META) and charter amendments.
  - **Form 25 & Form 15**: Captures exchange delistings, transfers (e.g., NYSE $\rightarrow$ NASDAQ), and corporate deregistrations.
  - **Form 8-K Item 2.01**: Captures M&A merger completions, linking acquired entities to their successor CIK.
* **OpenFIGI Enrichment**: Uses the existing `OpenFIGIClient` to append permanent, non-proprietary FIGI identifiers with **zero redistribution fees**.

### 9.3 Physical Iceberg Table: `edgar.security_master`
```sql
CREATE TABLE edgar.security_master (
    security_id             VARCHAR,        -- e.g. 'SEC:0000320193:037833100'
    company_cik             VARCHAR,        -- '0000320193'
    company_name            VARCHAR,        -- 'Apple Inc.'
    current_ticker          VARCHAR,        -- 'AAPL'
    primary_exchange        VARCHAR,        -- 'NASDAQ'
    
    -- Symbology Cross-Reference
    cusip_6                 VARCHAR,        -- Issuer code: '037833'
    cusip_8                 VARCHAR,        -- Issue code: '03783310'
    cusip_9                 VARCHAR,        -- Full CUSIP: '037833100'
    isin                    VARCHAR,        -- 'US0378331005'
    openfigi_id             VARCHAR,        -- OpenFIGI global identifier
    
    -- Classification
    security_class          VARCHAR,        -- 'Common Stock', 'Class A', 'ADR'
    sic_code                VARCHAR,        -- '3571'
    sic_description         VARCHAR,        -- 'ELECTRONIC COMPUTERS'
    fiscal_year_end         VARCHAR,        -- '0930'
    state_of_incorporation  VARCHAR,        -- 'CA'
    
    -- Status & Corporate Actions
    status                  VARCHAR,        -- 'ACTIVE', 'DELISTED', 'ACQUIRED', 'MERGED'
    first_filing_date       DATE,           -- Date company first filed
    last_filing_date        DATE,           -- Most recent filing date
    successor_cik           VARCHAR,        -- If acquired, successor entity CIK
    
    -- Point-in-Time Ticker History
    ticker_history          LIST(STRUCT<
                                ticker: VARCHAR, 
                                start_date: DATE, 
                                end_date: DATE, 
                                exchange: VARCHAR, 
                                source_accession: VARCHAR
                            >)
) USING iceberg
PARTITIONED BY (status, bucket(16, company_cik));
```

### 9.4 Synchronization with Wagtail
* In Wagtail, the existing `Company` and `CompanyInfo` models serve as the local Django ORM cache for web views.
* The Rust ingestion worker maintains `edgar.security_master` on Iceberg, and Wagtail queries it zero-copy via Arrow Flight or updates local Django model instances during scheduled syncs.
```
