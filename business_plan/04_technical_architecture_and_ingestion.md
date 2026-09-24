# 04. Technical Architecture & Ingestion Strategy

**Status:** Technical Architecture Specification  
**Component:** Ingestion Engine & Lakehouse Integration  
**Applies To:** BenoStreamDB Core, EdgarStreamDB Ingestor, EdgarStream Wagtail  

---

## 1. Architectural Decision: Rust Ingestion vs. Python Bindings

A core engineering question is whether to build the ingestion pipeline in Rust or Python, and where Python bindings belong in the overall system.

```
/home/ralbright/data/edgarstream/edgar/data/ (14TB HDD, raw .zst files)
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │  RUST INGESTION ENGINE (Native CLI / Background Service)  │
 │  • Single sequential readahead worker (zero seek thrash) │
 │  • In-memory Zstandard stream decompression              │
 │  • `<SEC-HEADER>` sniffer -> Form Type Dispatcher        │
 │  • SIMD-accelerated zero-copy parsers:                   │
 │    - quick-xml (Forms 3, 4, 5, 144, 13F, Form D, N-PORT)│
 │    - lol-html (10-K, 10-Q, 8-K, S-1, DEF 14A)            │
 │  • Direct Arrow RecordBatch serialization in memory      │
 │  • Flushes 128 MB Parquet splits directly to S3 / disk   │
 │  • Builds RoaringBitmap (.idx) & TQ8 HNSW (.hnsw) sidecars│
 └──────────────────────────────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │  BENOSTREAMDB ICEBERG LAKEHOUSE                         │
 │  • Immutable Parquet data files on S3 or local NVMe/HDD  │
 │  • Sidecar indices prune 99% of splits before scan       │
 │  • Arrow Flight SQL Server (Port 50051)                  │
 │  • Dual REST Search Gateway (Port 9200 / 6333)           │
 └──────────────────────────────────────────────────────────┘
                      ▲
                      │  (Arrow Flight / PyO3 bindings)
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │  EdgarStream WAGTAIL APPLICATION (Python / Django)       │
 │  • Python Bindings (`benostreamdb-python`) for queries  │
 │  • Lean PostgreSQL (<20 GB) for CMS, Auth, Watchlists    │
 │  • Sub-millisecond analytical & vector retrieval         │
 │  • Zero text/vector bloat in PostgreSQL                  │
 └──────────────────────────────────────────────────────────┘
```

### Architectural Verdict:
1. **Ingestion must be built as a native Rust engine**:
   - Decompressing and parsing **6.5 TB of raw Zstandard archives** covering 12 years (2014–Present) involves **millions of filings** (~400,000 Form 4 XML files per year alone = ~4.5 million Form 4s).
   - In Python, parsing millions of XML/HTML filings chokes on Python object allocation overhead, Global Interpreter Lock (GIL) contention, and garbage collector pauses. Memory easily balloons beyond 8–16 GB RAM, and backfills take weeks.
   - In Rust, zero-copy string slicing, streaming decompression (`zstd-rs`), SIMD-accelerated XML parsing (`quick-xml`), and direct Arrow RecordBatch construction achieve **30,000 to 80,000+ filings/second** on a single node with **RAM strictly bounded under 2 GB**.
2. **Python Bindings (`benostreamdb-python`) belong in the Wagtail Frontend**:
   - EdgarStream's frontend is built on **Wagtail (Python / Django)**.
   - Wagtail should never touch raw `.zst` files, parse massive XML/SGML, or manage heavy ingestion workers.
   - Instead, Wagtail views and API endpoints use the Python bindings (or PyArrow Flight) as a high-speed analytical query client to fetch already-indexed Parquet rows, run vector semantic queries, and render Graph RAG relationships in sub-milliseconds.

---

## 2. Ingesting Raw Zstandard (`.zst`) Archives by Form Type

The files residing on the 14TB spinning disk (`/home/ralbright/data/edgarstream/edgar/data/`) are **not Parquet files**; they are **raw SEC EDGAR submission archives compressed with Zstandard (`.zst`)**, organized by year and quarter:
```
/home/ralbright/data/edgarstream/edgar/data/
├── 2014/
│   ├── QTR1/
│   │   ├── 0000000000-14-000001.zst
│   │   └── ... (millions of raw submission files)
...
└── 2026/
    └── QTR1/
```

### 2.1 The Sequential Spinning Disk Constraint (14TB HDD)
* **Mechanical Physics**: The 14TB disk (`/dev/sda1`) is mechanical spinning media with an average seek time of ~10ms and a throughput limit of ~150–220 MB/s for sequential reads, but only **80–120 IOPS for random reads**.
* **The Thrashing Trap**: Running recursive directory traversals (`du`, `find`, or unsorted `ls` across 90MB+ directory files) or launching multi-threaded random file readers causes severe disk head actuator thrashing, collapsing throughput from 200 MB/s down to 2 MB/s and hanging system processes.
* **The Solution**: The Rust ingestion engine uses a **Single-Threaded Sequential Readahead Pipeline**:
  - The reader reads raw file paths in physical inode/directory order, buffering large sequential blocks into memory.
  - Decompression and parsing happen entirely in memory across a multi-core Rayon thread pool, decoupling mechanical disk I/O from CPU parsing.

### 2.2 The In-Memory Sniffer & Form-Type Dispatcher
Raw SEC submission files contain an initial SGML envelope known as the `<SEC-HEADER>`. The parser does not need to decompress or parse the entire file to determine what it is:

```rust
// 1. Read first 4KB to 8KB of uncompressed stream into an in-memory ring buffer
let header_chunk = zstd_decoder.read_chunk(8192)?;

// 2. Sniff the CONFORMED SUBMISSION TYPE field
let form_type = sniffer::extract_submission_type(&header_chunk)?;

// 3. Dispatch in-memory to specialized zero-copy parser
match form_type.as_str() {
    "4" | "4/A" | "3" | "5" => {
        form4_parser::parse_xml(&mut zstd_decoder, &mut insider_batch_builder)?;
    }
    "13F-HR" | "13F-HR/A" => {
        form13f_parser::parse_xml(&mut zstd_decoder, &mut holdings_batch_builder)?;
    }
    "10-K" | "10-Q" | "8-K" => {
        narrative_parser::parse_html(&mut zstd_decoder, &mut text_chunks_builder)?;
    }
    "D" | "D/A" => {
        form_d_parser::parse_xml(&mut zstd_decoder, &mut private_offerings_builder)?;
    }
    _ => {
        // Log catalog entry, skip heavy narrative parsing
        catalog_builder.append_skipped(&header_chunk);
    }
}
```

---

## 3. Streaming Arrow Serialization & Iceberg Parquet Output

Once filings are parsed by form type into structured in-memory structs, they are appended to pre-allocated **Apache Arrow RecordBatch Builders**:

```
[In-Memory Parsed Records]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  Arrow RecordBatch (In-Memory Buffer, 128 MB Chunk)    │
│  • High-throughput columnar append                     │
│  • Memory usage strictly bounded (< 1.5 GB total)      │
└──────────────────────────┬─────────────────────────────┘
                           │ Chunk Reaches 128 MB
                           ▼
┌────────────────────────────────────────────────────────┐
│  Parquet Split Serialization + BenoStreamDB Sidecars  │
│  • Writes Snappy/ZSTD compressed Parquet file to S3/HDD│
│  • Computes RoaringBitmap (.idx) on high-cardinality   │
│    fields (CIK, Ticker, Insider Name, Trans Code)      │
│  • Computes TurboQuant TQ8 HNSW (.hnsw) on text chunks │
│  • Updates Iceberg Snapshot Metadata Manifest (JSON)   │
└────────────────────────────────────────────────────────┘
```

* **Zero Intermediate Scratch Files**: Data moves directly from raw `.zst` $\rightarrow$ memory $\rightarrow$ final Iceberg Parquet split + `.idx` / `.hnsw` sidecars.
* **No Database Locks**: Writing to S3/Iceberg is entirely stateless and immutable. There are no PostgreSQL row locks, table locks, or vacuum bloat.

---

## 4. EdgarStream Wagtail (Python / Django) Integration

### 4.1 Keeping PostgreSQL Lean (<20 GB)
In legacy architectures (like InsiderScore), PostgreSQL stores full disclosure text, inverted indexes, and transaction history, resulting in **6 TB+ of disk bloat**, constant autovacuum contention, and multi-thousand-dollar monthly database bills.

In the EdgarStream architecture, PostgreSQL is intentionally restricted to its core competency:
* **What stays in PostgreSQL**:
  - Wagtail CMS page trees, blog posts, and research article drafts.
  - User authentication, roles, session tokens, and Stripe/payment metadata.
  - User-saved screens, watchlists, portfolio tracking lists, and custom alert webhooks.
  - Crawl ingestion queue status flags (where small row-level ACID updates are needed).
  - Total PostgreSQL storage footprint: **< 20 GB** (costs **~$30–$50/month** on AWS RDS or a small local instance).
* **What lives in BenoStreamDB (Iceberg on S3)**:
  - All 12 years of parsed insider transactions (Form 4, 144).
  - All 13F institutional portfolio holdings.
  - All 10-K, 10-Q, 8-K narrative text chunks and 768-dimensional TurboQuant embeddings.
  - All graph adjacency edges and PageRank scores.

### 4.2 Querying via Python Bindings in Wagtail Views
Wagtail controllers and GraphQL/REST API endpoints query BenoStreamDB via the `benostreamdb` Python bindings or Arrow Flight:

```python
# edgarstream/views/insider_views.py
from django.shortcuts import render
from django.http import JsonResponse
import benostreamdb as hsdb

# Connect to local BenoStreamDB flight gateway
client = hsdb.connect("flight://127.0.0.1:50051")

def company_insider_summary(request, ticker):
    """
    Sub-millisecond analytical query executed by BenoStreamDB's
    Rust engine over Iceberg Parquet + RoaringBitmap sidecars.
    """
    query = """
        SELECT 
            filing_date,
            insider_name,
            officer_title,
            transaction_code,
            shares,
            price_per_share,
            total_value,
            post_trade_shares_held,
            is_10b5_1
        FROM edgar.insider_transactions
        WHERE ticker = ? AND filing_date >= '2024-01-01'
        ORDER BY filing_date DESC
        LIMIT 100
    """
    
    # Returns an Arrow Table zero-copy, converted directly to Polars or Pandas
    table = client.sql(query, params=[ticker.upper()])
    trades = table.to_pandas().to_dict(orient="records")
    
    return JsonResponse({"ticker": ticker, "trades": trades})
```

---

## 5. Market Data Integration: Free Open-Source Tiingo BYOK Connector & Forward Alpha

To eliminate market data redistribution licensing fees (which require expensive exchange agreements), **we do not sell or redistribute raw or bundled Tiingo pricing data**. 

Instead, we provide **free, open-source Bring-Your-Own-Key (BYOK) connector code** (`edgar_price_sync`):
* **User-Supplied Tiingo Subscription**: Users configure their personal or institutional Tiingo API key (`TIINGO_API_KEY`) via environment variables or Wagtail settings.
* **Direct Fetch to Local Iceberg**: The open-source pipeline calls Tiingo’s EOD REST API directly under the user's license, fetching split- and dividend-adjusted closing prices (`adjClose`, `adjVolume`) into an Iceberg table (`market.eod_prices`).
* **Automated In-Engine Alpha Attribution**:
  $$\text{Excess Return}_{i, \tau} = R_{i, [t, t+\tau]} - R_{\text{SPY}, [t, t+\tau]} \quad \text{for } \tau \in \{30\text{d}, 90\text{d}, 180\text{d}\}$$
* When an insider files a Form 4 cluster purchase, the engine joins insider trades with the user's locally ingested price table, calculating historical win rates and forward excess returns for Wagtail UI display—100% compliant with financial exchange rules.

---

## 6. Implementation Summary

| Component | Technology | Responsibility | Hardware / Resource Boundary |
| :--- | :--- | :--- | :--- |
| **Ingestion Worker** | **Rust** (`crates/edgar_ingest`) | Read raw `.zst` archives, sniff `<SEC-HEADER>`, parse XML/HTML by form type, serialize Arrow RecordBatches to Parquet | Single sequential thread for disk read; Rayon thread pool for parse; **< 2 GB RAM** |
| **Storage & Indexing** | **BenoStreamDB + Iceberg** | Apache Iceberg Parquet splits on S3 / local disk; RoaringBitmap `.idx` filters; TurboQuant TQ8 `.hnsw` vectors | Bounded by OS page cache; **< 4 GB RAM** |
| **Query Gateway** | **Rust** (`benostreamdb-flight`) | Arrow Flight SQL (Port 50051) & OpenSearch REST (Port 9200) | Concurrent async Tokio runtime |
| **Frontend & API** | **Python / Wagtail** | CMS, analyst terminal UI, user authentication, watchlists, Graph RAG visualizations | Standard Gunicorn/Django web workers; **Lean PostgreSQL (<20 GB)** |
