# 03. Financial Model & Unit Economics

**Status:** Financial Architecture & Projection Model  
**Target:** 5-Year Horizon (Year 1 to Year 5)  

---

## 1. Unit Economics & Gross Margins

Because BenoStreamDB operates directly on object storage (AWS S3, Google Cloud Storage, MinIO) with persistent sidecars (`.idx`, `.hnsw`, `.inv`) and metadata-only manifest commits, its COGS (Cost of Goods Sold) are fundamentally lower than traditional databases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 5-YEAR REVENUE & EXPENSE PROJECTIONS                        │
├────────────────────┬──────────┬──────────┬──────────┬──────────┬────────────┤
│ METRIC             │ YEAR 1   │ YEAR 2   │ YEAR 3   │ YEAR 4   │ YEAR 5     │
├────────────────────┼──────────┼──────────┼──────────┼──────────┼────────────┤
│ Community Installs │ 10,000   │ 50,000   │ 150,000  │ 350,000  │ 750,000    │
│ Team Customers     │ 15       │ 80       │ 250      │ 600      │ 1,200      │
│ Enterprise Logos   │ 2        │ 12       │ 35       │ 80       │ 160        │
├────────────────────┼──────────┼──────────┼──────────┼──────────┼────────────┤
│ Team ARR ($499/mo) │ $90,000  │ $480,000 │ $1.50M   │ $3.60M   │ $7.20M     │
│ Enterprise ARR     │ $120,000 │ $720,000 │ $2.80M   │ $7.20M   │ $16.00M    │
│ EdgarStreamDB Rev  │ $50,000  │ $250,000 │ $600,000 │ $1.20M   │ $2.00M     │
├────────────────────┼──────────┼──────────┼──────────┼──────────┼────────────┤
│ TOTAL ARR          │ $260,000 │ $1.45M   │ $4.90M   │ $12.00M  │ $25.20M    │
├────────────────────┼──────────┼──────────┼──────────┼──────────┼────────────┤
│ Hosting & Cloud    │ $18,000  │ $65,000  │ $180,000 │ $420,000 │ $850,000   │
│ Gross Margin (%)   │ 93.1%    │ 95.5%    │ 96.3%    │ 96.5%    │ 96.6%      │
├────────────────────┼──────────┼──────────┼──────────┼──────────┼────────────┤
│ Headcount (FTEs)   │ 2        │ 5        │ 12       │ 24       │ 45         │
│ Net Burn / Profit  │ ($85k)   │ $250k    │ $1.80M   │ $5.20M   │ $11.50M    │
└────────────────────┴──────────┴──────────┴──────────┴──────────┴────────────┘
```

---

## 2. Real-World Production Cost Teardown: Legacy vs. BenoStreamDB

A typical mid-sized institutional intelligence platform (such as InsiderScore / VerityData) running a 6 TB disclosure corpus spends approximately **$20,000/month** on legacy infrastructure:

### Legacy Relational & Search Stack ($240,000/Year)
* **PostgreSQL Primary + Read Standby**: 6 TB provisioned high-IOPS `gp3`/`io2` EBS storage with 64GB RAM instances $\rightarrow$ **~$10,500/month**.
* **OpenSearch / Elasticsearch Cluster**: 3 master nodes + 6 data nodes (r6g.2xlarge, 64GB RAM each) to index text chunks $\rightarrow$ **~$6,000/month**.
* **Kubernetes Worker Fleet**: High-memory nodes handling uncompressed SGML and XML parsing in Python/Java $\rightarrow$ **~$3,500/month**.
* **Total Legacy Monthly Bill**: **~$20,000/month ($240,000/year)**.

### Modern BenoStreamDB + Wagtail Stack ($17,400/Year)
* **AWS S3 Standard (2 TB Compressed Parquet + Sidecars)**: 2,048 GB $\times$ $0.023$/GB $\rightarrow$ **~$47/month**.
* **Lean PostgreSQL (Wagtail CMS, Users, Watchlists)**: 20 GB `db.t4g.medium` RDS $\rightarrow$ **~$45/month**.
* **BenoStreamDB Stateless Query Nodes**: 2 $\times$ 4 GB RAM instances (ARM Graviton3 `c7g.large`) $\rightarrow$ **~$108/month**.
* **Stateless K8s Rust Ingestion Workers**: Spot instances running sequential Zstd decompressors $\rightarrow$ **~$1,250/month**.
* **Total Modern Monthly Bill**: **~$1,450/month ($17,400/year)**.

> **Net Annual Infrastructure Savings**: **$222,600/year (92.8% reduction)**.

---

## 3. Solo-Founder Bootstrap & Academic Synergies

* **Zero Dilution Required**: Capturing just **5 to 10 hedge fund contracts** for EdgarStream / EdgarStreamDB ($30,000 to $50,000/year) yields **$250,000 to $500,000 ARR** with 95%+ gross margin.
* **Fall 2026 Medical Leave & TBI Recovery**: The founder is on medical leave from PhD coursework during Fall 2026 due to TBI recovery. Engineering is paced sustainably without artificial academic deadlines, focusing on core infrastructure (Graph RAG v0.8.0, local benchmarks).
* **Turn-Key Academic Platform for January 2027**: Resuming PhD work in January 2027, the 12-year EDGAR dataset and Graph RAG engine serve as a turn-key empirical laboratory to run the Carhart 4-factor asset pricing model and Insider Sentiment Factor ($\text{ISF}_t$), establishing quantitative authority without venture capital dependency.
