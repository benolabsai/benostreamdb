#!/usr/bin/env python3
"""Unified Benchmark Suite Summary Generator.

Scans result artifacts across:
- benchmarks/ann_benchmarks/results/
- benchmarks/graph/results/
- benchmarks/sql/results/
- benchmarks/results/
And generates a comprehensive executive summary in benchmarks/results/BENCHMARK_REPORT.md.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def generate_summary():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "BENCHMARK_REPORT.md")

    sections = [
        "# BenoStreamDB Comprehensive Benchmark Report",
        "",
        f"- **Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Platform**: {platform.platform()}",
        f"- **Python**: {platform.python_version()}",
        "",
        "---",
        "",
    ]

    # 1. Vector ANN Benchmarks
    sections.append("## 1. Vector ANN Performance (SIFT / HNSW / TurboQuant)")
    sections.append("")
    ann_files = glob.glob(os.path.join(BASE_DIR, "ann_benchmarks", "results", "*.md"))
    if ann_files:
        for f in sorted(ann_files):
            sections.append(f"### Results from `{os.path.basename(f)}`")
            sections.append("")
            with open(f, encoding="utf-8") as fh:
                sections.append(fh.read().strip())
            sections.append("")
    else:
        sections.append("*No vector ANN results found.*")
    sections.append("")

    # 2. Graph Analytics Benchmarks
    sections.append("## 2. Graph Analytics Performance (BenoStreamDB vs NetworkX)")
    sections.append("")
    graph_files = glob.glob(os.path.join(BASE_DIR, "graph", "results", "*.md"))
    if graph_files:
        for f in sorted(graph_files):
            sections.append(f"### Results from `{os.path.basename(f)}`")
            sections.append("")
            with open(f, encoding="utf-8") as fh:
                sections.append(fh.read().strip())
            sections.append("")
    else:
        sections.append("*No graph benchmark results found.*")
    sections.append("")

    # 3. SQL OLAP Benchmarks
    sections.append("## 3. SQL OLAP Performance (ClickBench Q0–Q9: BenoStreamDB vs DuckDB vs DataFusion)")
    sections.append("")
    sql_files = glob.glob(os.path.join(BASE_DIR, "sql", "results", "*.md"))
    if sql_files:
        for f in sorted(sql_files):
            sections.append(f"### Results from `{os.path.basename(f)}`")
            sections.append("")
            with open(f, encoding="utf-8") as fh:
                sections.append(fh.read().strip())
            sections.append("")
    else:
        sections.append("*No SQL benchmark results found.*")
    sections.append("")

    # 4. Lexical & Hybrid Search Benchmarks
    sections.append("## 4. Lexical & Hybrid Search Performance (BEIR SciFact: BM25 vs Tantivy & Hybrid RRF)")
    sections.append("")
    # Roll the per-engine JSON records (written by the BEIR harness) into one
    # table, so the lexical/hybrid numbers sit alongside the vector/graph/SQL
    # ones in the consolidated report.
    beir_json = glob.glob(os.path.join(BASE_DIR, "beir", "results", "*.json"))
    if beir_json:
        sections.append("### Rolled-up results (JSON)")
        sections.append("")
        sections.append(
            "| Engine | Backend | Workload | Dataset | Recall@k | nDCG@k | MRR@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |"
        )
        sections.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for f in sorted(beir_json):
            try:
                with open(f, encoding="utf-8") as fh:
                    r = json.load(fh)
            except (OSError, ValueError):
                continue
            sections.append(
                f"| {r.get('engine')} | {r.get('backend', '-')} | {r.get('workload')} | "
                f"{r.get('dataset')} | "
                f"{r.get('recall_at_k')} | {r.get('ndcg_at_k')} | {r.get('mrr_at_k')} | "
                f"{r.get('qps')} | {r.get('p50_ms')} | {r.get('p99_ms')} | "
                f"{r.get('build_s')} | {r.get('index_mb')} |"
            )
        sections.append("")
    beir_files = glob.glob(os.path.join(BASE_DIR, "beir", "results", "*.md"))
    if beir_files:
        for f in sorted(beir_files):
            sections.append(f"### Results from `{os.path.basename(f)}`")
            sections.append("")
            with open(f, encoding="utf-8") as fh:
                sections.append(fh.read().strip())
            sections.append("")
    else:
        sections.append("*No BEIR benchmark results found.*")
    sections.append("")

    # 5. Production Workload & Concurrency
    sections.append("## 5. Production Workload & Concurrency Performance")
    sections.append("")

    conc_md = os.path.join(RESULTS_DIR, "production_concurrency.md")
    if os.path.exists(conc_md):
        with open(conc_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")

    recov_md = os.path.join(RESULTS_DIR, "production_recovery.md")
    if os.path.exists(recov_md):
        with open(recov_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")

    throttling_md = os.path.join(RESULTS_DIR, "production_throttling.md")
    if os.path.exists(throttling_md):
        with open(throttling_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")

    sections.append("### Tail Latency Under Background Maintenance (Compaction)")
    sections.append("")
    prod_json = os.path.join(RESULTS_DIR, "production_maintenance.json")
    if os.path.exists(prod_json):
        with open(prod_json, encoding="utf-8") as fh:
            data = json.load(fh)
        sections.append("| Metric | Value |")
        sections.append("|---|---|")
        for k, v in data.items():
            sections.append(f"| {k} | {v} |")
    else:
        sections.append("*No production maintenance results found.*")
    sections.append("")

    # Multi-Writer Concurrency Correctness (§7.7)
    mw_md = os.path.join(RESULTS_DIR, "production_concurrency_correctness.md")
    if os.path.exists(mw_md):
        with open(mw_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")

    # 6. Pathological Filters & Differential Oracle (§7.5)
    sections.append("## 6. Pathological Filters & Differential Oracle Benchmark")
    sections.append("")
    filters_md = os.path.join(RESULTS_DIR, "production_filters.md")
    if os.path.exists(filters_md):
        with open(filters_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")
    else:
        sections.append("*No pathological filter results found.*")
    sections.append("")

    # 7. Skewed Data & Power-Law Distribution Benchmark (§7.6)
    sections.append("## 7. Skewed Data & Power-Law Distribution Benchmark")
    sections.append("")
    skew_md = os.path.join(RESULTS_DIR, "production_skew.md")
    if os.path.exists(skew_md):
        with open(skew_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")
    else:
        sections.append("*No skewed data benchmark results found.*")
    sections.append("")

    # 8. Mixed Workload Soak & Endurance Benchmark (§7.8)
    sections.append("## 8. Mixed Workload Soak & Endurance Benchmark")
    sections.append("")
    soak_md = os.path.join(RESULTS_DIR, "production_soak.md")
    if os.path.exists(soak_md):
        with open(soak_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")
    else:
        sections.append("*No soak benchmark results found.*")
    sections.append("")

    # 9. Docker Competitor Matrix (single shared hardware envelope)
    sections.append("## 9. Docker Competitor Matrix (shared hardware envelope)")
    sections.append("")
    rollup_md = os.path.join(BASE_DIR, "competitors", "results", "rollup.md")
    if os.path.exists(rollup_md):
        with open(rollup_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")
    else:
        sections.append("*No Docker competitor matrix results found.*")
    sections.append("")

    content = "\n".join(sections) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated comprehensive report at {report_path}")


if __name__ == "__main__":
    generate_summary()
