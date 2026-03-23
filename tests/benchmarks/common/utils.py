"""
Common utilities for benchmark suite.

Provides dataset generation, metrics collection, and MinIO setup.
"""
import time
import psutil
import os
import numpy as np
import pyarrow as pa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BenchmarkMetrics:
    """Collect and track benchmark metrics."""
    
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    latencies: List[float] = field(default_factory=list)
    throughput: Optional[float] = None
    memory_start_mb: float = field(default_factory=lambda: psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    memory_end_mb: Optional[float] = None
    
    def record_latency(self, latency_ms: float):
        """Record a single query latency in milliseconds."""
        self.latencies.append(latency_ms)
    
    def finish(self, total_operations: int = None):
        """Mark benchmark as complete and calculate final metrics."""
        self.end_time = time.time()
        self.memory_end_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        if total_operations:
            elapsed = self.end_time - self.start_time
            self.throughput = total_operations / elapsed if elapsed > 0 else 0
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        if not self.latencies:
            return {
                "name": self.name,
                "throughput": self.throughput,
                "elapsed_sec": self.end_time - self.start_time if self.end_time else None,
                "memory_delta_mb": self.memory_end_mb - self.memory_start_mb if self.memory_end_mb else None,
            }
        
        latencies = np.array(self.latencies)
        return {
            "name": self.name,
            "throughput": self.throughput,
            "latency_p50_ms": np.percentile(latencies, 50),
            "latency_p95_ms": np.percentile(latencies, 95),
            "latency_p99_ms": np.percentile(latencies, 99),
            "latency_mean_ms": np.mean(latencies),
            "latency_min_ms": np.min(latencies),
            "latency_max_ms": np.max(latencies),
            "elapsed_sec": self.end_time - self.start_time if self.end_time else None,
            "memory_delta_mb": self.memory_end_mb - self.memory_start_mb if self.memory_end_mb else None,
        }
    
    def print_summary(self):
        """Print human-readable summary."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*60}")
        
        if stats.get("throughput"):
            print(f"Throughput: {stats['throughput']:,.0f} ops/sec")
        
        if "latency_p50_ms" in stats:
            print(f"Latency (ms):")
            print(f"  p50: {stats['latency_p50_ms']:.2f}")
            print(f"  p95: {stats['latency_p95_ms']:.2f}")
            print(f"  p99: {stats['latency_p99_ms']:.2f}")
            print(f"  mean: {stats['latency_mean_ms']:.2f}")
        
        if stats.get("elapsed_sec"):
            print(f"Elapsed: {stats['elapsed_sec']:.2f} sec")
        
        if stats.get("memory_delta_mb"):
            print(f"Memory delta: {stats['memory_delta_mb']:.1f} MB")
        
        print(f"{'='*60}\n")


def generate_openai_embeddings(n: int = 1_000_000, dim: int = 1536, seed: int = 42) -> pa.Table:
    """
    Generate synthetic OpenAI-style embeddings.
    
    Args:
        n: Number of vectors to generate
        dim: Dimensionality (1536 for Ada-002)
        seed: Random seed for reproducibility
    
    Returns:
        PyArrow table with id, embedding, and metadata columns
    """
    np.random.seed(seed)
    
    print(f"Generating {n:,} vectors ({dim}D)...")
    
    # Generate embeddings in batches to avoid memory issues
    batch_size = 10_000
    batches = []
    
    for i in range(0, n, batch_size):
        current_batch_size = min(batch_size, n - i)
        
        # Generate normalized vectors (like real embeddings)
        vectors = np.random.randn(current_batch_size, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Generate metadata
        ids = np.arange(i, i + current_batch_size, dtype=np.int64)
        categories = np.random.choice(['A', 'B', 'C', 'D', 'E'], size=current_batch_size)
        user_ids = np.random.randint(1, 10000, size=current_batch_size, dtype=np.int32)
        
        # Create batch
        embedding_array = pa.FixedSizeListArray.from_arrays(vectors.flatten(), dim)
        batch = pa.RecordBatch.from_arrays([
            pa.array(ids),
            embedding_array,
            pa.array(categories),
            pa.array(user_ids),
        ], names=['id', 'embedding', 'category', 'user_id'])
        
        batches.append(batch)
        
        if (i + batch_size) % 100_000 == 0:
            print(f"  Generated {i + batch_size:,} vectors...")
    
    table = pa.Table.from_batches(batches)
    print(f"✓ Generated {n:,} vectors")
    return table


def generate_tpch_lineitem(scale_factor: float = 0.01, seed: int = 42) -> pa.Table:
    """
    Generate TPC-H lineitem-style data.
    
    Args:
        scale_factor: TPC-H scale factor (0.01 = ~60K rows, 1.0 = ~6M rows)
        seed: Random seed
    
    Returns:
        PyArrow table with lineitem-style columns
    """
    np.random.seed(seed)
    
    # TPC-H lineitem has ~6M rows at scale factor 1
    n_rows = int(6_000_000 * scale_factor)
    
    print(f"Generating TPC-H lineitem data (SF={scale_factor}, {n_rows:,} rows)...")
    
    # Generate data
    orderkeys = np.random.randint(1, int(1_500_000 * scale_factor), size=n_rows, dtype=np.int64)
    partkeys = np.random.randint(1, int(200_000 * scale_factor), size=n_rows, dtype=np.int64)
    suppkeys = np.random.randint(1, int(10_000 * scale_factor), size=n_rows, dtype=np.int32)
    linenumbers = np.random.randint(1, 8, size=n_rows, dtype=np.int32)
    quantities = np.random.randint(1, 51, size=n_rows, dtype=np.int32)
    extendedprices = np.random.uniform(900, 105000, size=n_rows).astype(np.float64)
    discounts = np.random.uniform(0, 0.1, size=n_rows).astype(np.float64)
    taxes = np.random.uniform(0, 0.08, size=n_rows).astype(np.float64)
    
    # Generate dates (1992-1998)
    start_date = np.datetime64('1992-01-01')
    end_date = np.datetime64('1998-12-31')
    date_range = (end_date - start_date).astype(int)
    shipdates = start_date + np.random.randint(0, date_range, size=n_rows).astype('timedelta64[D]')
    
    returnflags = np.random.choice(['A', 'R', 'N'], size=n_rows)
    linestatuses = np.random.choice(['O', 'F'], size=n_rows)
    
    table = pa.table({
        'l_orderkey': orderkeys,
        'l_partkey': partkeys,
        'l_suppkey': suppkeys,
        'l_linenumber': linenumbers,
        'l_quantity': quantities,
        'l_extendedprice': extendedprices,
        'l_discount': discounts,
        'l_tax': taxes,
        'l_returnflag': returnflags,
        'l_linestatus': linestatuses,
        'l_shipdate': shipdates,
    })
    
    print(f"✓ Generated {n_rows:,} rows")
    return table


def format_results_markdown(results: List[Dict]) -> str:
    """Format benchmark results as markdown table."""
    if not results:
        return "No results to display"
    
    # Build header
    headers = list(results[0].keys())
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Build rows
    for result in results:
        row = []
        for key in headers:
            value = result[key]
            if isinstance(value, float):
                row.append(f"{value:.2f}")
            elif isinstance(value, int):
                row.append(f"{value:,}")
            else:
                row.append(str(value))
        md += "| " + " | ".join(row) + " |\n"
    
    return md


def save_results(results: List[Dict], filename: str):
    """Save benchmark results to file."""
    import json
    
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{filename}_{timestamp}.json")
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {filepath}")
    
    # Also save markdown
    md_filepath = filepath.replace('.json', '.md')
    with open(md_filepath, 'w') as f:
        f.write(format_results_markdown(results))
    
    print(f"✓ Markdown saved to {md_filepath}")
