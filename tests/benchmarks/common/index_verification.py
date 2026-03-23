"""
Utility functions for verifying index existence in benchmarks.
"""
import time
from typing import Optional
from hyperstreamdb import Table
import numpy as np


def verify_index_built(table: Table, column: str, max_wait: int = 60, check_interval: float = 1.0) -> bool:
    """
    Verify that an index was built for a given column.
    
    Uses a heuristic: performs a small test search and checks latency.
    If the search is fast (< 100ms for small dataset), index likely exists.
    If it's slow (> 500ms), index probably doesn't exist (falling back to full scan).
    
    Args:
        table: The Table instance
        column: Column name to check
        max_wait: Maximum time to wait in seconds
        check_interval: How often to check (seconds)
    
    Returns:
        True if index exists, False if timeout
    """
    start_time = time.time()
    
    # Generate a test query vector
    test_query = np.random.randn(1536).astype(np.float32)
    test_query = test_query / np.linalg.norm(test_query)
    
    while time.time() - start_time < max_wait:
        try:
            # Perform a small test search
            # If index exists, this should be fast (< 100ms)
            # If index doesn't exist, this will be slow (> 500ms) due to full scan
            search_start = time.time()
            results = table.search(column=column, query=test_query.tolist(), k=10)
            search_latency_ms = (time.time() - search_start) * 1000
            
            # Heuristic: Fast search (< 100ms) likely means index exists
            # Slow search (> 500ms) likely means full scan (no index)
            if search_latency_ms < 100:
                print(f"✓ Index verified: search latency {search_latency_ms:.2f}ms indicates index exists")
                return True
            elif search_latency_ms > 500:
                # Likely no index, wait a bit more
                print(f"  Index check: search latency {search_latency_ms:.2f}ms suggests index not ready, waiting...")
                time.sleep(check_interval)
            else:
                # Ambiguous, wait and check again
                time.sleep(check_interval)
        except Exception as e:
            # If search fails, index definitely not ready
            print(f"  Index check: search failed ({e}), waiting...")
            time.sleep(check_interval)
    
    return False  # Timeout


def wait_for_index_built(table: Table, column: str, max_wait: int = 60) -> None:
    """
    Wait for index to be built, raising an error if it doesn't appear.
    
    Raises:
        TimeoutError: If index not built within max_wait seconds
    """
    print(f"\nVerifying index for column '{column}' is built...")
    if not verify_index_built(table, column, max_wait):
        raise TimeoutError(
            f"Index for column '{column}' was not built within {max_wait} seconds. "
            "This may indicate an issue with index building or the benchmark setup. "
            "Queries may be falling back to full scans, which would explain poor performance."
        )
    print(f"✓ Index for column '{column}' verified")
