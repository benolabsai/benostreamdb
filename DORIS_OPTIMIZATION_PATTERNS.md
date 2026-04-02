# Apache Doris Optimization Patterns for HyperStreamDB

## Executive Summary
This document extracts key optimization patterns from Apache Doris (OLAP database) that can enhance HyperStreamDB's query execution, indexing strategy, and data ingestion pipeline. These are architectural lessons, not direct code copies.

---

## 1. Vectorized Execution Model

### Doris Pattern: Block-Based Execution
**How Doris Does It:**
- Divides data into fixed-size blocks (typically 4K or 64K rows)
- Processes blocks through vectorized operators (SIMD-friendly)
- Maintains CPU cache locality across entire block
- Reduces function call overhead vs row-at-a-time execution

**Metrics:**
- 10-100x faster than row-oriented for analytical workloads
- L1/L2 cache hit rate >95% with proper block sizing
- Minimal branch mispredictions with predictable data patterns

### HyperStreamDB Application:
**Current State:** Your segment-based approach is already block-oriented

**Enhancement Opportunities:**

1. **HNSW Cluster Batch Processing**
   ```rust
   // Instead of processing vectors one-by-one in distance calculation
   // Process entire cluster batches with SIMD instructions
   
   pub struct VectorBatch {
       vectors: Vec<Vec<f32>>,      // Multiple vectors
       cluster_id: u32,
       batch_size: usize,           // Optimal: 64-256 vectors
   }
   
   pub fn distance_batch_l2(
       query: &[f32],
       batch: &VectorBatch,
   ) -> Vec<f32> {
       // SIMD-friendly: process 4-8 vectors per CPU cycle
       // Use avx2 or neon for SIMD instructions
   }
   ```

2. **Inverted Index Batch Decoding**
   ```
   Instead of:     docids[0], docids[1], docids[2]...
   
   Use batch:      docids[0..64] decoded in one operation
                   - Delta compression (Doris pattern)
                   - Bit-packing for integer arrays
   ```

3. **Optimal Batch Size Analysis**
   - Profile L1 cache size (typically 32KB)
   - Tune block size to fit: vectors + metadata + query state
   - Recommended: 128-256 vectors per batch for f32 embeddings

---

## 2. Columnar Compression with Adaptive Codec Selection

### Doris Pattern: Automatic Codec Selection
**How Doris Does It:**
- Analyzes column cardinality and value distribution
- Tries multiple codecs (delta, RLE, LZ4, zstd) on sample
- Picks codec with best compression ratio for that column
- Cache codec decision in metadata

**Typical Results:**
- Numeric columns: 5-10x compression (delta + bit-packing)
- String columns: 2-4x compression (dictionary + RLE)
- Sparse columns: 50-100x compression (RLE)

### HyperStreamDB Application:

1. **Column-Aware Compression Decision**
   ```rust
   pub enum CodecChoice {
       // For embedding vectors (dense, float32)
       ZstdFloat {
           level: i32,  // 3-10, tuned per segment
       },
       // For IDs and indices (integer sequences)
       DeltaBitpack {
           bits: u32,   // 8, 16, 32 bits
       },
       // For string columns (high cardinality)
       DictionaryRLE,
       // For title/metadata (often repetitive)
       LZ4Fast,
   }
   ```

2. **Codec Selection at Flush Time**
   ```rust
   // In your commit() path:
   // 1. Sample first 1% of each column
   // 2. Try 3-4 codec options
   // 3. Pick best compression ratio
   // 4. Store choice in segment metadata
   ```

3. **Embedding-Specific: IEEE 754 Quantization**
   - Most embeddings don't need full f32 precision
   - Quantize to f16 (BFloat16) after normalization
   - Save 50% space, negligible accuracy loss
   - Still need for RAG: ≈99.5% recall at k=3-5

---

## 3. Column Statistics for Query Optimization

### Doris Pattern: Zone Maps + Bloom Filters

**Zone Maps (Min/Max per Block):**
- Track MIN/MAX for each column in each row group
- Enables block skipping: `WHERE age > 100` skips blocks where max_age ≤ 100
- Tiny overhead: 2 int64 per column per block

**Bloom Filters:**
- For SET membership: `WHERE user_id IN (...)`
- False positive rate: 1-5%
- Found useful for sparse predicates

### HyperStreamDB Application:

1. **Segment-Level Zone Maps**
   ```rust
   pub struct SegmentStats {
       column_name: String,
       min_value: ScalarValue,    // MIN for this segment
       max_value: ScalarValue,    // MAX for this segment
       null_count: u64,
       
       // For strings: length stats
       min_length: u32,
       max_length: u32,
       
       // For embeddings: value range
       min_norm: f32,
       max_norm: f32,
   }
   ```

2. **Predicate Pushdown with Statistics**
   ```rust
   // In your scalar filter execution:
   fn can_skip_segment(
       filter: &FilterPredicate,
       stats: &SegmentStats,
   ) -> bool {
       match filter {
           // WHERE title = 'X': never skip (need bloom filter)
           // WHERE age > 50: skip if segment max_age ≤ 50
           FilterPredicate::Greater(col, val) => {
               stats.max_value < val  // Can skip
           }
       }
   }
   ```

3. **Vector Value Range Statistics**
   ```rust
   // For embeddings, track:
   pub struct VectorStats {
       min_norm: f32,              // Smallest L2 norm in segment
       max_norm: f32,              // Largest L2 norm
       mean_norm: f32,             // Average norm
       
       // Per-dimension ranges (useful for filtering)
       dim_min: Vec<f32>,          // MIN value per dimension
       dim_max: Vec<f32>,          // MAX value per dimension
   }
   
   // Use to skip non-promising clusters in HNSW search
   ```

---

## 4. Write-Ahead Logging for Atomic Commits

### Doris Pattern: Memtable + Write-Ahead Log (WAL)

**How Doris Does It:**
- Incoming writes buffer in memtable (in-memory)
- All modifications logged to WAL first (durability)
- Once logged, WAL pages are immutable
- Memtable flush merges all pending transactions
- Recovery: replay WAL entries into clean state

**Benefits:**
- No durability vs speed tradeoff
- Can guarantee ACID semantics
- Point-in-time recovery possible

### HyperStreamDB Application:

1. **Write Path Enhancement**
   ```rust
   pub struct WriteBuffer {
       pending_rows: Vec<Row>,
       wal_entry_id: u64,
       created_at: Instant,
   }
   
   pub fn add_row(&mut self, row: Row) -> Result<()> {
       // 1. Append to WAL (fsync every N entries)
       write_wal_entry(&row)?;
       
       // 2. Add to memtable
       self.pending_rows.push(row);
       
       // 3. Return immediately (durability guaranteed)
       Ok(())
   }
   ```

2. **Atomic Batch Flush**
   ```rust
   pub fn commit(&mut self) -> Result<()> {
       // Current: flush() then metadata
       // Enhanced: 
       // 1. Write segment file (atomic operation)
       // 2. Write indexes (can be retried)
       // 3. Update manifest (single atomic write)
       // 4. Mark WAL entries as committed
       
       // If crash before step 4: recovery replays WAL safely
   }
   ```

3. **Checkpointing Strategy**
   ```rust
   pub fn checkpoint_manifest(&mut self) -> Result<()> {
       // Periodically:
       // 1. Snap current manifest
       // 2. Mark old WAL entries as safe to delete
       // 3. Clean up old WAL files
       // 4. Reduces recovery time on startup
   }
   ```

---

## 5. Segment Pruning and Multi-Level Indexing

### Doris Pattern: Hierarchical Index Structure

**How Doris Does It:**
- Global index: all segments metadata (loaded once)
- Segment-level index: statistics + inverted indexes
- Block-level index: zone maps within segment
- Query planner: uses global index to select candidates
- Detailed scan only on candidate segments

**Decision Tree:**
```
Query: WHERE title = 'NFL' AND score > 0.8

1. Global phase: Which segments might have results?
   - Use inverted index on title (if built)
   - Use zone map on score (if available)
   
2. Segment phase: Read only matching segments
   
3. Block phase: Skip blocks within segment
```

### HyperStreamDB Application:

1. **Segment Metadata Index**
   ```rust
   pub struct GlobalSegmentIndex {
       segments: HashMap<SegmentId, SegmentMetadata>,
       
       // String inverted indexes
       title_inverted: InvertedIndex,
       context_inverted: InvertedIndex,
       
       // Column statistics
       zone_maps: HashMap<String, ZoneMapIndex>,
       
       // Vector metadata
       vector_bounds: HashMap<String, VectorStats>,
   }
   
   pub fn find_candidate_segments(
       &self,
       filter: &ScalarFilter,
   ) -> Vec<SegmentId> {
       // Use inverted indexes and zone maps to narrow down
       // Example: title = 'NFL' -> only 5 segments
   }
   ```

2. **Two-Phase Execution**
   ```rust
   pub fn execute_query(&self, plan: &QueryPlan) -> Result<Vec<Row>> {
       // Phase 1: Determine execution scope
       let candidate_segs = self.find_candidate_segments(&plan.scalar_filters);
       
       // Phase 2: Detailed scan
       let mut results = vec![];
       for seg_id in candidate_segs {
           let rows = self.scan_segment(seg_id, &plan)?;
           results.extend(rows);
       }
       
       Ok(results)
   }
   ```

3. **Compound Filter Optimization**
   ```rust
   // Instead of: scan all segments, filter by both conditions
   // Do this:
   pub fn optimize_filter(&self, filter: &ComplexFilter) -> ExecutionPlan {
       // 1. Decompose into independent predicates
       let scalar_preds = extract_scalar_predicates(filter);
       let vector_preds = extract_vector_predicates(filter);
       
       // 2. Get candidates from scalar first (faster, narrower)
       let candidates = self.find_candidates(&scalar_preds);
       
       // 3. Apply vector search only on candidates
       let results = self.vector_search(&candidates, vector_preds);
       
       results
   }
   ```

---

## 6. Adaptive Index Building

### Doris Pattern: Deferred Index Construction

**How Doris Does It:**
- Data compaction happens first (merges small files)
- Once stable, deferred indexes are built asynchronously
- Query answerable without indexes (just slower)
- Background thread builds indexes, doesn't block ingestion
- New queries benefit from indexes as they're built

**Benefit:** Fast write path + Eventual fast read path (no tradeoff)

### HyperStreamDB Application:

1. **Deferred HNSW Building**
   ```rust
   pub enum IndexState {
       NotBuilt,           // Just ingested
       Building(Arc<Mutex<BuildProgress>>),  // Background task
       Complete,           // Ready for use
   }
   
   pub struct Segment {
       vector_index: Arc<RwLock<IndexState>>,
       // ...
   }
   
   pub fn commit(&mut self) -> Result<()> {
       // 1. Flush data (takes 100ms)
       self.flush_segment()?;
       
       // 2. Mark index as "Building"
       *self.vector_index.write() = IndexState::Building(...);
       
       // 3. Spawn background task
       let segment_clone = self.clone();
       thread::spawn(move || {
           let hnsw = build_hnsw(&segment_clone.data);
           *segment_clone.vector_index.write() = IndexState::Complete(hnsw);
       });
       
       // 4. Return immediately (commit done)
       Ok(())
   }
   ```

2. **Query Planner Awareness**
   ```rust
   pub fn plan_vector_search(
       &self,
       params: &VectorSearchParams,
   ) -> Result<VectorSearchPlan> {
       match &*self.vector_index.read() {
           IndexState::Complete(hnsw) => {
               // Use index
               VectorSearchPlan::indexed(hnsw.clone())
           }
           IndexState::Building(_) => {
               // Index building, but we can still search
               VectorSearchPlan::bruteforce()  // Slow but correct
           }
           IndexState::NotBuilt => {
               VectorSearchPlan::bruteforce()
           }
       }
   }
   ```

---

## 7. Memory-Efficient Inverted Index with Delta Compression

### Doris Pattern: Block-Based Inverted Index

**How Doris Does It:**
- Inverted lists stored as blocks of docids
- Within block: delta encoding (store differences, not absolute)
- Across blocks: simple compression (LZ4)
- Seek to block, decompress, scan deltas

**Space Savings:** 50-70% vs naive list storage

### HyperStreamDB Application:

```rust
// Instead of: vec![1, 5, 12, 23, 45, 98, 156...]
// Store as:  block[0] = [1, 4, 7, 11, 22, 53, 58]  (deltas)

pub struct DeltaCompressedList {
    blocks: Vec<Vec<u32>>,  // Each block stores deltas
    block_offsets: Vec<u64>, // File offsets for seeking
}

pub fn decode_range(&self, start: u32, end: u32) -> Vec<u32> {
    // Find blocks containing [start..end]
    let mut result = vec![];
    let mut current = 0;
    
    for block in &self.blocks {
        for &delta in block {
            if current >= start && current < end {
                result.push(current);
            }
            current += delta as u32;
        }
    }
    result
}
```

---

## 8. Cardinality-Aware Dictionary Encoding

### Doris Pattern: Adaptive Dictionary Selection

**How Doris Does It:**
- High cardinality (>10k unique): Store without dictionary
- Medium cardinality (100-10k): Small dictionary + table
- Low cardinality (<100): Global dictionary + indices

**Decision:** Made during segment flush, stored in metadata

### HyperStreamDB Application:

```rust
pub fn choose_string_encoding(
    column: &[String],
) -> StringEncoding {
    let unique_count = column.iter().collect::<HashSet<_>>().len();
    let total_count = column.len();
    let cardinality_ratio = unique_count as f64 / total_count as f64;
    
    if cardinality_ratio > 0.5 {
        // High cardinality: Store raw with compression
        StringEncoding::RawZstd
    } else if unique_count < 1000 {
        // Medium cardinality: Dictionary + indices
        StringEncoding::Dictionary {
            dict: build_dict(column),
            indices: encode_indices(column),
        }
    } else {
        // Low cardinality: Packed dictionary
        StringEncoding::CompactDictionary {
            // See Doris technique
        }
    }
}
```

---

## 9. Efficient Null Handling

### Doris Pattern: Null Bitmap with RLE

**How Doris Does It:**
- Track nulls separately from data
- Null bitmap: 1 bit per value
- Compress null bitmap with RLE (many consecutive non-nulls)
- Reduces space from 1 byte to ~1 bit per null

### HyperStreamDB Application:

Already using Parquet's null handling, but can optimize:

```rust
// Instead of nullable<T>, use explicit null bitmap for vectors
pub struct NullableColumn {
    data: Vec<f32>,          // Only non-null values
    null_bitmap: Vec<u64>,   // Bit 1 = not null, 0 = null
    null_count: u32,
}

// Saves 7x space for nullable float columns (if sparse nulls)
```

---

## 10. Query Cache with Segment Versioning

### Doris Pattern: Query Result Cache

**How Doris Does It:**
- Cache query results keyed by (query_text, segment_versions)
- When segments change: invalidate cache entries
- Useful for repeated RAG queries

### HyperStreamDB Application:

**For RAG workloads (repeated queries on same knowledge base):**

```rust
pub struct QueryCache {
    cache: HashMap<QueryKey, CachedResult>,
    segment_versions: HashMap<SegmentId, u64>,
}

pub struct QueryKey {
    sql_text: String,
    vector_query: Vec<f32>,
    segment_version_hash: u64,
}

pub fn get_or_execute(&mut self, query: &Query) -> Result<Vec<Row>> {
    let key = QueryKey::from(query, &self.segment_versions);
    
    if let Some(cached) = self.cache.get(&key) {
        return Ok(cached.results.clone());
    }
    
    let results = execute_query(query)?;
    self.cache.insert(key, CachedResult {
        results: results.clone(),
        cached_at: Instant::now(),
    });
    
    Ok(results)
}

pub fn invalidate_on_segment_update(&mut self, seg_id: SegmentId) {
    // Increment version for this segment
    self.segment_versions
        .entry(seg_id)
        .and_modify(|v| *v += 1)
        .or_insert(1);
    
    // Cache entries with old segment_version_hash are now stale
    // They'll be recomputed on next access
}
```

---

## Implementation Priority for HyperStreamDB

### Phase 1 (Immediate - High ROI)
1. ✅ Column statistics (zone maps) - Easy, high payoff
2. ✅ Delta compression in inverted indexes - Moderate effort
3. ✅ Vectorized distance batch processing - Medium effort, good perf gain

### Phase 2 (Medium-term)
4. Adaptive codec selection - Improves storage 20-30%
5. Segment pruning with global index - Better query planning
6. Deferred index building - Better UX (no latency spike)

### Phase 3 (Long-term)
7. Query result caching - Good for RAG workloads
8. WAL for stronger durability - Enterprise feature

---

## Doris Attribute Notice

These patterns are inspired by Apache Doris (Apache 2.0 Licensed) available at:
https://github.com/apache/doris

Specific optimizations adapted:
- Block-based vectorized execution (Doris Storage Engine)
- Column statistics and zone maps (Doris Query Optimizer)
- Delta-encoded inverted indexes (Doris Inverted Index)
- Deferred index construction (Doris Background Task)
- Cardinality-aware encoding (Doris Segment Writer)

These are **architectural patterns and algorithms**, not code copies. Implementations are written from scratch to fit HyperStreamDB's architecture.

