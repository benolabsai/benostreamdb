// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Graph traversal and neighborhood analysis primitives for BenoStreamDB tables.
//!
//! Provides first-class graph querying over tables:
//! - High-speed zero-copy CSR (Compressed Sparse Row) index path (`.graph_v2.csr.*`)
//! - General table-scan BFS fallback for unindexed edge tables and relation-filtered queries

use std::collections::{HashMap, HashSet, VecDeque};

use ahash::AHashMap;

use anyhow::{Context, Result};
use arrow::array::{Array, Int32Array, Int64Array, UInt32Array, UInt64Array};

use crate::core::cache::{CacheExt, DiskCache};
use crate::core::index::csr_graph::{MmapCsrGraph, MultiSegmentCsrGraph};
use crate::core::manifest::ManifestManager;
use crate::core::table::Table;

/// Configuration options for graph neighborhood traversal.
#[derive(Debug, Clone, Default)]
pub struct GraphNeighborhoodOptions {
    /// Initial seed node IDs to expand from.
    pub seeds: Vec<u64>,
    /// Maximum traversal depth (number of hops from seeds).
    pub hops: u32,
    /// If true, traverse edges only in the forward (source -> target) direction.
    /// If false, traverse edges symmetrically (source <-> target).
    pub directed: bool,
    /// Optional whitelist of allowed edge relation/predicate strings.
    pub allowed_relations: Option<Vec<String>>,
    /// Name of the source column with an associated CSR graph index (if any).
    pub graph_column: Option<String>,
    /// Optional max-degree limit for combinatorial explosion control (super-node truncation).
    pub max_degree: Option<usize>,
    /// Optional token budget cap: stop expanding once visited set reaches this size.
    pub max_nodes: Option<usize>,
    /// Explicit source column name (defaults to auto-detect: "source", "src", "src_id").
    pub source_column: Option<String>,
    /// Explicit target column name (defaults to auto-detect: "target", "dst", "dst_id").
    pub target_column: Option<String>,
    /// Explicit relation column name (defaults to auto-detect: "relation", "predicate", "type", "rel", "edge_type").
    pub relation_column: Option<String>,
}

impl Table {
    /// Load memory-mapped [`MultiSegmentCsrGraph`] for `graph_column` from the table's
    /// index files (`{file}.graph_v2.csr.{offsets,edges,dict}`).
    ///
    /// Returns `Ok(None)` if no CSR graph index exists for that column.
    /// Load memory-mapped [`MultiSegmentCsrGraph`] for multiple columns in a single manifest pass.
    pub async fn load_graph_indices(
        &self,
        columns: &[&str],
    ) -> Result<HashMap<String, MultiSegmentCsrGraph>> {
        let manifest = self.manifest().await?;
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let entries = manifest_manager.load_all_entries(&manifest).await?;
        self.load_graph_indices_from_entries(&entries, columns)
            .await
    }

    async fn load_graph_indices_from_entries(
        &self,
        entries: &[crate::core::manifest::ManifestEntry],
        columns: &[&str],
    ) -> Result<HashMap<String, MultiSegmentCsrGraph>> {
        let mut col_segments: HashMap<String, Vec<MmapCsrGraph>> = HashMap::new();
        let cache = DiskCache::new(self.store.clone());

        // A Puffin compound bundle registers one `IndexFile` per blob (offsets,
        // edges, dict), so the same logical CSR graph appears as several
        // entries. Deduplicate by (file_path, column) so each graph is loaded
        // exactly once — otherwise the segments would be concatenated and every
        // neighbor would be reported N times.
        let mut seen: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();

        for entry in entries {
            for idx in &entry.index_files {
                if idx.index_category == "graph_v2" {
                    let col = idx.column_name.as_str();
                    {
                        if columns.contains(&col) {
                            if !seen.insert((idx.file_path.clone(), col.to_string())) {
                                continue;
                            }
                            if idx.blob_offset.is_some() {
                                // Puffin compound bundle: read the CSR blobs into
                                // memory and build the graph from owned bytes.
                                match self.load_csr_from_puffin(idx).await {
                                    Ok(graph) => {
                                        tracing::debug!(
                                            column = %col,
                                            file = %idx.file_path,
                                            num_nodes = graph.num_nodes,
                                            num_edges = graph.num_edges,
                                            "loaded CSR graph from Puffin bundle"
                                        );
                                        col_segments
                                            .entry(col.to_string())
                                            .or_default()
                                            .push(graph);
                                    }
                                    Err(e) => {
                                        // Do NOT swallow this: a failed load makes
                                        // the caller fall back to a scan, which can
                                        // silently change query results.
                                        tracing::warn!(
                                            column = %col,
                                            file = %idx.file_path,
                                            error = %e,
                                            "failed to load CSR graph from Puffin bundle; \
                                             falling back to a relational scan"
                                        );
                                    }
                                }
                            } else {
                                let offsets_str = format!("{}.graph_v2.csr.offsets", idx.file_path);
                                let edges_str = format!("{}.graph_v2.csr.edges", idx.file_path);
                                let dict_str = format!("{}.graph_v2.csr.dict", idx.file_path);

                                if let (Ok(offsets_mmap), Ok(edges_mmap), Ok(dict_mmap)) = (
                                    cache.get_mmap(&offsets_str).await,
                                    cache.get_mmap(&edges_str).await,
                                    cache.get_mmap(&dict_str).await,
                                ) {
                                    col_segments.entry(col.to_string()).or_default().push(
                                        MmapCsrGraph::from_mmaps(
                                            offsets_mmap,
                                            edges_mmap,
                                            dict_mmap,
                                        ),
                                    );
                                } else {
                                    tracing::warn!(
                                        column = %col,
                                        file = %idx.file_path,
                                        "failed to mmap loose CSR sidecars; \
                                         falling back to a relational scan"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut res = HashMap::new();
        for (col, segs) in col_segments {
            if !segs.is_empty() {
                tracing::debug!(
                    column = %col,
                    segments = segs.len(),
                    "resolved CSR graph segments"
                );
                res.insert(col, MultiSegmentCsrGraph::new(segs));
            }
        }
        if res.is_empty() {
            tracing::debug!(
                requested = ?columns,
                "no CSR graph index found for the requested column(s)"
            );
        }
        Ok(res)
    }

    /// Read a CSR graph packed inside a Puffin compound bundle into memory.
    ///
    /// The decoded graph is cached in [`crate::core::cache::CSR_GRAPH_CACHE`]
    /// (budgeted from `BSDB_CACHE_GB`) so repeated `load_graph_index` calls do
    /// not re-read and re-copy the bundle.
    async fn load_csr_from_puffin(
        &self,
        idx: &crate::core::manifest::IndexFile,
    ) -> Result<MmapCsrGraph> {
        let cache_key = format!("{}:{}", idx.file_path, idx.blob_offset.unwrap_or(0));
        if let Some(cached) = crate::core::cache::CSR_GRAPH_CACHE
            .get_with_metrics(&cache_key, "csr_graph")
            .await
        {
            tracing::debug!(
                file = %idx.file_path,
                column = %idx.column_name,
                "CSR graph cache hit"
            );
            return Ok((*cached).clone());
        }
        tracing::debug!(
            file = %idx.file_path,
            column = %idx.column_name,
            blob_offset = ?idx.blob_offset,
            "CSR graph cache miss; reading Puffin bundle"
        );

        let bytes = self
            .store
            .get(&object_store::path::Path::from(idx.file_path.as_str()))
            .await
            .with_context(|| format!("Failed to read Puffin bundle '{}'", idx.file_path))?
            .bytes()
            .await?;
        let mut reader =
            crate::core::puffin::PuffinReader::new(std::io::Cursor::new(bytes.to_vec()))?;
        let blobs = reader.footer().blobs.clone();
        tracing::debug!(
            file = %idx.file_path,
            blob_types = ?blobs.iter().map(|b| b.r#type.as_str()).collect::<Vec<_>>(),
            "Puffin bundle footer"
        );

        // A Puffin bundle holds one CSR triple per graph column (e.g. a forward
        // `source` graph and a reverse `target` graph). Only read the blobs
        // whose filename belongs to the requested column — otherwise the last
        // triple of each type wins and the wrong adjacency is loaded (the blob
        // order is not stable, so this was intermittent).
        let marker = format!(".{}.graph_v2.csr.", idx.column_name);
        let mut offsets = None;
        let mut edges = None;
        let mut dict = None;
        for (i, blob) in blobs.iter().enumerate() {
            let filename = blob
                .properties
                .get("filename")
                .map(String::as_str)
                .unwrap_or("");
            if !filename.contains(&marker) {
                continue;
            }
            match blob.r#type.as_str() {
                crate::core::puffin::PUFFIN_BLOB_GRAPH_CSR_OFFSETS => {
                    offsets = Some(reader.read_blob(i)?)
                }
                crate::core::puffin::PUFFIN_BLOB_GRAPH_CSR_EDGES => {
                    edges = Some(reader.read_blob(i)?)
                }
                crate::core::puffin::PUFFIN_BLOB_GRAPH_CSR_DICT => {
                    dict = Some(reader.read_blob(i)?)
                }
                _ => {}
            }
        }

        let offsets = offsets.context("Puffin CSR bundle is missing the offsets blob")?;
        let edges = edges.context("Puffin CSR bundle is missing the edges blob")?;
        let dict = dict.context("Puffin CSR bundle is missing the dict blob")?;
        let graph = MmapCsrGraph::from_bytes(offsets, edges, dict);
        tracing::debug!(
            file = %idx.file_path,
            column = %idx.column_name,
            num_nodes = graph.num_nodes,
            num_edges = graph.num_edges,
            bytes = graph.size_in_bytes(),
            "decoded CSR graph from Puffin bundle"
        );
        crate::core::cache::CSR_GRAPH_CACHE
            .insert(cache_key, std::sync::Arc::new(graph.clone()))
            .await;
        Ok(graph)
    }

    /// Load memory-mapped [`MultiSegmentCsrGraph`] for `graph_column` from the table's
    /// index files (`{file}.graph_v2.csr.{offsets,edges,dict}`).
    ///
    /// Returns `Ok(None)` if no CSR graph index exists for that column.
    pub async fn load_graph_index(
        &self,
        graph_column: &str,
    ) -> Result<Option<MultiSegmentCsrGraph>> {
        let mut map = self.load_graph_indices(&[graph_column]).await?;
        Ok(map.remove(graph_column))
    }

    /// Compute the k-hop neighborhood from `options.seeds` over this table.
    ///
    /// Execution Strategy:
    /// 1. If `allowed_relations` is None and a CSR graph index exists for `graph_column`,
    ///    uses the zero-copy memory-mapped CSR fast path (`csr_bfs_visited`).
    /// 2. For undirected traversals on directed CSR graphs, the engine automatically checks for
    ///    a reverse index named `{graph_column}_rev` (e.g. `source_rev`). If not found,
    ///    it falls back to relational scan so results remain complete.
    /// 3. Otherwise, falls back to the relational table scan with optional predicate pushdown,
    ///    extracting source/target node IDs into an in-memory graph representation.
    pub async fn graph_neighborhood(&self, options: &GraphNeighborhoodOptions) -> Result<Vec<u64>> {
        if options.seeds.is_empty() {
            return Ok(Vec::new());
        }

        // Fast path: CSR index if available and no relation filtering is requested
        if options.allowed_relations.is_none() {
            if let Some(ref col) = options.graph_column {
                let rev_col = format!("{col}_rev");
                let cols_to_load: Vec<&str> = if !options.directed {
                    vec![col.as_str(), rev_col.as_str()]
                } else {
                    vec![col.as_str()]
                };

                let mut indices = self
                    .load_graph_indices(&cols_to_load)
                    .await
                    .unwrap_or_default();
                if let Some(forward) = indices.remove(col) {
                    let reverse = if !options.directed {
                        indices.remove(&rev_col)
                    } else {
                        None
                    };

                    // If undirected traversal was requested but no reverse CSR exists,
                    // fall through to relational path so traversal is complete.
                    if options.directed || reverse.is_some() {
                        return Ok(csr_bfs_visited(
                            &forward,
                            reverse.as_ref(),
                            &options.seeds,
                            options.hops,
                            options.directed,
                            options.max_degree,
                            options.max_nodes,
                        ));
                    }
                }
            }
        }

        // Relational table-scan path (supports allowed_relations filtering and unindexed tables)
        self.graph_neighborhood_scan(options).await
    }

    async fn graph_neighborhood_scan(
        &self,
        options: &GraphNeighborhoodOptions,
    ) -> Result<Vec<u64>> {
        let schema = self.arrow_schema();

        // 1. Resolve source and target columns
        let src_col_name = options.source_column.clone().unwrap_or_else(|| {
            ["source", "src", "src_id", "from", "u"]
                .iter()
                .find(|c| schema.column_with_name(c).is_some())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "source".to_string())
        });

        let tgt_col_name = options.target_column.clone().unwrap_or_else(|| {
            ["target", "dst", "dst_id", "to", "v"]
                .iter()
                .find(|c| schema.column_with_name(c).is_some())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "target".to_string())
        });

        // 2. Build relation filter if requested
        let rel_filter = if let Some(ref rels) = options.allowed_relations {
            let rel_col = options.relation_column.clone().or_else(|| {
                ["relation", "predicate", "type", "rel", "edge_type"]
                    .iter()
                    .find(|c| schema.column_with_name(c).is_some())
                    .map(|s| s.to_string())
            });

            if let Some(col) = rel_col {
                let vals: Vec<String> = rels
                    .iter()
                    .map(|r| format!("'{}'", r.replace('\'', "''")))
                    .collect();
                Some(format!("{col} IN ({})", vals.join(", ")))
            } else {
                None
            }
        } else {
            None
        };

        // 3. Scan the table with predicate pushdown
        let batches = self
            .read_async(rel_filter.as_deref(), None, None)
            .await
            .context("failed to read table batches for graph neighborhood scan")?;

        // 4. Construct adjacency list
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for batch in &batches {
            let b_schema = batch.schema();
            let src_idx = match b_schema.index_of(&src_col_name) {
                Ok(i) => i,
                Err(_) => continue,
            };
            let tgt_idx = match b_schema.index_of(&tgt_col_name) {
                Ok(i) => i,
                Err(_) => continue,
            };

            let src_col = batch.column(src_idx);
            let tgt_col = batch.column(tgt_idx);

            let sources = extract_u64_values(src_col.as_ref());
            let targets = extract_u64_values(tgt_col.as_ref());

            for i in 0..batch.num_rows() {
                if let (Some(Some(s)), Some(Some(t))) =
                    (sources.get(i).copied(), targets.get(i).copied())
                {
                    adj.entry(s).or_default().push(t);
                    if !options.directed {
                        adj.entry(t).or_default().push(s);
                    }
                }
            }
        }

        // 5. Breadth-First Search (BFS)
        let mut visited: HashSet<u64> = HashSet::new();
        let mut queue: VecDeque<(u64, u32)> = VecDeque::new();

        for &seed in &options.seeds {
            if visited.insert(seed) {
                queue.push_back((seed, 0));
            }
        }

        let budget_exhausted =
            |visited: &HashSet<u64>| options.max_nodes.is_some_and(|cap| visited.len() >= cap);

        while let Some((node, depth)) = queue.pop_front() {
            if budget_exhausted(&visited) {
                break;
            }
            if depth >= options.hops {
                continue;
            }

            if let Some(neighbors) = adj.get(&node) {
                if let Some(limit) = options.max_degree {
                    if neighbors.len() > limit {
                        continue;
                    }
                }

                for &nbr in neighbors {
                    if budget_exhausted(&visited) {
                        break;
                    }
                    if visited.insert(nbr) {
                        queue.push_back((nbr, depth + 1));
                    }
                }
            }
        }

        let mut out: Vec<u64> = visited.into_iter().collect();
        out.sort_unstable();
        Ok(out)
    }

    /// Compute the shortest path between `start_node` and `end_node`.
    ///
    /// Fast path: If a CSR graph index exists, executes BFS over the CSR graph.
    /// For undirected traversals on directed graphs, the engine automatically checks for
    /// a reverse index named `{graph_column}_rev` (e.g. `source_rev`). If `graph_column` is `None`,
    /// defaults to `"source"`.
    ///
    /// Fallback path: If unindexed, performs table scan with relation/endpoint detection.
    /// Both paths are bounded by `MAX_SHORTEST_PATH_NODES` to avoid runaway memory on super-nodes.
    pub async fn shortest_path(
        &self,
        start_node: u64,
        end_node: u64,
        directed: bool,
        graph_column: Option<&str>,
    ) -> Result<Vec<u64>> {
        if start_node == end_node {
            return Ok(vec![start_node]);
        }

        const MAX_SHORTEST_PATH_NODES: usize = 100_000;
        const MAX_BFS_DEGREE: usize = 10_000;

        use crate::core::sql::graph_udf::graph_view::GraphView;

        // Fast path: CSR index
        let col_to_check = graph_column.unwrap_or("source");
        let rev_col = format!("{col_to_check}_rev");
        let cols_to_load: Vec<&str> = if !directed {
            vec![col_to_check, rev_col.as_str()]
        } else {
            vec![col_to_check]
        };

        let mut indices = self
            .load_graph_indices(&cols_to_load)
            .await
            .unwrap_or_default();
        if let Some(forward) = indices.remove(col_to_check) {
            let reverse = if !directed {
                indices.remove(&rev_col)
            } else {
                None
            };

            if directed || reverse.is_some() {
                let mut queue = VecDeque::new();
                let mut visited = HashMap::new();

                queue.push_back(start_node);
                visited.insert(start_node, start_node);

                let mut found = false;
                while let Some(current) = queue.pop_front() {
                    if current == end_node {
                        found = true;
                        break;
                    }

                    if visited.len() >= MAX_SHORTEST_PATH_NODES {
                        tracing::warn!(
                            "shortest_path exceeded maximum node expansion limit ({} nodes); aborting search",
                            MAX_SHORTEST_PATH_NODES
                        );
                        break;
                    }

                    for neighbor in forward
                        .get_neighbors(current)
                        .into_iter()
                        .take(MAX_BFS_DEGREE)
                    {
                        if let std::collections::hash_map::Entry::Vacant(entry) =
                            visited.entry(neighbor)
                        {
                            entry.insert(current);
                            queue.push_back(neighbor);
                        }
                    }

                    if let Some(ref rev) = reverse {
                        for neighbor in rev.get_neighbors(current).into_iter().take(MAX_BFS_DEGREE)
                        {
                            if let std::collections::hash_map::Entry::Vacant(entry) =
                                visited.entry(neighbor)
                            {
                                entry.insert(current);
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }

                if found {
                    let mut path = Vec::new();
                    let mut curr = end_node;
                    while curr != start_node {
                        path.push(curr);
                        curr = visited[&curr];
                    }
                    path.push(start_node);
                    path.reverse();
                    return Ok(path);
                } else {
                    return Ok(Vec::new());
                }
            }
        }

        // Fallback path: Table scan BFS
        let batches = self.read_async(None, None, None).await?;
        let schema = self.arrow_schema();
        let src_col_name = ["source", "src", "src_id", "from", "u"]
            .iter()
            .find(|c| schema.column_with_name(c).is_some())
            .unwrap_or(&"source");
        let tgt_col_name = ["target", "dst", "dst_id", "to", "v"]
            .iter()
            .find(|c| schema.column_with_name(c).is_some())
            .unwrap_or(&"target");

        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for batch in &batches {
            let b_schema = batch.schema();
            let src_idx = match b_schema.index_of(src_col_name) {
                Ok(i) => i,
                Err(_) => continue,
            };
            let tgt_idx = match b_schema.index_of(tgt_col_name) {
                Ok(i) => i,
                Err(_) => continue,
            };

            let sources = extract_u64_values(batch.column(src_idx).as_ref());
            let targets = extract_u64_values(batch.column(tgt_idx).as_ref());

            for i in 0..batch.num_rows() {
                if let (Some(Some(s)), Some(Some(t))) =
                    (sources.get(i).copied(), targets.get(i).copied())
                {
                    adj.entry(s).or_default().push(t);
                    if !directed {
                        adj.entry(t).or_default().push(s);
                    }
                }
            }
        }

        let mut queue = VecDeque::new();
        let mut visited = HashMap::new();

        queue.push_back(start_node);
        visited.insert(start_node, start_node);

        let mut found = false;
        while let Some(current) = queue.pop_front() {
            if current == end_node {
                found = true;
                break;
            }

            if visited.len() >= MAX_SHORTEST_PATH_NODES {
                tracing::warn!(
                    "shortest_path scan fallback exceeded maximum node expansion limit ({} nodes); aborting search",
                    MAX_SHORTEST_PATH_NODES
                );
                break;
            }

            if let Some(neighbors) = adj.get(&current) {
                for &neighbor in neighbors.iter().take(MAX_BFS_DEGREE) {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        visited.entry(neighbor)
                    {
                        entry.insert(current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if found {
            let mut path = Vec::new();
            let mut curr = end_node;
            while curr != start_node {
                path.push(curr);
                curr = visited[&curr];
            }
            path.push(start_node);
            path.reverse();
            Ok(path)
        } else {
            Ok(Vec::new())
        }
    }

    /// Extract pairwise shortest connecting paths between seed nodes.
    pub async fn connecting_paths(
        &self,
        seeds: &[u64],
        directed: bool,
        graph_column: Option<&str>,
    ) -> Result<Vec<(u64, u64)>> {
        if seeds.len() < 2 {
            return Ok(Vec::new());
        }

        let mut edges = HashSet::new();

        for i in 0..seeds.len() {
            for j in (i + 1)..seeds.len() {
                let u = seeds[i];
                let v = seeds[j];

                let forward_path = self.shortest_path(u, v, directed, graph_column).await?;
                if forward_path.len() >= 2 {
                    for k in 0..forward_path.len() - 1 {
                        edges.insert((forward_path[k], forward_path[k + 1]));
                    }
                }

                if directed && forward_path.is_empty() {
                    let rev_path = self.shortest_path(v, u, directed, graph_column).await?;
                    if rev_path.len() >= 2 {
                        for k in 0..rev_path.len() - 1 {
                            edges.insert((rev_path[k], rev_path[k + 1]));
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u64, u64)> = edges.into_iter().collect();
        out.sort_unstable();
        Ok(out)
    }

    /// Find reachable neighbors from `node` within `hops` steps.
    pub async fn graph_neighbors(
        &self,
        node: u64,
        hops: u32,
        graph_column: Option<&str>,
    ) -> Result<Vec<u64>> {
        let opts = GraphNeighborhoodOptions {
            seeds: vec![node],
            hops,
            directed: true,
            graph_column: graph_column.map(String::from),
            max_nodes: Some(100_000),
            max_degree: Some(10_000),
            ..Default::default()
        };
        let mut visited = self.graph_neighborhood(&opts).await?;
        visited.retain(|&x| x != node);
        visited.sort_unstable();
        visited.dedup();
        Ok(visited)
    }

    /// Extract induced subgraph edges within `hops` steps of `seeds`.
    pub async fn subgraph_edges(
        &self,
        seeds: &[u64],
        hops: u32,
        directed: bool,
        max_degree: Option<usize>,
        max_nodes: Option<usize>,
        graph_column: Option<&str>,
    ) -> Result<Vec<(u64, u64)>> {
        use crate::core::sql::graph_udf::graph_view::GraphView;

        let opts = GraphNeighborhoodOptions {
            seeds: seeds.to_vec(),
            hops,
            directed,
            max_degree,
            max_nodes,
            graph_column: graph_column.map(String::from),
            ..Default::default()
        };
        let visited_nodes = self.graph_neighborhood(&opts).await?;
        let node_set: HashSet<u64> = visited_nodes.into_iter().collect();

        if node_set.is_empty() {
            return Ok(Vec::new());
        }

        let mut edges = HashSet::new();
        let col_to_check = graph_column.unwrap_or("source");

        if let Ok(Some(forward)) = self.load_graph_index(col_to_check).await {
            for &u in &node_set {
                for v in forward.get_neighbors(u) {
                    if node_set.contains(&v) {
                        edges.insert((u, v));
                    }
                }
            }
        } else {
            let batches = self.read_async(None, None, None).await?;
            let schema = self.arrow_schema();
            let src_col_name = ["source", "src", "src_id", "from", "u"]
                .iter()
                .find(|c| schema.column_with_name(c).is_some())
                .unwrap_or(&"source");
            let tgt_col_name = ["target", "dst", "dst_id", "to", "v"]
                .iter()
                .find(|c| schema.column_with_name(c).is_some())
                .unwrap_or(&"target");

            for batch in &batches {
                let b_schema = batch.schema();
                let src_idx = match b_schema.index_of(src_col_name) {
                    Ok(i) => i,
                    Err(_) => continue,
                };
                let tgt_idx = match b_schema.index_of(tgt_col_name) {
                    Ok(i) => i,
                    Err(_) => continue,
                };

                let sources = extract_u64_values(batch.column(src_idx).as_ref());
                let targets = extract_u64_values(batch.column(tgt_idx).as_ref());

                for i in 0..batch.num_rows() {
                    if let (Some(Some(s)), Some(Some(t))) =
                        (sources.get(i).copied(), targets.get(i).copied())
                    {
                        if node_set.contains(&s) && node_set.contains(&t) {
                            edges.insert((s, t));
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u64, u64)> = edges.into_iter().collect();
        out.sort_unstable();
        Ok(out)
    }

    /// Execute regional DRIFT search around query `seeds` over this edge table.
    ///
    /// Uses [`GraphMode::Auto`]: the in-memory graph when the regional
    /// subgraph fits the DRIFT memory budget, else the out-of-core CSR.
    pub async fn regional_drift(
        &self,
        query: &str,
        seeds: &[u64],
        top_k: usize,
        hops: u32,
        n_depth: u32,
        k_followups: usize,
    ) -> Result<crate::core::sql::graph_udf::drift_search::DriftSearchResult> {
        self.regional_drift_with_mode(
            query,
            seeds,
            top_k,
            hops,
            n_depth,
            k_followups,
            crate::core::sql::graph_udf::graph_view::GraphMode::Auto,
        )
        .await
    }

    /// Resolve a [`GraphMode`] into a concrete graph view mode.
    /// `Auto` estimates the regional subgraph and picks
    /// `InMemory` when it fits the DRIFT memory budget, else `OutOfCore`.
    pub async fn resolve_graph_mode(
        &self,
        mode: crate::core::sql::graph_udf::graph_view::GraphMode,
        seeds: &[u64],
        hops: u32,
    ) -> Result<crate::core::sql::graph_udf::graph_view::GraphMode> {
        use crate::core::sql::graph_udf::graph_view::{graph_memory_budget_bytes, GraphMode};
        match mode {
            GraphMode::Auto => {
                let est = self.estimate_regional_subgraph_bytes(seeds, hops).await?;
                if est <= graph_memory_budget_bytes() as usize {
                    Ok(GraphMode::InMemory)
                } else {
                    Ok(GraphMode::OutOfCore)
                }
            }
            m => Ok(m),
        }
    }

    /// Load the graph backing for OutOfCore or Cached modes.
    async fn load_graph_backing(
        &self,
        mode: crate::core::sql::graph_udf::graph_view::GraphMode,
    ) -> Result<Box<dyn crate::core::sql::graph_udf::graph_view::GraphView>> {
        use crate::core::sql::graph_udf::graph_view::{
            graph_memory_budget_bytes, CachingGraph, GraphMode,
        };
        match mode {
            GraphMode::InMemory => unreachable!("InMemory mode handled separately"),
            GraphMode::OutOfCore => {
                let graph = self.load_graph_index("source").await?.ok_or_else(|| {
                    anyhow::anyhow!("out-of-core graph requires a CSR graph index on 'source'")
                })?;
                tracing::debug!(
                    table = %self.uri,
                    mode = "out_of_core",
                    "resolved out-of-core graph backing from CSR index"
                );
                Ok(Box::new(graph))
            }
            GraphMode::Cached => {
                let graph = self.load_graph_index("source").await?.ok_or_else(|| {
                    anyhow::anyhow!("cached graph requires a CSR graph index on 'source'")
                })?;
                let cached = CachingGraph::new(graph, graph_memory_budget_bytes() as usize);
                tracing::debug!(
                    table = %self.uri,
                    mode = "cached",
                    "resolved cached graph backing from CSR index"
                );
                Ok(Box::new(cached))
            }
            GraphMode::Auto => unreachable!("Auto resolved above"),
        }
    }

    /// Construct a [`GraphView`] according to the given mode, seeds, and hops.
    pub async fn graph_view(
        &self,
        mode: crate::core::sql::graph_udf::graph_view::GraphMode,
        seeds: &[u64],
        hops: u32,
    ) -> Result<Box<dyn crate::core::sql::graph_udf::graph_view::GraphView>> {
        use crate::core::sql::graph_udf::graph_view::{GraphMode, SimpleGraph, SubgraphView};

        let mode = self.resolve_graph_mode(mode, seeds, hops).await?;

        match mode {
            GraphMode::InMemory => {
                let edges = self
                    .subgraph_edges(seeds, hops, false, Some(1000), Some(50000), None)
                    .await?;
                Ok(Box::new(SimpleGraph::from_directed_edges(&edges)))
            }
            _ => {
                let backing = self.load_graph_backing(mode).await?;
                let region = self.region_nodes(seeds, hops).await?;
                Ok(Box::new(SubgraphView::new(backing, region)))
            }
        }
    }

    /// The set of nodes within `hops` of `seeds` — the regional subgraph every
    /// graph mode operates on.
    async fn region_nodes(&self, seeds: &[u64], hops: u32) -> Result<HashSet<u64>> {
        Ok(self
            .graph_neighborhood(&GraphNeighborhoodOptions {
                seeds: seeds.to_vec(),
                hops,
                directed: false,
                max_degree: Some(1000),
                max_nodes: Some(50000),
                ..Default::default()
            })
            .await?
            .into_iter()
            .collect())
    }

    /// Order-of-magnitude estimate of the in-memory regional subgraph size.
    async fn estimate_regional_subgraph_bytes(&self, seeds: &[u64], hops: u32) -> Result<usize> {
        use crate::core::sql::graph_udf::drift_search::estimate_subgraph_bytes;

        let nodes = self.region_nodes(seeds, hops).await?.len();
        // A modest average degree is enough to choose a mode.
        Ok(estimate_subgraph_bytes(nodes, 16))
    }

    /// Execute regional DRIFT search with an explicit graph mode.
    ///
    /// - `InMemory` materializes the regional subgraph (`SimpleGraph`).
    /// - `OutOfCore` walks the mmap CSR directly (no materialization).
    /// - `Cached` walks the CSR with a bounded in-memory neighbor cache.
    /// - `Auto` picks `InMemory` when the estimated subgraph fits the DRIFT
    ///   memory budget, else `OutOfCore`.
    pub async fn regional_drift_with_mode(
        &self,
        query: &str,
        seeds: &[u64],
        top_k: usize,
        hops: u32,
        n_depth: u32,
        k_followups: usize,
        mode: crate::core::sql::graph_udf::graph_view::GraphMode,
    ) -> Result<crate::core::sql::graph_udf::drift_search::DriftSearchResult> {
        use crate::core::sql::graph_udf::drift_search::{
            execute_drift_search, DriftSearchParams, HeuristicFollowUpGenerator,
        };

        // `graph_view` already restricts every mode to the regional node set, so
        // the in-memory and out-of-core paths traverse the identical induced
        // subgraph and return the same result.
        let graph = self.graph_view(mode, seeds, hops).await?;
        let visited = self
            .graph_neighborhood(&GraphNeighborhoodOptions {
                seeds: seeds.to_vec(),
                hops,
                directed: false,
                max_degree: Some(1000),
                max_nodes: Some(50000),
                ..Default::default()
            })
            .await?;
        let (community_map, top_communities) =
            drift_communities_and_top(graph.as_ref(), seeds, &visited, top_k);

        let generator = HeuristicFollowUpGenerator {
            graph: graph.as_ref(),
            community_map: &community_map,
        };
        let params = DriftSearchParams {
            n_depth,
            k_followups,
            top_k,
            hops,
            alpha: 0.85,
            confidence_threshold: 0.0,
            ..Default::default()
        };
        let mut result =
            execute_drift_search(query, graph.as_ref(), &top_communities, &generator, &params);
        result.community_assignments = community_map;
        Ok(result)
    }
}

/// Connected-component communities over `visited`, scored by seed presence and
/// degree, returning `(community_map, top_communities)`.
fn drift_communities_and_top(
    graph: &dyn crate::core::sql::graph_udf::graph_view::GraphView,
    seeds: &[u64],
    visited: &[u64],
    top_k: usize,
) -> (AHashMap<u64, u64>, Vec<u64>) {
    // Integer-keyed label maps: ahash plus a single reused neighbor scratch
    // buffer keeps this 15-round label-propagation pass allocation-light. The
    // return type is `AHashMap` to match `DriftSearchResult::community_assignments`
    // and `HeuristicFollowUpGenerator::community_map`.
    let mut community_map: AHashMap<u64, u64> = visited.iter().map(|&n| (n, n)).collect();
    let mut neighbors: Vec<u64> = Vec::new();
    for _ in 0..15 {
        let mut changed = false;
        for &node in visited {
            let mut label_counts: AHashMap<u64, usize> = AHashMap::new();
            neighbors.clear();
            graph.get_neighbors_into(node, &mut neighbors);
            for &neighbor in &neighbors {
                if let Some(&lbl) = community_map.get(&neighbor) {
                    *label_counts.entry(lbl).or_default() += 1;
                }
            }
            if label_counts.is_empty() {
                continue;
            }
            let best = label_counts
                .into_iter()
                .max_by_key(|&(lbl, count)| (count, std::cmp::Reverse(lbl)))
                .map(|(lbl, _)| lbl)
                .unwrap_or(node);
            if community_map.get(&node) != Some(&best) {
                community_map.insert(node, best);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut comm_scores: AHashMap<u64, usize> = AHashMap::new();
    for &s in seeds {
        if let Some(&c) = community_map.get(&s) {
            *comm_scores.entry(c).or_default() += 10;
        }
    }
    for (&node, &c) in &community_map {
        *comm_scores.entry(c).or_default() += graph.get_degree(node);
    }
    let mut sorted: Vec<(u64, usize)> = comm_scores.into_iter().collect();
    sorted.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
    let top: Vec<u64> = sorted.into_iter().take(top_k).map(|(c, _)| c).collect();
    (community_map, top)
}

/// Multi-hop BFS over a CSR graph, returning the sorted visited node set (seeds included).
pub fn csr_bfs_visited(
    forward: &MultiSegmentCsrGraph,
    reverse: Option<&MultiSegmentCsrGraph>,
    seeds: &[u64],
    hops: u32,
    directed: bool,
    max_degree: Option<usize>,
    max_nodes: Option<usize>,
) -> Vec<u64> {
    use crate::core::sql::graph_udf::graph_view::GraphView;

    let mut visited: HashSet<u64> = seeds.iter().copied().collect();
    let mut frontier: Vec<u64> = visited.iter().copied().collect();

    let budget_exhausted = |visited: &HashSet<u64>| max_nodes.is_some_and(|l| visited.len() >= l);

    for _ in 0..hops {
        let mut next: Vec<u64> = Vec::new();
        for &node in &frontier {
            if budget_exhausted(&visited) {
                break;
            }
            if let Some(limit) = max_degree {
                let mut degree = forward.get_degree(node);
                if !directed {
                    if let Some(rev) = reverse {
                        degree += rev.get_degree(node);
                    }
                }
                if degree > limit {
                    continue;
                }
            }
            for n in forward.get_neighbors(node) {
                if budget_exhausted(&visited) {
                    break;
                }
                if visited.insert(n) {
                    next.push(n);
                }
            }
            if !directed {
                if let Some(rev) = reverse {
                    for n in rev.get_neighbors(node) {
                        if budget_exhausted(&visited) {
                            break;
                        }
                        if visited.insert(n) {
                            next.push(n);
                        }
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    let mut out: Vec<u64> = visited.into_iter().collect();
    out.sort_unstable();
    out
}

/// Extract numeric IDs into `Option<u64>` (supporting UInt64, Int64, UInt32, Int32).
fn extract_u64_values(col: &dyn Array) -> Vec<Option<u64>> {
    let n = col.len();
    let mut out = Vec::with_capacity(n);

    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i))
            });
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i) as u64)
            });
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i) as u64)
            });
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i) as u64)
            });
        }
    } else {
        out.resize(n, None);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_core_graph_neighborhood_traversal() -> Result<()> {
        let dir = tempdir()?;
        let uri = format!("file://{}", dir.path().display());
        let table = Table::new_async(uri).await?;

        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::UInt64, false),
            Field::new("target", DataType::UInt64, false),
            Field::new("relation", DataType::Utf8, false),
        ]));

        // Graph edges:
        // 1 -> 2 (rel1)
        // 2 -> 3 (rel2)
        // 4 -> 5 (rel1)
        // 1 -> 4 (rel2)
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 4, 1])),
                Arc::new(UInt64Array::from(vec![2, 3, 5, 4])),
                Arc::new(StringArray::from(vec!["rel1", "rel2", "rel1", "rel2"])),
            ],
        )?;

        table.write_async(vec![batch]).await?;
        table.commit_async().await?;

        // 1. hops = 1, directed = true from seed 1 -> {1, 2, 4}
        let opts = GraphNeighborhoodOptions {
            seeds: vec![1],
            hops: 1,
            directed: true,
            ..Default::default()
        };
        let res = table.graph_neighborhood(&opts).await?;
        assert_eq!(res, vec![1, 2, 4]);

        // 2. hops = 1, directed = true, allowed_relations = ["rel1"] from seed 1 -> {1, 2}
        let opts_filtered = GraphNeighborhoodOptions {
            seeds: vec![1],
            hops: 1,
            directed: true,
            allowed_relations: Some(vec!["rel1".to_string()]),
            ..Default::default()
        };
        let res_filtered = table.graph_neighborhood(&opts_filtered).await?;
        assert_eq!(res_filtered, vec![1, 2]);

        // 3. hops = 2, directed = true from seed 1 -> {1, 2, 3, 4, 5}
        let opts_2hop = GraphNeighborhoodOptions {
            seeds: vec![1],
            hops: 2,
            directed: true,
            ..Default::default()
        };
        let res_2hop = table.graph_neighborhood(&opts_2hop).await?;
        assert_eq!(res_2hop, vec![1, 2, 3, 4, 5]);

        // 4. Undirected 1-hop from node 3 -> {2, 3} (2 -> 3 in edges)
        let opts_undirected = GraphNeighborhoodOptions {
            seeds: vec![3],
            hops: 1,
            directed: false,
            ..Default::default()
        };
        let res_undirected = table.graph_neighborhood(&opts_undirected).await?;
        assert_eq!(res_undirected, vec![2, 3]);

        Ok(())
    }
}
