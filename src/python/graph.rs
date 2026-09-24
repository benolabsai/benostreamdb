// Copyright (c) 2026 Richard Albright. All rights reserved.
#![allow(deprecated)]

use crate::core::sql::session::BenoStreamSession;
use crate::python::helpers::{arrow_batches_to_pyarrow, TOKIO_RUNTIME};
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::prelude::*;
use pyo3::prelude::*;
use std::sync::Arc;
use tempfile::tempdir;

#[pyclass(name = "GraphAPI")]
pub struct PyGraphAPI {
    pub(crate) table: crate::core::table::Table,
}

impl PyGraphAPI {
    fn load_multi_graph(
        &self,
        graph_column: &str,
    ) -> PyResult<crate::core::index::csr_graph::MultiSegmentCsrGraph> {
        let entries = crate::python::helpers::TOKIO_RUNTIME
            .block_on(async {
                let manifest = self
                    .table
                    .manifest()
                    .await
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let manifest_manager = crate::core::manifest::ManifestManager::new(
                    self.table.store.clone(),
                    "",
                    &self.table.uri,
                );
                manifest_manager
                    .load_all_entries(&manifest)
                    .await
                    .map_err(|e| anyhow::anyhow!(e.to_string()))
            })
            .map_err(|e: anyhow::Error| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let segments = crate::python::helpers::TOKIO_RUNTIME.block_on(async {
            let mut segments = Vec::new();
            let cache = crate::core::cache::DiskCache::new(self.table.store.clone());

            for entry in &entries {
                for idx in &entry.index_files {
                    if idx.index_type == "graph" && idx.column_name.as_deref() == Some(graph_column)
                    {
                        let offsets_str = format!("{}.graph.csr.offsets", idx.file_path);
                        let edges_str = format!("{}.graph.csr.edges", idx.file_path);
                        let dict_str = format!("{}.graph.csr.dict", idx.file_path);

                        if let (Ok(offsets_mmap), Ok(edges_mmap), Ok(dict_mmap)) = (
                            cache.get_mmap(&offsets_str).await,
                            cache.get_mmap(&edges_str).await,
                            cache.get_mmap(&dict_str).await,
                        ) {
                            let mmap_graph =
                                crate::core::index::csr_graph::MmapCsrGraph::from_mmaps(
                                    offsets_mmap,
                                    edges_mmap,
                                    dict_mmap,
                                );
                            segments.push(mmap_graph);
                        }
                    }
                }
            }
            Ok::<_, pyo3::PyErr>(segments)
        })?;

        if segments.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "No CSR graph index found for column {}",
                graph_column
            )));
        }

        Ok(crate::core::index::csr_graph::MultiSegmentCsrGraph::new(
            segments,
        ))
    }
}

#[pymethods]
impl PyGraphAPI {
    #[new]
    pub fn py_new(table: pyo3::PyRef<'_, crate::python::table::PyTable>) -> Self {
        Self {
            table: table.table.clone(),
        }
    }

    #[pyo3(signature = (source_col, target_col, damping=0.85, iterations=20))]
    pub fn pagerank(
        &self,
        py: Python<'_>,
        source_col: &str,
        target_col: &str,
        damping: f64,
        iterations: u32,
    ) -> PyResult<Py<PyAny>> {
        let table_clone = self.table.clone();
        let src = source_col.to_string();
        let dst = target_col.to_string();

        #[allow(deprecated)]
        let (batches, schema) = py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(async {
                let session = BenoStreamSession::new(None);
                session.register_table("edges", Arc::new(table_clone))?;
                let ctx = session.get_ctx();

                // Create a temporary directory for out-of-core intermediate files
                let temp_dir = tempdir()?;
                let base_path = temp_dir.path().to_string_lossy().to_string();

                // 1. Calculate Out Degree
                let out_deg_path = format!("{}/out_degree", base_path);
                let out_deg_df = ctx.sql(&format!(
                    "SELECT {} as node, count(*)::DOUBLE as deg FROM edges GROUP BY {}",
                    src, src
                )).await?;
                out_deg_df.write_parquet(&out_deg_path, DataFrameWriteOptions::new(), None).await?;
                ctx.register_parquet("out_degree", &out_deg_path, ParquetReadOptions::default()).await?;

                // 2. Initialize PR (1.0 / N isn't strictly necessary if we just start at 1.0, but let's use 1.0)
                let pr_init_path = format!("{}/pr_0", base_path);
                let init_df = ctx.sql(&format!(
                    "SELECT node, 1.0::DOUBLE as pr FROM (SELECT {} as node FROM edges UNION SELECT {} as node FROM edges)",
                    src, dst
                )).await?;
                init_df.write_parquet(&pr_init_path, DataFrameWriteOptions::new(), None).await?;

                let mut current_pr_table = "pr_0".to_string();
                ctx.register_parquet(&current_pr_table, &pr_init_path, ParquetReadOptions::default()).await?;

                // 3. Iterative Loop
                for i in 1..=iterations {
                    let next_pr_table = format!("pr_{}", i);
                    let next_pr_path = format!("{}/{}", base_path, next_pr_table);

                    let query = format!("
                        SELECT
                            nodes.node,
                            (1.0 - {damping}) + {damping} * COALESCE(sum(in_nodes.pr / out_degree.deg), 0.0) as pr
                        FROM {current_pr_table} as nodes
                        LEFT JOIN edges ON nodes.node = edges.{dst}
                        LEFT JOIN {current_pr_table} in_nodes ON edges.{src} = in_nodes.node
                        LEFT JOIN out_degree ON edges.{src} = out_degree.node
                        GROUP BY nodes.node
                    ");

                    let df = ctx.sql(&query).await?;
                    df.write_parquet(&next_pr_path, DataFrameWriteOptions::new(), None).await?;

                    ctx.register_parquet(&next_pr_table, &next_pr_path, ParquetReadOptions::default()).await?;

                    // Deregister old table to save memory
                    ctx.deregister_table(&current_pr_table)?;
                    current_pr_table = next_pr_table;
                }

                // 4. Collect final results
                let final_df = ctx.sql(&format!("SELECT node, pr FROM {} ORDER BY pr DESC", current_pr_table)).await?;
                let schema = final_df.schema().inner().clone();
                let batches = final_df.collect().await?;

                Ok::<_, anyhow::Error>((batches, schema))
            })
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        arrow_batches_to_pyarrow(py, batches, schema)
    }

    #[pyo3(signature = (graph_column, start_node, end_node))]
    pub fn shortest_path(
        &self,
        py: Python<'_>,
        graph_column: &str,
        start_node: u64,
        end_node: u64,
    ) -> PyResult<Vec<u64>> {
        use crate::core::sql::graph_udf::drift_search::DriftGraph;
        use std::collections::{HashMap, VecDeque};
        let graph = self.load_multi_graph(graph_column)?;

        py.allow_threads(|| {
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

                for neighbor in graph.get_neighbors(current) {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        visited.entry(neighbor)
                    {
                        entry.insert(current);
                        queue.push_back(neighbor);
                    }
                }
            }

            if !found {
                return Ok(vec![]);
            }

            let mut path = Vec::new();
            let mut curr = end_node;
            while curr != start_node {
                path.push(curr);
                match visited.get(&curr) {
                    Some(next) => curr = *next,
                    None => {
                        // Unreachable: `visited` was populated by the BFS above.
                        tracing::warn!("graph: broken predecessor chain; returning partial path");
                        break;
                    }
                }
            }
            path.push(start_node);
            path.reverse();

            Ok(path)
        })
    }

    #[pyo3(signature = (graph_column, node))]
    pub fn neighbors(&self, py: Python<'_>, graph_column: &str, node: u64) -> PyResult<Vec<u64>> {
        use crate::core::sql::graph_udf::drift_search::DriftGraph;
        let graph = self.load_multi_graph(graph_column)?;
        py.allow_threads(|| Ok(graph.get_neighbors(node)))
    }

    #[pyo3(signature = (graph_column, nodes))]
    pub fn subgraph(
        &self,
        py: Python<'_>,
        graph_column: &str,
        nodes: Vec<u64>,
    ) -> PyResult<Vec<(u64, u64)>> {
        use crate::core::sql::graph_udf::drift_search::DriftGraph;
        let graph = self.load_multi_graph(graph_column)?;
        py.allow_threads(|| {
            let mut edges = Vec::new();
            use std::collections::HashSet;
            let node_set: HashSet<u64> = nodes.into_iter().collect();
            for &node in &node_set {
                for neighbor in graph.get_neighbors(node) {
                    if node_set.contains(&neighbor) {
                        edges.push((node, neighbor));
                    }
                }
            }
            Ok(edges)
        })
    }

    #[pyo3(signature = (graph_column, seeds))]
    pub fn connecting_paths(
        &self,
        py: Python<'_>,
        graph_column: &str,
        seeds: Vec<u64>,
    ) -> PyResult<Vec<(u64, u64)>> {
        use crate::core::sql::graph_udf::drift_search::DriftGraph;
        use std::collections::{HashMap, HashSet, VecDeque};
        let graph = self.load_multi_graph(graph_column)?;

        py.allow_threads(|| {
            let mut unique_seeds = seeds.clone();
            unique_seeds.sort_unstable();
            unique_seeds.dedup();

            let mut out_edges = HashSet::new();

            for i in 0..unique_seeds.len() {
                for j in (i + 1)..unique_seeds.len() {
                    let start = unique_seeds[i];
                    let goal = unique_seeds[j];

                    let mut queue = VecDeque::new();
                    let mut visited = HashSet::new();
                    let mut parent_map = HashMap::new();

                    queue.push_back(start);
                    visited.insert(start);

                    let mut found = false;
                    while let Some(curr) = queue.pop_front() {
                        if curr == goal {
                            found = true;
                            break;
                        }

                        for neighbor in graph.get_neighbors(curr) {
                            if visited.insert(neighbor) {
                                parent_map.insert(neighbor, curr);
                                queue.push_back(neighbor);
                            }
                        }
                    }

                    if found {
                        let mut curr = goal;
                        while let Some(&prev) = parent_map.get(&curr) {
                            out_edges.insert((prev, curr));
                            curr = prev;
                            if curr == start {
                                break;
                            }
                        }
                    }
                }
            }

            Ok(out_edges.into_iter().collect())
        })
    }
}
