// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! DataFusion **table functions** for graph traversal, so a graph walk can be a
//! `FROM` source rather than an aggregate:
//!
//! ```sql
//! SELECT * FROM graph_neighbors('edges', '101', 2, 'auto');
//! SELECT * FROM graph_shortest_path('edges', 101, 205, 'auto');
//! SELECT * FROM graph_all_shortest_paths('edges', 101, 205, 'auto');
//! SELECT * FROM graph_subgraph('edges', '101,102', 2, 'auto');
//! SELECT * FROM graph_connecting_paths('edges', '101,205,309', 'auto');
//! ```
//!
//! Every function follows the same **in-memory / out-of-core** pattern as the
//! graph UDAFs: the `mode` argument (`auto` | `in_memory` | `out_of_core` |
//! `cached`) is resolved through [`Table::graph_view`], which materializes a
//! `SimpleGraph` for `in_memory`/`auto`-that-fits and walks the mmap CSR for
//! `out_of_core`/`cached`. The traversal itself runs over the resulting
//! [`GraphView`], so results are mode-invariant.
//!
//! Arguments are positional (DataFusion 52 drops argument names for table
//! functions, so `seeds => '101'` is accepted but the name is ignored):
//!   * `table`  — a string literal naming a registered table (resolved in the
//!     session's default catalog/schema, or `catalog.schema.table`).
//!   * `seeds`  — a string literal of comma-separated `u64` node ids.
//!   * `hops`   — an integer literal (default `1`).
//!   * `mode`   — a string literal (default `'auto'`).
//!   * `source`, `target` — optional trailing string literals naming the edge
//!     endpoint columns. When omitted they are auto-detected from the standard
//!     candidates (`source`/`src`/`src_id`/`from`/`u` and
//!     `target`/`dst`/`dst_id`/`to`/`v`), matching the engine's edge-table
//!     convention and networkx's `source`/`target` defaults. `source` also
//!     selects the CSR index used by the out-of-core / cached modes.

use std::any::Any;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use arrow::array::{ArrayRef, ListBuilder, UInt32Builder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::common::plan_err;
use datafusion::datasource::TableType;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionState;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;

use crate::core::sql::graph_udf::graph_view::{parse_graph_mode, GraphMode, GraphView};
use crate::core::sql::BenoStreamTableProvider;
use crate::core::table::Table;

/// Hop cap for the path-finding functions, which need the regional subgraph to
/// span the whole reachable component rather than a user-supplied depth. The
/// underlying BFS is still bounded by `max_nodes` in `Table::graph_view`.
const PATH_HOPS: u32 = 1_000;

/// The graph traversal table functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphTraversalKind {
    /// Nodes reachable within `hops` of the seeds, with hop distance + origin.
    Neighbors,
    /// The shortest path (as `(node, hop)` rows) between two nodes.
    ShortestPath,
    /// Every shortest path (as a `List<UInt64>` per path) between two nodes.
    AllShortestPaths,
    /// The induced subgraph edges within `hops` of the seeds.
    Subgraph,
    /// The union of pairwise shortest paths between the seeds, as edges.
    ConnectingPaths,
}

impl GraphTraversalKind {
    /// The SQL name of the function.
    pub fn name(self) -> &'static str {
        match self {
            GraphTraversalKind::Neighbors => "graph_neighbors",
            GraphTraversalKind::ShortestPath => "graph_shortest_path",
            GraphTraversalKind::AllShortestPaths => "graph_all_shortest_paths",
            GraphTraversalKind::Subgraph => "graph_subgraph",
            GraphTraversalKind::ConnectingPaths => "graph_connecting_paths",
        }
    }

    /// The result schema of the function.
    pub fn schema(self) -> SchemaRef {
        let fields = match self {
            GraphTraversalKind::Neighbors => vec![
                Field::new("node", DataType::UInt64, false),
                Field::new("hop", DataType::UInt32, false),
                Field::new("seed", DataType::UInt64, false),
            ],
            GraphTraversalKind::ShortestPath => vec![
                Field::new("node", DataType::UInt64, false),
                Field::new("hop", DataType::UInt32, false),
            ],
            GraphTraversalKind::AllShortestPaths => vec![Field::new(
                "path",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                false,
            )],
            GraphTraversalKind::Subgraph | GraphTraversalKind::ConnectingPaths => vec![
                Field::new("source", DataType::UInt64, false),
                Field::new("target", DataType::UInt64, false),
            ],
        };
        Arc::new(Schema::new(fields))
    }
}

/// All graph traversal table functions, for registration.
pub fn all_graph_table_functions() -> Vec<(String, Arc<dyn TableFunctionImpl>)> {
    [
        GraphTraversalKind::Neighbors,
        GraphTraversalKind::ShortestPath,
        GraphTraversalKind::AllShortestPaths,
        GraphTraversalKind::Subgraph,
        GraphTraversalKind::ConnectingPaths,
    ]
    .into_iter()
    .map(|kind| {
        (
            kind.name().to_string(),
            Arc::new(GraphTraversalTableFunc { kind }) as Arc<dyn TableFunctionImpl>,
        )
    })
    .collect()
}

/// The names of every graph traversal table function.
pub fn graph_table_function_names() -> Vec<String> {
    all_graph_table_functions()
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

/// Register every graph traversal table function on a DataFusion context.
pub fn register_graph_table_functions(ctx: &mut datafusion::prelude::SessionContext) {
    for (name, func) in all_graph_table_functions() {
        ctx.register_udtf(&name, func);
    }
}

/// Parsed, validated arguments for a traversal.
#[derive(Debug, Clone)]
struct GraphTraversalParams {
    table: String,
    seeds: Vec<u64>,
    hops: u32,
    mode: GraphMode,
    source: Option<u64>,
    target: Option<u64>,
    /// Explicit source/target column names (auto-detected when `None`).
    source_col: Option<String>,
    target_col: Option<String>,
}

impl GraphTraversalParams {
    fn parse(kind: GraphTraversalKind, args: &[Expr]) -> Result<Self> {
        let table = args
            .first()
            .ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "{} requires a table name as its first argument",
                    kind.name()
                ))
            })
            .and_then(as_string)?;

        let mut params = GraphTraversalParams {
            table,
            seeds: Vec::new(),
            hops: 1,
            mode: GraphMode::Auto,
            source: None,
            target: None,
            source_col: None,
            target_col: None,
        };

        // Index of the first optional trailing argument (`source_col`, `target_col`).
        let trailing_start = match kind {
            GraphTraversalKind::Neighbors | GraphTraversalKind::Subgraph => {
                params.seeds = args
                    .get(1)
                    .map(as_seed_list)
                    .transpose()?
                    .unwrap_or_default();
                if let Some(h) = args.get(2) {
                    params.hops = as_u64(h)? as u32;
                }
                if let Some(m) = args.get(3) {
                    params.mode = parse_graph_mode(&as_string(m)?);
                }
                4
            }
            GraphTraversalKind::ShortestPath | GraphTraversalKind::AllShortestPaths => {
                params.source = Some(as_u64(args.get(1).ok_or_else(|| {
                    DataFusionError::Plan(format!("{} requires a source node", kind.name()))
                })?)?);
                params.target = Some(as_u64(args.get(2).ok_or_else(|| {
                    DataFusionError::Plan(format!("{} requires a target node", kind.name()))
                })?)?);
                if let Some(m) = args.get(3) {
                    params.mode = parse_graph_mode(&as_string(m)?);
                }
                4
            }
            GraphTraversalKind::ConnectingPaths => {
                params.seeds = args
                    .get(1)
                    .map(as_seed_list)
                    .transpose()?
                    .unwrap_or_default();
                if let Some(m) = args.get(2) {
                    params.mode = parse_graph_mode(&as_string(m)?);
                }
                3
            }
        };

        if let Some(s) = args.get(trailing_start) {
            params.source_col = Some(as_string(s)?);
        }
        if let Some(t) = args.get(trailing_start + 1) {
            params.target_col = Some(as_string(t)?);
        }

        Ok(params)
    }

    /// The seeds and hop budget used to build the regional graph view.
    fn view_seeds_hops(&self, kind: GraphTraversalKind) -> (Vec<u64>, u32) {
        match kind {
            GraphTraversalKind::ShortestPath | GraphTraversalKind::AllShortestPaths => {
                (self.source.into_iter().collect(), PATH_HOPS)
            }
            GraphTraversalKind::ConnectingPaths => (self.seeds.clone(), PATH_HOPS),
            _ => (self.seeds.clone(), self.hops),
        }
    }
}

/// A `TableFunctionImpl` for one traversal kind.
#[derive(Debug)]
struct GraphTraversalTableFunc {
    kind: GraphTraversalKind,
}

impl TableFunctionImpl for GraphTraversalTableFunc {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        let params = GraphTraversalParams::parse(self.kind, args)?;
        Ok(Arc::new(GraphTraversalTableProvider {
            kind: self.kind,
            params,
        }))
    }
}

/// The provider returned by a traversal table function. It resolves the named
/// table and runs the traversal lazily in `scan`, where the `SessionState` (and
/// therefore the catalog) is available.
#[derive(Debug)]
struct GraphTraversalTableProvider {
    kind: GraphTraversalKind,
    params: GraphTraversalParams,
}

#[async_trait]
impl TableProvider for GraphTraversalTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.kind.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let session_state = state
            .as_any()
            .downcast_ref::<SessionState>()
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "graph traversal table functions require a DataFusion SessionState".to_string(),
                )
            })?;

        let table = resolve_table(session_state, &self.params.table).await?;

        // Build the regional graph view under the requested mode (in-memory /
        // out-of-core / cached / auto), exactly like the graph UDAFs.
        let (seeds, hops) = self.params.view_seeds_hops(self.kind);
        let view = table
            .graph_view_with_columns(
                self.params.mode,
                &seeds,
                hops,
                self.params.source_col.as_deref(),
                self.params.target_col.as_deref(),
            )
            .await
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        let batch = self.kind.execute(view.as_ref(), &self.params)?;

        // Delegate the physical plan to a MemTable so projection/limit handling
        // is DataFusion's, not ours.
        let mem = datafusion::datasource::memory::MemTable::try_new(
            self.kind.schema(),
            vec![vec![batch]],
        )?;
        mem.scan(state, projection, filters, limit).await
    }
}

impl GraphTraversalKind {
    /// Run the traversal over the resolved graph view and build the result batch.
    fn execute(
        self,
        view: &dyn GraphView,
        params: &GraphTraversalParams,
    ) -> Result<arrow::record_batch::RecordBatch> {
        let schema = self.schema();
        let columns: Vec<ArrayRef> = match self {
            GraphTraversalKind::Neighbors => {
                let rows = bfs_dist_seed(view, &params.seeds, params.hops);
                let mut node = UInt64Builder::with_capacity(rows.len());
                let mut hop = UInt32Builder::with_capacity(rows.len());
                let mut seed = UInt64Builder::with_capacity(rows.len());
                for (n, d, s) in rows {
                    node.append_value(n);
                    hop.append_value(d);
                    seed.append_value(s);
                }
                vec![
                    Arc::new(node.finish()),
                    Arc::new(hop.finish()),
                    Arc::new(seed.finish()),
                ]
            }
            GraphTraversalKind::ShortestPath => {
                let path = bfs_path(view, params.source.unwrap_or(0), params.target.unwrap_or(0));
                let mut node = UInt64Builder::with_capacity(path.len());
                let mut hop = UInt32Builder::with_capacity(path.len());
                for (i, n) in path.iter().enumerate() {
                    node.append_value(*n);
                    hop.append_value(i as u32);
                }
                vec![Arc::new(node.finish()), Arc::new(hop.finish())]
            }
            GraphTraversalKind::AllShortestPaths => {
                let paths = all_shortest_paths(
                    view,
                    params.source.unwrap_or(0),
                    params.target.unwrap_or(0),
                );
                let mut builder = ListBuilder::new(UInt64Builder::new());
                for path in &paths {
                    builder.values().append_slice(path);
                    builder.append(true);
                }
                vec![Arc::new(builder.finish())]
            }
            GraphTraversalKind::Subgraph => {
                let edges = subgraph_edges(view, &params.seeds, params.hops);
                edge_columns(&edges)
            }
            GraphTraversalKind::ConnectingPaths => {
                let edges = connecting_paths(view, &params.seeds);
                edge_columns(&edges)
            }
        };
        arrow::record_batch::RecordBatch::try_new(schema, columns)
            .map_err(|e| DataFusionError::Execution(e.to_string()))
    }
}

fn edge_columns(edges: &[(u64, u64)]) -> Vec<ArrayRef> {
    let mut source = UInt64Builder::with_capacity(edges.len());
    let mut target = UInt64Builder::with_capacity(edges.len());
    for (u, v) in edges {
        source.append_value(*u);
        target.append_value(*v);
    }
    vec![Arc::new(source.finish()), Arc::new(target.finish())]
}

/// BFS from every seed, returning `(node, hop, origin_seed)` for each reachable
/// node (seeds included at hop 0), sorted.
fn bfs_dist_seed(view: &dyn GraphView, seeds: &[u64], hops: u32) -> Vec<(u64, u32, u64)> {
    let mut best: HashMap<u64, (u32, u64)> = HashMap::new();
    let mut queue: VecDeque<(u64, u32, u64)> = VecDeque::new();
    for &s in seeds {
        if best.insert(s, (0, s)).is_none() {
            queue.push_back((s, 0, s));
        }
    }
    let mut scratch: Vec<u64> = Vec::new();
    while let Some((node, dist, seed)) = queue.pop_front() {
        if dist >= hops {
            continue;
        }
        scratch.clear();
        view.get_neighbors_into(node, &mut scratch);
        for &n in &scratch {
            if let std::collections::hash_map::Entry::Vacant(e) = best.entry(n) {
                e.insert((dist + 1, seed));
                queue.push_back((n, dist + 1, seed));
            }
        }
    }
    let mut out: Vec<(u64, u32, u64)> = best
        .into_iter()
        .map(|(node, (dist, seed))| (node, dist, seed))
        .collect();
    out.sort_unstable();
    out
}

/// BFS shortest path from `source` to `target` (inclusive). Empty if unreachable.
fn bfs_path(view: &dyn GraphView, source: u64, target: u64) -> Vec<u64> {
    if source == target {
        return vec![source];
    }
    let mut prev: HashMap<u64, u64> = HashMap::new();
    let mut queue: VecDeque<u64> = VecDeque::new();
    prev.insert(source, source);
    queue.push_back(source);
    let mut scratch: Vec<u64> = Vec::new();
    while let Some(node) = queue.pop_front() {
        if node == target {
            break;
        }
        scratch.clear();
        view.get_neighbors_into(node, &mut scratch);
        for &n in &scratch {
            if let std::collections::hash_map::Entry::Vacant(e) = prev.entry(n) {
                e.insert(node);
                queue.push_back(n);
            }
        }
    }
    if !prev.contains_key(&target) {
        return Vec::new();
    }
    let mut path = vec![target];
    let mut curr = target;
    while curr != source {
        curr = prev[&curr];
        path.push(curr);
    }
    path.reverse();
    path
}

/// Every shortest path from `source` to `target`, sorted and deduplicated.
fn all_shortest_paths(view: &dyn GraphView, source: u64, target: u64) -> Vec<Vec<u64>> {
    if source == target {
        return vec![vec![source]];
    }
    let mut dist: HashMap<u64, u32> = HashMap::new();
    let mut preds: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut queue: VecDeque<u64> = VecDeque::new();
    dist.insert(source, 0);
    queue.push_back(source);
    let mut scratch: Vec<u64> = Vec::new();
    while let Some(node) = queue.pop_front() {
        let d = dist[&node];
        scratch.clear();
        view.get_neighbors_into(node, &mut scratch);
        for &n in &scratch {
            match dist.get(&n) {
                None => {
                    dist.insert(n, d + 1);
                    preds.entry(n).or_default().push(node);
                    queue.push_back(n);
                }
                Some(&dn) if dn == d + 1 => {
                    preds.entry(n).or_default().push(node);
                }
                _ => {}
            }
        }
    }
    if !dist.contains_key(&target) {
        return Vec::new();
    }
    // Enumerate paths by walking predecessors from the target back to the source.
    let mut paths: Vec<Vec<u64>> = Vec::new();
    let mut stack: Vec<Vec<u64>> = vec![vec![target]];
    while let Some(partial) = stack.pop() {
        let last = *partial.last().unwrap_or(&target);
        if last == source {
            let mut path = partial;
            path.reverse();
            paths.push(path);
            continue;
        }
        if let Some(ps) = preds.get(&last) {
            for &p in ps {
                let mut next = partial.clone();
                next.push(p);
                stack.push(next);
            }
        }
    }
    paths.sort_unstable();
    paths.dedup();
    paths
}

/// Induced subgraph edges within `hops` of `seeds`.
fn subgraph_edges(view: &dyn GraphView, seeds: &[u64], hops: u32) -> Vec<(u64, u64)> {
    let region: HashSet<u64> = bfs_dist_seed(view, seeds, hops)
        .into_iter()
        .map(|(node, _, _)| node)
        .collect();
    let mut edges: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut scratch: Vec<u64> = Vec::new();
    for &u in &region {
        scratch.clear();
        view.get_neighbors_into(u, &mut scratch);
        for &v in &scratch {
            if region.contains(&v) {
                edges.insert((u, v));
            }
        }
    }
    edges.into_iter().collect()
}

/// The union of pairwise shortest paths between the seeds, as edges.
fn connecting_paths(view: &dyn GraphView, seeds: &[u64]) -> Vec<(u64, u64)> {
    let mut edges: BTreeSet<(u64, u64)> = BTreeSet::new();
    for i in 0..seeds.len() {
        for j in (i + 1)..seeds.len() {
            let path = bfs_path(view, seeds[i], seeds[j]);
            for w in path.windows(2) {
                edges.insert((w[0], w[1]));
            }
        }
    }
    edges.into_iter().collect()
}

/// Resolve a table name (unqualified, or `catalog.schema.table`) to its core
/// [`Table`] via the session's catalog.
///
/// The direct lookup uses the session's default catalog/schema for unqualified
/// names. If that misses, every catalog/schema is searched for the table name —
/// the table may be registered under a non-default catalog (e.g. the dbt
/// adapter's `database`), and a table function should not depend on which
/// catalog a surface happened to use.
pub(crate) async fn resolve_table(state: &SessionState, name: &str) -> Result<Arc<Table>> {
    let cfg = state.config_options();
    let parts: Vec<&str> = name.split('.').collect();
    let (catalog, schema, table) = match parts.as_slice() {
        [t] => (
            cfg.catalog.default_catalog.as_str(),
            cfg.catalog.default_schema.as_str(),
            *t,
        ),
        [s, t] => (cfg.catalog.default_catalog.as_str(), *s, *t),
        [c, s, t] => (*c, *s, *t),
        _ => {
            return plan_err!("invalid table name '{name}'");
        }
    };

    let list = state.catalog_list();

    // Direct lookup.
    if let Some(cat) = list.catalog(catalog) {
        if let Some(sch) = cat.schema(schema) {
            if let Some(provider) = sch.table(table).await? {
                return downcast_table(provider, name);
            }
        }
    }

    // Fallback: search every catalog/schema for the table name.
    for cat_name in list.catalog_names() {
        let Some(cat) = list.catalog(&cat_name) else {
            continue;
        };
        for sch_name in cat.schema_names() {
            let Some(sch) = cat.schema(&sch_name) else {
                continue;
            };
            if let Some(provider) = sch.table(table).await? {
                return downcast_table(provider, name);
            }
        }
    }

    plan_err!("table '{name}' not found")
}

/// Downcast a resolved provider to the core [`Table`].
pub(crate) fn downcast_table(provider: Arc<dyn TableProvider>, name: &str) -> Result<Arc<Table>> {
    provider
        .as_any()
        .downcast_ref::<BenoStreamTableProvider>()
        .map(|bs| bs.table.clone())
        .ok_or_else(|| DataFusionError::Plan(format!("'{name}' is not a BenoStreamDB table")))
}

/// Extract a string literal argument.
fn as_string(expr: &Expr) -> Result<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => Ok(s.clone()),
        other => plan_err!("expected a string literal, got {other:?}"),
    }
}

/// Extract an integer literal argument as `u64`.
fn as_u64(expr: &Expr) -> Result<u64> {
    match expr {
        Expr::Literal(ScalarValue::Int64(Some(n)), _) => Ok(*n as u64),
        Expr::Literal(ScalarValue::Int32(Some(n)), _) => Ok(*n as u64),
        Expr::Literal(ScalarValue::Int16(Some(n)), _) => Ok(*n as u64),
        Expr::Literal(ScalarValue::Int8(Some(n)), _) => Ok(*n as u64),
        Expr::Literal(ScalarValue::UInt64(Some(n)), _) => Ok(*n),
        Expr::Literal(ScalarValue::UInt32(Some(n)), _) => Ok(*n as u64),
        Expr::Literal(ScalarValue::UInt16(Some(n)), _) => Ok(*n as u64),
        Expr::Literal(ScalarValue::UInt8(Some(n)), _) => Ok(*n as u64),
        other => plan_err!("expected an integer literal, got {other:?}"),
    }
}

/// Parse a comma-separated `u64` seed list from a string literal.
fn as_seed_list(expr: &Expr) -> Result<Vec<u64>> {
    let raw = as_string(expr)?;
    let mut seeds = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let id = part
            .parse::<u64>()
            .map_err(|_| DataFusionError::Plan(format!("invalid seed node id '{part}'")))?;
        seeds.push(id);
    }
    Ok(seeds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sql::graph_udf::graph_view::SimpleGraph;

    fn line_graph(n: u64) -> SimpleGraph {
        let edges: Vec<(u64, u64)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        SimpleGraph::from_undirected_edges(&edges)
    }

    #[test]
    fn neighbors_are_hop_and_seed_annotated() {
        let g = line_graph(5);
        let rows = bfs_dist_seed(&g, &[0], 2);
        // 0@0, 1@1, 2@2
        assert_eq!(rows, vec![(0, 0, 0), (1, 1, 0), (2, 2, 0)]);
    }

    #[test]
    fn shortest_path_is_ordered_and_inclusive() {
        let g = line_graph(5);
        assert_eq!(bfs_path(&g, 0, 3), vec![0, 1, 2, 3]);
        assert_eq!(bfs_path(&g, 3, 0), vec![3, 2, 1, 0]);
        assert!(bfs_path(&g, 0, 0) == vec![0]);
    }

    #[test]
    fn all_shortest_paths_enumerates_ties() {
        // Diamond: 0 -> 1 -> 3 and 0 -> 2 -> 3.
        let g = SimpleGraph::from_directed_edges(&[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let paths = all_shortest_paths(&g, 0, 3);
        assert_eq!(paths, vec![vec![0, 1, 3], vec![0, 2, 3]]);
    }

    #[test]
    fn subgraph_edges_are_induced() {
        let g = line_graph(5);
        // Within 1 hop of 2: nodes {1,2,3}; induced edges (1,2),(2,3) (+ reverse).
        let edges = subgraph_edges(&g, &[2], 1);
        assert!(edges.contains(&(1, 2)));
        assert!(edges.contains(&(2, 3)));
        assert!(!edges.contains(&(0, 1)));
    }

    #[test]
    fn connecting_paths_union_pairwise() {
        let g = line_graph(5);
        let edges = connecting_paths(&g, &[0, 4]);
        assert_eq!(edges, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn seed_list_parses_and_trims() {
        let expr = Expr::Literal(ScalarValue::Utf8(Some(" 101, 205 ,309".to_string())), None);
        assert_eq!(as_seed_list(&expr).unwrap(), vec![101, 205, 309]);
    }

    #[test]
    fn schemas_are_stable() {
        assert_eq!(GraphTraversalKind::Neighbors.schema().fields().len(), 3);
        assert_eq!(GraphTraversalKind::ShortestPath.schema().fields().len(), 2);
        assert_eq!(
            GraphTraversalKind::AllShortestPaths.schema().fields().len(),
            1
        );
        assert_eq!(GraphTraversalKind::Subgraph.schema().fields().len(), 2);
    }
}
