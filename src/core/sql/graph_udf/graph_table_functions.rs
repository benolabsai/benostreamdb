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

use arrow::array::{
    Array, ArrayRef, Float64Array, Float64Builder, ListBuilder, UInt32Builder, UInt64Array,
    UInt64Builder,
};
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
    /// Global PageRank (`node`, `rank`).
    PageRank,
    /// Louvain community detection (`node`, `community`).
    Louvain,
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
            GraphTraversalKind::PageRank => "graph_pagerank",
            GraphTraversalKind::Louvain => "graph_louvain",
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
            GraphTraversalKind::PageRank => vec![
                Field::new("node", DataType::UInt64, false),
                Field::new("rank", DataType::Float64, false),
            ],
            GraphTraversalKind::Louvain => vec![
                Field::new("node", DataType::UInt64, false),
                Field::new("community", DataType::UInt64, false),
            ],
        };
        Arc::new(Schema::new(fields))
    }
}

/// All graph traversal table functions, for registration.
pub fn all_graph_table_functions() -> Vec<(String, Arc<dyn TableFunctionImpl>)> {
    let mut funcs: Vec<(String, Arc<dyn TableFunctionImpl>)> = Vec::new();
    let kinds: &[(GraphTraversalKind, &[&str])] = &[
        (
            GraphTraversalKind::Neighbors,
            &["graph_neighbors", "neighbors"],
        ),
        (
            GraphTraversalKind::ShortestPath,
            &["graph_shortest_path", "shortest_path"],
        ),
        (
            GraphTraversalKind::AllShortestPaths,
            &["graph_all_shortest_paths", "all_shortest_paths"],
        ),
        (
            GraphTraversalKind::Subgraph,
            &["graph_subgraph", "subgraph"],
        ),
        (
            GraphTraversalKind::ConnectingPaths,
            &["graph_connecting_paths", "connecting_paths"],
        ),
        (
            GraphTraversalKind::PageRank,
            &["graph_pagerank", "pagerank"],
        ),
        (GraphTraversalKind::Louvain, &["graph_louvain", "louvain"]),
    ];

    for &(kind, names) in kinds {
        let impl_arc: Arc<dyn TableFunctionImpl> = Arc::new(GraphTraversalTableFunc { kind });
        for &name in names {
            funcs.push((name.to_string(), impl_arc.clone()));
        }
    }
    funcs
}

/// The canonical names of every graph traversal table function.
pub fn graph_table_function_names() -> Vec<String> {
    vec![
        "graph_neighbors".to_string(),
        "graph_shortest_path".to_string(),
        "graph_all_shortest_paths".to_string(),
        "graph_subgraph".to_string(),
        "graph_connecting_paths".to_string(),
    ]
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
    damping: Option<f64>,
    resolution: Option<f32>,
    iterations: Option<u32>,
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
            damping: None,
            resolution: None,
            iterations: None,
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
            GraphTraversalKind::PageRank => {
                if let Some(d) = args.get(1) {
                    params.damping = Some(as_f64(d)?);
                }
                if let Some(it) = args.get(2) {
                    params.iterations = Some(as_u64(it)? as u32);
                }
                if let Some(m) = args.get(3) {
                    params.mode = parse_graph_mode(&as_string(m)?);
                }
                4
            }
            GraphTraversalKind::Louvain => {
                if let Some(r) = args.get(1) {
                    params.resolution = Some(as_f64(r)? as f32);
                }
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
            GraphTraversalKind::PageRank | GraphTraversalKind::Louvain => (Vec::new(), 0),
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
            GraphTraversalKind::PageRank => {
                let (nodes, ranks) = execute_pagerank(
                    view,
                    params.damping.unwrap_or(0.85),
                    params.iterations.unwrap_or(30),
                );
                vec![nodes, ranks]
            }
            GraphTraversalKind::Louvain => {
                let (nodes, comms) = execute_louvain(view, params.resolution.unwrap_or(1.0));
                vec![nodes, comms]
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

/// Extract a float literal argument as `f64`.
fn as_f64(expr: &Expr) -> Result<f64> {
    match expr {
        Expr::Literal(ScalarValue::Float64(Some(n)), _) => Ok(*n),
        Expr::Literal(ScalarValue::Float32(Some(n)), _) => Ok(*n as f64),
        Expr::Literal(ScalarValue::Int64(Some(n)), _) => Ok(*n as f64),
        Expr::Literal(ScalarValue::Int32(Some(n)), _) => Ok(*n as f64),
        Expr::Literal(ScalarValue::UInt64(Some(n)), _) => Ok(*n as f64),
        Expr::Literal(ScalarValue::UInt32(Some(n)), _) => Ok(*n as f64),
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| DataFusionError::Plan(format!("cannot parse '{s}' as float"))),
        other => plan_err!("expected a float literal, got {other:?}"),
    }
}

/// Parse seeds from an expression: string literal ('101,102'), integer literal (101),
/// or array/list ([101, 102], make_array(101, 102)).
fn as_seed_list(expr: &Expr) -> Result<Vec<u64>> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => {
            let mut seeds = Vec::new();
            for part in s.split(',') {
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
        Expr::Literal(scalar, _) => match scalar {
            ScalarValue::Int64(Some(n)) => Ok(vec![*n as u64]),
            ScalarValue::Int32(Some(n)) => Ok(vec![*n as u64]),
            ScalarValue::Int16(Some(n)) => Ok(vec![*n as u64]),
            ScalarValue::Int8(Some(n)) => Ok(vec![*n as u64]),
            ScalarValue::UInt64(Some(n)) => Ok(vec![*n]),
            ScalarValue::UInt32(Some(n)) => Ok(vec![*n as u64]),
            ScalarValue::UInt16(Some(n)) => Ok(vec![*n as u64]),
            ScalarValue::UInt8(Some(n)) => Ok(vec![*n as u64]),
            ScalarValue::List(arr) => {
                let list_arr = arr.as_ref();
                if list_arr.is_empty() {
                    return Ok(Vec::new());
                }
                let inner = list_arr.value(0);
                let mut out = Vec::new();
                if let Ok(u64s) =
                    crate::core::sql::graph_udf::graph_view::coerce_to_u64_array(&inner)
                {
                    for i in 0..u64s.len() {
                        if !u64s.is_null(i) {
                            out.push(u64s.value(i));
                        }
                    }
                }
                Ok(out)
            }
            other => plan_err!("expected a string, integer, or array seed literal, got {other:?}"),
        },
        Expr::ScalarFunction(func) if func.name() == "make_array" || func.name() == "array" => {
            let mut seeds = Vec::new();
            for arg in &func.args {
                seeds.push(as_u64(arg)?);
            }
            Ok(seeds)
        }
        other => plan_err!("expected a string, integer, or array seed literal, got {other:?}"),
    }
}

fn execute_pagerank(view: &dyn GraphView, damping: f64, iterations: u32) -> (ArrayRef, ArrayRef) {
    let mut nodes = view.all_nodes();
    if nodes.is_empty() {
        return (
            Arc::new(UInt64Array::from(Vec::<u64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
        );
    }
    nodes.sort_unstable();
    nodes.dedup();
    let num_nodes = nodes.len();
    let node_to_idx: HashMap<u64, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    let mut adj: Vec<Vec<usize>> = Vec::with_capacity(num_nodes);
    let mut scratch = Vec::new();
    for &u in &nodes {
        scratch.clear();
        view.get_neighbors_into(u, &mut scratch);
        let mapped: Vec<usize> = scratch
            .iter()
            .filter_map(|v| node_to_idx.get(v).copied())
            .collect();
        adj.push(mapped);
    }

    let mut scores = vec![1.0 / (num_nodes as f64); num_nodes];
    let mut new_scores = vec![0.0; num_nodes];
    let base_score = (1.0 - damping) / (num_nodes as f64);

    for _ in 0..iterations {
        new_scores.fill(base_score);
        let mut dangling = 0.0;

        for u in 0..num_nodes {
            let s = scores[u];
            let neighbors = &adj[u];
            if !neighbors.is_empty() {
                let trans = (damping * s) / (neighbors.len() as f64);
                for &v in neighbors {
                    new_scores[v] += trans;
                }
            } else {
                dangling += damping * s;
            }
        }

        if dangling > 0.0 {
            let share = dangling / (num_nodes as f64);
            for s in &mut new_scores {
                *s += share;
            }
        }
        std::mem::swap(&mut scores, &mut new_scores);
    }

    let mut node_builder = UInt64Builder::with_capacity(num_nodes);
    let mut rank_builder = Float64Builder::with_capacity(num_nodes);
    for i in 0..num_nodes {
        node_builder.append_value(nodes[i]);
        rank_builder.append_value(scores[i]);
    }
    (
        Arc::new(node_builder.finish()),
        Arc::new(rank_builder.finish()),
    )
}

fn execute_louvain(view: &dyn GraphView, resolution: f32) -> (ArrayRef, ArrayRef) {
    let mut nodes = view.all_nodes();
    if nodes.is_empty() {
        return (
            Arc::new(UInt64Array::from(Vec::<u64>::new())),
            Arc::new(UInt64Array::from(Vec::<u64>::new())),
        );
    }
    nodes.sort_unstable();
    nodes.dedup();
    let num_nodes = nodes.len();
    let node_to_idx: HashMap<u64, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    let mut adj: Vec<Vec<(usize, f32)>> = Vec::with_capacity(num_nodes);
    let mut degrees: Vec<f32> = vec![0.0; num_nodes];
    let mut total_weight: f32 = 0.0;
    let mut scratch = Vec::new();

    for &u in &nodes {
        scratch.clear();
        view.get_neighbors_into(u, &mut scratch);
        let mut u_adj = Vec::new();
        for &v in &scratch {
            if let Some(&v_idx) = node_to_idx.get(&v) {
                u_adj.push((v_idx, 1.0f32));
                degrees[node_to_idx[&u]] += 1.0;
                total_weight += 1.0;
            }
        }
        adj.push(u_adj);
    }

    let m2 = if total_weight == 0.0 {
        1.0
    } else {
        total_weight
    };
    let mut community: Vec<usize> = (0..num_nodes).collect();
    let mut comm_tot: Vec<f32> = degrees.clone();

    if total_weight > 0.0 {
        let mut moved = true;
        let mut passes = 0;
        while moved && passes < 20 {
            moved = false;
            passes += 1;
            for u in 0..num_nodes {
                let c_curr = community[u];
                let k_u = degrees[u];
                let mut comm_weights: HashMap<usize, f32> = HashMap::new();
                for &(v, w) in &adj[u] {
                    let c_v = community[v];
                    *comm_weights.entry(c_v).or_default() += w;
                }
                comm_tot[c_curr] -= k_u;
                let k_curr = comm_weights.get(&c_curr).copied().unwrap_or(0.0);

                let mut best_comm = c_curr;
                let mut best_delta = 0.0f32;

                for (&c, &k_c) in &comm_weights {
                    let delta =
                        k_c - k_curr - resolution * (comm_tot[c] - comm_tot[c_curr]) * k_u / m2;
                    if delta > best_delta {
                        best_delta = delta;
                        best_comm = c;
                    }
                }

                community[u] = best_comm;
                comm_tot[best_comm] += k_u;
                if best_comm != c_curr {
                    moved = true;
                }
            }
        }
    }

    let mut comm_remap: HashMap<usize, u64> = HashMap::new();
    let mut next_id: u64 = 0;
    let mut node_builder = UInt64Builder::with_capacity(num_nodes);
    let mut comm_builder = UInt64Builder::with_capacity(num_nodes);

    for (u, &n) in nodes.iter().enumerate() {
        let c = community[u];
        let cid = *comm_remap.entry(c).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        node_builder.append_value(n);
        comm_builder.append_value(cid);
    }

    (
        Arc::new(node_builder.finish()),
        Arc::new(comm_builder.finish()),
    )
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

        // Integer literal seed
        let expr_int = Expr::Literal(ScalarValue::Int64(Some(133)), None);
        assert_eq!(as_seed_list(&expr_int).unwrap(), vec![133]);

        // UInt32 literal seed
        let expr_u32 = Expr::Literal(ScalarValue::UInt32(Some(42)), None);
        assert_eq!(as_seed_list(&expr_u32).unwrap(), vec![42]);
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
        assert_eq!(GraphTraversalKind::PageRank.schema().fields().len(), 2);
        assert_eq!(GraphTraversalKind::Louvain.schema().fields().len(), 2);
    }
}
