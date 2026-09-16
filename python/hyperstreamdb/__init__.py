from typing import List, Optional, Union, Dict, Any
import os
from .hyperstreamdb import Device as _Device
from .hyperstreamdb import Table as _RustTable
from .hyperstreamdb import Session as _RustSession
from .hyperstreamdb import *

def _has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False

class Device:
    """
    HyperStreamDB Compute Device (CPU, CUDA, MPS, ROCm, Intel).
    - **Torch Alignment** - Automatically aliases `cuda` to `rocm` on AMD hardware if `torch.version.hip` is detected.
    """
    def __new__(cls, device: str = "cpu", index: Optional[int] = None):
        device = device.lower()
        # 1. Handle Torch-style alignment
        if device.startswith("cuda"):
            # If Torch is present and on AMD, or if native probing finds only ROCm
            if _has_torch():
                import torch
                if getattr(torch.version, 'hip', None):
                    return _Device("rocm", index=index)
            
            # If not using torch but only AMD hardware is present, alias cuda to rocm
            if not _Device.is_available("cuda") and _Device.is_available("rocm"):
                return _Device("rocm", index=index)
        
        if (device.startswith("xpu") or device == "intel"):
            return _Device("intel", index=index)

        # 2. Native direct mapping
        return _Device(device, index=index)

    @staticmethod
    def is_available(device_type: str) -> bool:
        device_type = device_type.lower()
        if device_type == "cuda":
            # Torch compatibility: 'cuda' is true if either NVIDIA or AMD is present
            return _Device.is_available("cuda") or _Device.is_available("rocm")
        if device_type == "xpu":
            return _Device.is_available("intel")
        return _Device.is_available(device_type)

    @staticmethod
    def list_available_backends():
        backends = ['cpu']
        # Use precise strings for native listing
        for b in ['cuda', 'rocm', 'mps', 'intel']:
            if _Device.is_available(b):
                backends.append(b)
        return backends

    @staticmethod
    def auto_detect():
        # 1. Check Torch first | **Torch Alignment** | ❌ No | ✅ ROCm-as-CUDA |
        if _has_torch():
            import torch
            if torch.cuda.is_available():
                if getattr(torch.version, 'hip', None):
                    return _Device("rocm")
                return _Device("cuda")
            
            # Check for Intel IPEX
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return _Device("intel")

        # 2. Fallback to native probing
        for b in ['cuda', 'rocm', 'mps', 'intel']:
            if _Device.is_available(b):
                return _Device(b)
        return _Device('cpu')

    @staticmethod
    def deactivate():
        _Device.deactivate()

_Device.backend = property(lambda self: self.type_name)
_Device.device_id = property(lambda self: self.index)
ComputeContext = Device
GPUContext = Device

from .embeddings import registry, EmbeddingFunction
import pandas as pd
try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import polars as pl
except ImportError:
    pl = None

class IndexType:
    """
    HyperStreamDB Indexing Algorithms.
    """
    HNSW = "hnsw"
    BM25 = "bm25"
    BLOOM = "bloom"
    BITMAP = "bitmap"
    INVERTED = "inverted"

def _resolve_uri(uri: str) -> str:
    if not uri.startswith(("s3://", "file://", "az://", "gs://", "http://", "https://")):
        return f"file://{os.abspath(uri)}" if hasattr(os, "abspath") else uri
    return uri

def open_table(uri: str, **kwargs) -> Table:
    """Open an existing HyperStreamDB table."""
    return Table(uri, **kwargs)
class Query:
    """
    Fluent Query interface for HyperStreamDB.
    """
    def __init__(self, table, filter_expr: Optional[str] = None):
        self._table = table
        self._filter = filter_expr
        self._vector_filter = None
        self._columns = None

    def filter(self, expr: str) -> 'Query':
        """Apply a SQL-like filter expression."""
        if self._filter:
            self._filter = f"({self._filter}) AND ({expr})"
        else:
            self._filter = expr
        return self

    def vector_search(self, query: Union[List[float], str], column: Optional[str] = None, k: int = 10, **kwargs) -> 'Query':
        """
        Apply a vector search filter.
        
        Args:
            query: The query vector (list of floats) or a string to be vectorized.
            column: The vector column to search against.
            k: Number of nearest neighbors to return.
            **kwargs: Additional parameters (e.g., n_probe).
        """
        self._vector_filter = {
            "column": column,
            "query": query,
            "k": k,
            **kwargs
        }
        return self

    def select(self, columns: List[str]) -> 'Query':
        """Select specific columns to return."""
        self._columns = columns
        return self

    def to_pandas(self, device: Optional[Any] = None):
        """Execute the query and return results as a Pandas DataFrame."""
        return self._table.to_pandas(
            filter=self._filter, 
            vector_filter=self._vector_filter, 
            columns=self._columns, 
            device=device
        )

    def to_arrow(self, device: Optional[Any] = None):
        """Execute the query and return results as an Arrow Table."""
        return self._table.to_arrow(
            filter=self._filter, 
            vector_filter=self._vector_filter, 
            columns=self._columns, 
            device=device
        )

    def execute(self, device: Optional[Any] = None, to_arrow: bool = False):
        """Execute the query and return results as a Pandas DataFrame (default) or Arrow Table."""
        if to_arrow:
            return self.to_arrow(device)
        return self.to_pandas(device)

class GraphRagResult:
    """
    Result container for Graph RAG searches (Local or Global search mode).
    
    Attributes:
        mode (str): Search mode ('local' or 'global').
        nodes (pd.DataFrame): Retrieved entities/documents, augmented with relevance/pagerank scores.
        edges (pd.DataFrame): Induced subgraph edges or inter-community relationships.
        seeds (List[Any]): Seed node IDs (local mode) or top community IDs (global mode).
        communities (Optional[pd.DataFrame]): Discovered communities and their entity groupings.
    """
    def __init__(
        self,
        mode: str,
        nodes: Any,
        edges: Any,
        seeds: List[Any],
        communities: Optional[Any] = None,
        paths: Optional[List[List[dict]]] = None,
    ):
        import pandas as pd
        self.mode = mode
        if isinstance(nodes, pd.DataFrame):
            self.nodes = nodes
        elif hasattr(nodes, "to_pandas"):
            self.nodes = nodes.to_pandas()
        elif nodes is not None:
            self.nodes = pd.DataFrame(nodes)
        else:
            self.nodes = pd.DataFrame()

        if isinstance(edges, pd.DataFrame):
            self.edges = edges
        elif hasattr(edges, "to_pandas"):
            self.edges = edges.to_pandas()
        elif edges is not None:
            self.edges = pd.DataFrame(edges)
        else:
            self.edges = pd.DataFrame()

        self.seeds = list(seeds) if seeds is not None else []
        
        if communities is not None:
            if isinstance(communities, pd.DataFrame):
                self.communities = communities
            elif hasattr(communities, "to_pandas"):
                self.communities = communities.to_pandas()
            else:
                self.communities = pd.DataFrame(communities)
        else:
            self.communities = None

        self.paths = paths if paths is not None else []

    def format_context(
        self,
        max_tokens: int = 4000,
        include_edges: bool = True,
        content_column: Optional[str] = None,
        id_column: Optional[str] = None,
    ) -> str:
        """
        Format retrieved graph knowledge into prompt-ready markdown for direct LLM injection.
        
        Args:
            max_tokens: Approximate max tokens (~4 chars per token) to cap output context.
            include_edges: Whether to include the relational edge section.
            content_column: Document content/text column name. If None, auto-detected.
            id_column: Node/document ID column name. If None, auto-detected.
            
        Returns:
            Clean markdown string summarizing discovered entities, relevance scores, and relational edges.
        """
        import pandas as pd
        char_limit = max_tokens * 4
        parts = [f"### Graph RAG Context (Mode: {self.mode})\n"]

        if self.mode == "global" and self.communities is not None and not self.communities.empty:
            if "level" in self.communities.columns:
                for lvl, group in self.communities.groupby("level"):
                    parts.append(f"#### Level {lvl} Communities")
                    for _, row in group.iterrows():
                        cid = row.get("community_id", row.name)
                        members = row.get("members", [])
                        top_ent = row.get("top_entities", [])
                        summary = row.get("summary", "")
                        title = row.get("title", f"Community {cid}")
                        parent = row.get("parent_community_id")
                        parent_str = f", Parent: {parent}" if parent is not None and pd.notna(parent) and int(parent) > 0 else ""
                        line = f"- **{title}** ({len(members)} entities{parent_str})"
                        if len(top_ent) > 0:
                            line += f" | Key Entities: {list(top_ent)}"
                        if summary:
                            line += f"\n  {summary}"
                        parts.append(line)
                    parts.append("")
            else:
                parts.append("#### Discovered Communities")
                for _, row in self.communities.iterrows():
                    cid = row.get("community_id", row.name)
                    members = row.get("members", [])
                    top_ent = row.get("top_entities", [])
                    summary = row.get("summary", "")
                    title = row.get("title", f"Community {cid}")
                    line = f"- **{title}** ({len(members)} entities)"
                    if len(top_ent) > 0:
                        line += f" | Key Entities: {list(top_ent)}"
                    if summary:
                        line += f"\n  {summary}"
                    parts.append(line)
                parts.append("")

        if self.nodes is not None and not self.nodes.empty:
            parts.append("#### Discovered Entities (Ranked by Relevance)")
            
            # Detect ID and Content columns
            id_col = id_column
            if not id_col:
                for c in ["id", "doc_id", "node", "node_id", "key"]:
                    if c in self.nodes.columns:
                        id_col = c
                        break
                if not id_col:
                    id_col = self.nodes.columns[0]
                    
            cnt_col = content_column
            if not cnt_col:
                for c in ["content", "text", "body", "summary", "description", "title"]:
                    if c in self.nodes.columns:
                        cnt_col = c
                        break
                        
            score_col = None
            for c in ["pagerank", "score", "relevance", "degree"]:
                if c in self.nodes.columns:
                    score_col = c
                    break

            for _, row in self.nodes.iterrows():
                nid = row.get(id_col, "Unknown")
                score_str = f" (Score: {row[score_col]:.4f})" if score_col and pd.notna(row.get(score_col)) else ""
                content = str(row.get(cnt_col, "")).strip() if cnt_col else ""
                if len(content) > 300:
                    content = content[:297] + "..."
                if content:
                    parts.append(f"- [Node {nid}]{score_str}: {content}")
                else:
                    parts.append(f"- [Node {nid}]{score_str}")
            parts.append("")

        if include_edges and self.edges is not None and not self.edges.empty:
            parts.append("#### Relational Graph Context")
            s_col = "source" if "source" in self.edges.columns else self.edges.columns[0]
            t_col = "target" if "target" in self.edges.columns else self.edges.columns[1]
            rel_col = "relation" if "relation" in self.edges.columns else None
            w_col = "weight" if "weight" in self.edges.columns else None

            for _, row in self.edges.iterrows():
                s = row[s_col]
                t = row[t_col]
                rel = f" -[{row[rel_col]}]->" if rel_col and pd.notna(row.get(rel_col)) else " ->"
                w_str = f" (weight: {row[w_col]:.2f})" if w_col and pd.notna(row.get(w_col)) else ""
                parts.append(f"- Node {s}{rel} Node {t}{w_str}")
            parts.append("")

        if self.paths:
            parts.append("#### Relationship Paths")
            for path in self.paths:
                path_str = []
                for step in path:
                    src = step.get("source", "Unknown")
                    tgt = step.get("target", "Unknown")
                    rel = step.get("relation", "")
                    if not path_str:
                        path_str.append(f"{src}")
                    if rel:
                        path_str.append(f" -[{rel}]-> {tgt}")
                    else:
                        path_str.append(f" -> {tgt}")
                parts.append("- " + "".join(path_str))
            parts.append("")

        full_text = "\n".join(parts)
        if len(full_text) > char_limit:
            full_text = full_text[:char_limit - 15] + "\n\n...[truncated]"
        return full_text

    def to_dict(self) -> Dict[str, Any]:
        """Convert Graph RAG result to a serializable dictionary."""
        return {
            "mode": self.mode,
            "seeds": self.seeds,
            "nodes": self.nodes.to_dict(orient="records") if self.nodes is not None else [],
            "edges": self.edges.to_dict(orient="records") if self.edges is not None else [],
            "communities": self.communities.to_dict(orient="records") if self.communities is not None else None,
        }

    def __repr__(self):
        n_nodes = len(self.nodes) if self.nodes is not None else 0
        n_edges = len(self.edges) if self.edges is not None else 0
        return f"GraphRagResult(mode='{self.mode}', nodes={n_nodes}, edges={n_edges}, seeds={self.seeds})"

class Table:
    """
    HyperStreamDB Table — Apache Iceberg/Parquet-compatible columnar vector store.

    **Default behaviour (v0.4.1+)**

    - ``index_all = False`` — Vector indexes are *not* built automatically.
      Call ``table.index_all = True`` or ``table.add_index(column, 'hnsw')``
      to enable indexing for a specific session or column.
    - ``autocommit = False`` — Writes accumulate in an in-memory buffer.
      Call ``table.commit()`` (or ``await table.commit_async()``) to persist
      data to Parquet and advance the Iceberg snapshot.

    These defaults exist for performance: automatic indexing previously caused
    silent 15-18 s HNSW build latency on every ``commit()`` for tables with
    vector columns, even when the user had not requested an index.

    Args:
        uri:         Table location (``file:///path`` or cloud URI).
        inner_table: Internal — do not pass directly.
        device:      Optional compute device for GPU-accelerated index builds.
        index_all:   Enable automatic indexing of all compatible columns.
                     Defaults to ``False``. Set ``True`` to restore legacy behaviour.
        primary_key: Column name (or list) to use as primary key.
        explain:     If ``True``, return query plans instead of results.
    """
    def __init__(self, uri: str, inner_table: Optional[_RustTable] = None, device: Optional[Any] = None, index_all: bool = False, primary_key: Optional[str] = None, explain: bool = False):
        uri = _resolve_uri(uri)
        self.explain = explain
        if inner_table:
            self._inner = inner_table
        else:
            self._inner = _RustTable(uri, device=device)
        self._inner.set_index_all(index_all)
        if primary_key:
            if isinstance(primary_key, str):
                self._inner.set_primary_key([primary_key])
            else:
                self._inner.set_primary_key(list(primary_key))
        self._embedding_configs = {}

    @classmethod
    def create(cls, uri: str, schema, device: Optional[Any] = None) -> 'Table':
        """Create a new table with an explicit schema."""
        uri = _resolve_uri(uri)
        return cls(uri, inner_table=_RustTable.create(uri, schema, device=device))

    @classmethod
    def create_partitioned(cls, uri: str, schema, partition_spec: Dict[str, Any], device: Optional[Any] = None) -> 'Table':
        """Create a new table with an explicit schema and partitioning."""
        uri = _resolve_uri(uri)
        return cls(uri, inner_table=_RustTable.create_partitioned(uri, schema, partition_spec, device=device))

    @classmethod
    def register_external(cls, uri: str, iceberg_metadata_uri: str, device: Optional[Any] = None) -> 'Table':
        """Register an existing Iceberg table."""
        uri = _resolve_uri(uri)
        return cls(uri, inner_table=_RustTable.register_external(uri, iceberg_metadata_uri), device=device)

    @property
    def columns(self) -> List[str]:
        """Return the list of column names in the table."""
        return self._inner.columns

    @property
    def schema(self):
        """Return the table schema as PyArrow Schema."""
        return self._inner.schema

    def __len__(self) -> int:
        """Return the number of rows in the table."""
        try:
            res = self.execute_sql("SELECT count(*) as cnt FROM t")
            df = res.to_pandas()
            return int(df["cnt"].iloc[0])
        except Exception:
            return 0

    def personalized_pagerank(
        self,
        seeds: List[int],
        damping: float = 0.85,
        alpha: Optional[float] = None,
        iterations: int = 30,
        directed: bool = False,
        seed_weights: Optional[List[float]] = None,
    ):
        """
        Calculate Personalized PageRank (PPR) biased towards seed nodes.
        Supports both `alpha` (NetworkX convention) and `damping` parameter names.
        Optionally accepts continuous `seed_weights` (HippoRAG-style) to modulate teleportation probabilities.
        """
        d = alpha if alpha is not None else damping
        return self._inner.personalized_pagerank(seeds, d, iterations, directed, seed_weights)

    def subgraph(
        self,
        seeds: List[int],
        hops: int = 1,
        directed: bool = False,
        allowed_relations: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ):
        """
        Extract multi-hop induced subgraph starting from seed nodes.
        If `allowed_relations` is provided, filters edge traversals to only the specified relations/predicates.
        If `time_column`/`time_start`/`time_end` are provided, filters edges by timestamp window.
        """
        predicates = []

        # Relation filtering
        if allowed_relations:
            rel_col = next((c for c in ["relation", "predicate", "type", "rel", "edge_type"] if c in self.columns), None)
            if rel_col:
                rel_list = ", ".join(f"'{r}'" for r in allowed_relations)
                predicates.append(f"{rel_col} IN ({rel_list})")

        # Temporal filtering
        if time_column and time_column in self.columns:
            if time_start:
                predicates.append(f"{time_column} >= '{time_start}'")
            if time_end:
                predicates.append(f"{time_column} <= '{time_end}'")

        if predicates:
            source_col, target_col = self.edge_endpoints() if self.is_edge_table() else ("source", "target")
            seed_sql = "make_array(" + ", ".join(f"arrow_cast({s}, 'UInt64')" for s in seeds) + ")" if seeds else "make_array()"
            where_clause = " AND ".join(predicates)
            query = f"SELECT unnest(subgraph(arrow_cast({source_col}, 'UInt64'), arrow_cast({target_col}, 'UInt64'), {seed_sql}, arrow_cast({hops}, 'UInt32'), {str(directed).lower()})) FROM t WHERE {where_clause}"
            return self.execute_sql(query)
        return self._inner.subgraph(seeds, hops, directed)

    def connecting_paths(
        self,
        seeds: List[int],
        directed: bool = False,
        max_depth: Optional[int] = None,
    ):
        """Extract pairwise shortest connecting paths between seed nodes."""
        return self._inner.connecting_paths(seeds, directed)

    def louvain_communities(self, resolution: float = 1.0) -> Any:
        """
        Run Louvain community detection returning community groupings.
        Returns Arrow Table / DataFrame with 'community' column (lists of node IDs).
        """
        return self._inner.louvain_communities(resolution)

    def degree_centrality(self) -> Any:
        """
        Calculate degree centrality for all nodes in the edge table.
        Returns Arrow Table / DataFrame with 'node' and 'degree' columns.
        """
        return self._inner.degree_centrality()

    def execute_sql(self, query: str) -> Any:
        """
        Execute an arbitrary SQL query against this table (registered as 't') using DataFusion,
        with full access to vector and graph aggregate UDFs.
        """
        return self._inner.execute_sql(query)

    # ── Graph Algorithm Wrappers (1:1 with Rust PyO3 bindings) ─────────

    def pagerank(self, damping: float = 0.85, iterations: int = 30) -> Any:
        """
        Calculate PageRank for all nodes in the edge table.
        Returns Arrow Table / DataFrame with 'node' and 'pagerank' columns.
        """
        return self._inner.pagerank(damping, iterations)

    def shortest_path(self, start_node: int, end_node: int) -> Any:
        """
        Find the shortest path between two nodes using BFS.
        Returns Arrow Table / DataFrame with the path as a list of node IDs.
        """
        return self._inner.shortest_path(start_node, end_node)

    def connected_components(self) -> Any:
        """
        Find weakly connected components in the graph.
        Returns Arrow Table / DataFrame with 'component' column (component ID per node).
        """
        return self._inner.connected_components()

    def strongly_connected_components(self) -> Any:
        """
        Find strongly connected components in a directed graph (Tarjan's algorithm).
        Returns Arrow Table / DataFrame with 'scc_id' column.
        """
        return self._inner.strongly_connected_components()

    def topological_sort(self) -> Any:
        """
        Topological ordering of nodes in a directed acyclic graph (DAG).
        Returns Arrow Table / DataFrame with 'node' column in topological order.
        """
        return self._inner.topological_sort()

    def graph_neighbors(self, node: int, hops: int = 1) -> Any:
        """
        Find all nodes reachable from `node` within `hops` steps.
        Returns Arrow Table / DataFrame with 'neighbor' column.
        """
        return self._inner.graph_neighbors(node, hops)

    def label_propagation_communities(self) -> Any:
        """
        Run Label Propagation community detection.
        Returns Arrow Table / DataFrame with 'community' column.
        """
        return self._inner.label_propagation_communities()

    def modularity(self) -> Any:
        """
        Calculate Newman's modularity score for the current community assignment.
        Requires 'source_community' and 'target_community' columns in the edge table.
        """
        return self._inner.modularity()

    def adamic_adar(self, node1: int, node2: int) -> Any:
        """
        Calculate Adamic-Adar link prediction score between two nodes.
        Higher scores indicate more common neighbors with low degree.
        """
        return self._inner.adamic_adar(node1, node2)

    def preferential_attachment(self, node1: int, node2: int) -> Any:
        """
        Calculate Preferential Attachment link prediction score between two nodes.
        Score = degree(node1) * degree(node2).
        """
        return self._inner.preferential_attachment(node1, node2)

    def resource_allocation(self, node1: int, node2: int) -> Any:
        """
        Calculate Resource Allocation Index between two nodes.
        Sum of 1/degree(w) for each common neighbor w.
        """
        return self._inner.resource_allocation(node1, node2)

    def resource_allocation_index(self, node1: int, node2: int) -> Any:
        """Alias for resource_allocation()."""
        return self._inner.resource_allocation_index(node1, node2)

    def jaccard_coefficient(self, node1: int, node2: int) -> Any:
        """
        Calculate Jaccard coefficient between the neighbor sets of two nodes.
        Returns |N(u) ∩ N(v)| / |N(u) ∪ N(v)|.
        """
        return self._inner.jaccard_coefficient(node1, node2)

    def clustering_coefficient(self) -> Any:
        """
        Calculate the local clustering coefficient for each node.
        Measures how clustered each node's neighborhood is (ratio of triangles to possible triangles).
        Returns Arrow Table / DataFrame with 'node' and 'clustering_coefficient' columns.
        """
        return self._inner.clustering_coefficient()

    def to_graphviz(self) -> Any:
        """
        Export the edge table as a Graphviz DOT string.
        Returns Arrow Table / DataFrame with a single 'dot' column containing the DOT representation.
        """
        return self._inner.to_graphviz()

    def export_graph_gml(self, filepath: str):
        """
        Export the edge table to a GML format file.
        Iterates over the underlying PyArrow data in batches.
        """
        schema = self.schema
        if "source" not in schema.names or "target" not in schema.names:
            raise ValueError("Table must contain 'source' and 'target' columns to export as a graph")
        
        has_weight = "weight" in schema.names
        has_relation = "relation" in schema.names
        has_predicate = "predicate" in schema.names
        has_type = "type" in schema.names
        
        rel_col = None
        if has_relation: rel_col = "relation"
        elif has_predicate: rel_col = "predicate"
        elif has_type: rel_col = "type"

        nodes = set()
        edges = []

        df = self.execute_sql("SELECT * FROM t").to_pandas()
        for _, row in df.iterrows():
            src = int(row["source"])
            tgt = int(row["target"])
            nodes.add(src)
            nodes.add(tgt)
            
            edge = {"source": src, "target": tgt}
            if has_weight:
                edge["weight"] = float(row["weight"])
            if rel_col:
                edge["relation"] = str(row[rel_col])
            edges.append(edge)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("graph [\n")
            f.write("  directed 1\n")
            
            for node in sorted(nodes):
                f.write("  node [\n")
                f.write(f"    id {node}\n")
                f.write("  ]\n")
                
            for edge in edges:
                f.write("  edge [\n")
                f.write(f"    source {edge['source']}\n")
                f.write(f"    target {edge['target']}\n")
                if "weight" in edge:
                    f.write(f"    weight {edge['weight']}\n")
                if "relation" in edge:
                    rel = edge['relation'].replace('"', "'")
                    f.write(f'    relation "{rel}"\n')
                f.write("  ]\n")
                
            f.write("]\n")

    def export_graph_graphml(self, filepath: str):
        """
        Export the edge table to a GraphML format file.
        """
        schema = self.schema
        if "source" not in schema.names or "target" not in schema.names:
            raise ValueError("Table must contain 'source' and 'target' columns to export as a graph")
            
        has_weight = "weight" in schema.names
        has_relation = "relation" in schema.names
        
        rel_col = None
        if has_relation: rel_col = "relation"
        elif "predicate" in schema.names: rel_col = "predicate"
        
        nodes = set()
        edges = []

        df = self.execute_sql("SELECT * FROM t").to_pandas()
        for _, row in df.iterrows():
            src = int(row["source"])
            tgt = int(row["target"])
            nodes.add(src)
            nodes.add(tgt)
            
            edge = {"source": src, "target": tgt}
            if has_weight:
                edge["weight"] = float(row["weight"])
            if rel_col:
                edge["relation"] = str(row[rel_col])
            edges.append(edge)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n')
            f.write('         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
            f.write('         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n')
            f.write('         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n')
            
            if has_weight:
                f.write('  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>\n')
            if rel_col:
                f.write('  <key id="relation" for="edge" attr.name="relation" attr.type="string"/>\n')
                
            f.write('  <graph id="G" edgedefault="directed">\n')
            
            for node in sorted(nodes):
                f.write(f'    <node id="{node}"/>\n')
                
            for edge in edges:
                f.write(f'    <edge source="{edge["source"]}" target="{edge["target"]}">\n')
                if "weight" in edge:
                    f.write(f'      <data key="weight">{edge["weight"]}</data>\n')
                if "relation" in edge:
                    rel = edge['relation'].replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                    f.write(f'      <data key="relation">{rel}</data>\n')
                f.write('    </edge>\n')
                
            f.write('  </graph>\n')
            f.write('</graphml>\n')

    # ── Graph Statistics (computed in Python from existing UDFs) ────────

    def graph_density(self) -> float:
        """
        Calculate graph density: 2|E| / (|V| x (|V| - 1)) for undirected graphs.
        Returns a float in [0.0, 1.0] where 1.0 is a complete graph.
        """
        try:
            df2 = self.execute_sql(
                "SELECT count(*) as edge_count FROM t"
            ).to_pandas()
            e = int(df2["edge_count"].iloc[0])
            df_v = self.execute_sql(
                "SELECT count(*) as n FROM ("
                "SELECT DISTINCT source AS node FROM t "
                "UNION "
                "SELECT DISTINCT target AS node FROM t)"
            ).to_pandas()
            v = int(df_v["n"].iloc[0])
            if v <= 1:
                return 0.0
            return (2.0 * e) / (v * (v - 1))
        except Exception:
            return 0.0

    def avg_path_length(self, sample_size: int = 100) -> float:
        """
        Estimate the average shortest path length by BFS from a sample of nodes.
        Uses the shortest_path UDF on sampled node pairs.
        Returns the mean path length across sampled reachable pairs.
        """
        try:
            import random
            df_nodes = self.execute_sql(
                "SELECT DISTINCT node FROM ("
                "SELECT source AS node FROM t UNION SELECT target AS node FROM t"
                f") LIMIT {sample_size}"
            ).to_pandas()
            nodes = df_nodes["node"].tolist()
            if len(nodes) < 2:
                return 0.0

            total_dist = 0
            count = 0
            pairs = []
            for _ in range(min(sample_size, len(nodes) * (len(nodes) - 1))):
                a, b = random.sample(nodes, 2)
                pairs.append((int(a), int(b)))

            for (a, b) in pairs:
                try:
                    result = self.shortest_path(a, b)
                    path_df = result.to_pandas()
                    if len(path_df) > 0:
                        path_len = len(path_df) - 1
                        if path_len > 0:
                            total_dist += path_len
                            count += 1
                except Exception:
                    continue
            return total_dist / count if count > 0 else float('inf')
        except Exception:
            return float('inf')

    # ── Subgraph Export ─────────────────────────────────────────────────

    def export_subgraph(
        self,
        seeds: Optional[List[int]] = None,
        hops: int = 2,
        format: str = "graphml",
        output_path: Optional[str] = None,
        node_labels: Optional['Table'] = None,
    ) -> str:
        """
        Export a subgraph for visualization and interop with external graph tools.

        Args:
            seeds: Starting node IDs (None = export entire graph).
            hops: Neighborhood depth from seeds.
            format: Output format ('graphml', 'json_ld', 'dot', 'edge_list').
            output_path: File path to write output (None = return as string).
            node_labels: Optional Table to enrich nodes with metadata.

        Returns:
            Serialized graph as a string.
        """
        import json as json_mod
        from xml.etree.ElementTree import Element, SubElement, tostring as xml_tostring

        # Collect edges (optionally filtered by subgraph)
        if seeds:
            subgraph_result = self.subgraph(seeds, hops=hops)
            subgraph_df = subgraph_result.to_pandas()
            if subgraph_df.empty:
                node_set: Optional[set] = set()
            else:
                col = subgraph_df.columns[0]
                node_set = set(int(x) for x in subgraph_df[col].tolist())
        else:
            node_set = None

        edges_df = self.to_pandas()
        src_col = "source"
        tgt_col = "target"

        if node_set is not None:
            mask = edges_df[src_col].isin(node_set) & edges_df[tgt_col].isin(node_set)
            edges_df = edges_df[mask]

        all_nodes = set(int(x) for x in edges_df[src_col].tolist()) | set(int(x) for x in edges_df[tgt_col].tolist())

        # Enrich nodes with labels if provided
        node_meta: Dict[int, Dict[str, Any]] = {}
        if node_labels is not None:
            try:
                labels_df = node_labels.to_pandas()
                id_col = next((c for c in ["_id", "node", "id", "node_id"] if c in labels_df.columns), None)
                if id_col:
                    for _, row in labels_df.iterrows():
                        nid = int(row[id_col]) if not isinstance(row[id_col], str) else row[id_col]
                        meta = {k: str(v) for k, v in row.items() if k != id_col and v is not None}
                        node_meta[nid] = meta
            except Exception:
                pass

        rel_col = next((c for c in ["relation", "predicate", "type", "rel", "edge_type"] if c in edges_df.columns), None)

        if format == "dot":
            return self._export_dot(all_nodes, edges_df, src_col, tgt_col, rel_col, node_meta, output_path)
        elif format == "graphml":
            return self._export_graphml(all_nodes, edges_df, src_col, tgt_col, rel_col, node_meta, output_path)
        elif format == "json_ld":
            return self._export_json_ld(all_nodes, edges_df, src_col, tgt_col, rel_col, node_meta, output_path)
        elif format == "edge_list":
            return self._export_edge_list(edges_df, src_col, tgt_col, rel_col, output_path)
        else:
            raise ValueError(f"Unsupported export format '{format}'. Supported: graphml, json_ld, dot, edge_list")

    def _export_dot(self, nodes, edges_df, src_col, tgt_col, rel_col, node_meta, output_path):
        lines = ["digraph G {"]
        for n in sorted(nodes):
            label = str(n)
            if n in node_meta:
                attrs = ", ".join(f'{k}="{v}"' for k, v in node_meta[n].items())
                lines.append(f'  {n} [label="{label}", {attrs}];')
            else:
                lines.append(f'  {n} [label="{label}"];')
        for _, row in edges_df.iterrows():
            s, t = int(row[src_col]), int(row[tgt_col])
            if rel_col and rel_col in row.index:
                lines.append(f'  {s} -> {t} [label="{row[rel_col]}"];')
            else:
                lines.append(f"  {s} -> {t};")
        lines.append("}")
        result = "\n".join(lines)
        if output_path:
            with open(output_path, "w") as f:
                f.write(result)
        return result

    def _export_graphml(self, nodes, edges_df, src_col, tgt_col, rel_col, node_meta, output_path):
        from xml.etree.ElementTree import Element, SubElement, tostring as xml_tostring
        root = Element("graphml", xmlns="http://graphml.graphstruct.org/xmlns")
        if node_meta:
            all_keys: set = set()
            for meta in node_meta.values():
                all_keys.update(meta.keys())
            for key in sorted(all_keys):
                SubElement(root, "key", id=f"d_{key}", attrib={"for": "node", "attr.name": key, "attr.type": "string"})
        if rel_col:
            SubElement(root, "key", id="d_relation", attrib={"for": "edge", "attr.name": "relation", "attr.type": "string"})

        graph = SubElement(root, "graph", id="G", edgedefault="directed")
        for n in sorted(nodes):
            node_el = SubElement(graph, "node", id=str(n))
            if n in node_meta:
                for k, v in node_meta[n].items():
                    data_el = SubElement(node_el, "data", key=f"d_{k}")
                    data_el.text = str(v)
        for idx, row in edges_df.iterrows():
            edge_el = SubElement(graph, "edge", source=str(int(row[src_col])), target=str(int(row[tgt_col])))
            if rel_col and rel_col in row.index and row[rel_col] is not None:
                data_el = SubElement(edge_el, "data", key="d_relation")
                data_el.text = str(row[rel_col])

        result = xml_tostring(root, encoding="unicode")
        if output_path:
            with open(output_path, "w") as f:
                f.write(result)
        return result

    def _export_json_ld(self, nodes, edges_df, src_col, tgt_col, rel_col, node_meta, output_path):
        import json as json_mod
        doc = {
            "@context": {
                "@vocab": "https://schema.org/",
                "source": "https://schema.org/identifier",
                "target": "https://schema.org/identifier",
            },
            "@graph": []
        }
        for n in sorted(nodes):
            node_obj: Dict[str, Any] = {"@id": f"node:{n}", "@type": "Thing"}
            if n in node_meta:
                node_obj.update(node_meta[n])
            doc["@graph"].append(node_obj)
        for _, row in edges_df.iterrows():
            edge_obj: Dict[str, Any] = {
                "@type": "Relationship",
                "source": f"node:{int(row[src_col])}",
                "target": f"node:{int(row[tgt_col])}",
            }
            if rel_col and rel_col in row.index and row[rel_col] is not None:
                edge_obj["relation"] = str(row[rel_col])
            doc["@graph"].append(edge_obj)
        result = json_mod.dumps(doc, indent=2)
        if output_path:
            with open(output_path, "w") as f:
                f.write(result)
        return result

    def _export_edge_list(self, edges_df, src_col, tgt_col, rel_col, output_path):
        lines = []
        for _, row in edges_df.iterrows():
            parts = [str(int(row[src_col])), str(int(row[tgt_col]))]
            if rel_col and rel_col in row.index and row[rel_col] is not None:
                parts.append(str(row[rel_col]))
            lines.append(",".join(parts))
        result = "\n".join(lines)
        if output_path:
            with open(output_path, "w") as f:
                f.write(result)
        return result

    def graph_rag_search(
        self,
        query: Union[List[float], str],
        edge_table: Optional['Table'] = None,
        doc_table: Optional['Table'] = None,
        mode: str = "local",
        vector_column: str = "embedding",
        id_column: Optional[str] = None,
        top_k: int = 5,
        hops: int = 2,
        alpha: float = 0.85,
        directed: bool = False,
        resolution: float = 1.0,
        community_table: Optional['Table'] = None,
        max_nodes: int = 50,
        device: Optional[Any] = None,
        allowed_relations: Optional[List[str]] = None,
        seed_weights: Optional[List[float]] = None,
        search_edges: bool = False,
        edge_vector_column: str = "embedding",
        time_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> 'GraphRagResult':
        """
        Execute end-to-end Graph RAG search combining vector retrieval and topological graph reasoning.
        
        Supports two search paradigms:
        - **Local Search** (`mode='local'`):
          1. Discovers seed entities using vector/keyword similarity on `doc_table` (and optionally `edge_table` if `search_edges=True`).
          2. Extracts multi-hop induced subgraph from `edge_table` around seed entities (optionally filtered by `allowed_relations`).
          3. Computes Personalized PageRank (PPR) rooted at seeds with continuous seed weights (HippoRAG-style) for topological grounding.
          4. Retrieves enriched context documents sorted by PageRank score.
          
        - **Global Search** (`mode='global'`):
          1. Detects communities in `edge_table` via multi-resolution Louvain clustering (or uses `community_table`).
          2. Evaluates community relevance against query seeds and graph centrality.
          3. Synthesizes macro-level corpus context across top communities.
          
        Args:
            query: Query vector (List[float]) or keyword text string.
            edge_table: Edge table containing graph topology. Defaults to `self` if `self.is_edge_table()`.
            doc_table: Document/entity table. Defaults to `self` if not an edge table.
            mode: 'local' (seed discovery + subgraph + PPR) or 'global' (community clustering + synthesis).
            vector_column: Name of vector/embedding column in doc_table.
            id_column: Name of node/entity ID column in doc_table. If None, auto-detected.
            top_k: Number of seeds (local mode) or top communities (global mode) to retrieve.
            hops: Max hops for multi-hop induced subgraph extraction.
            alpha: Personalized PageRank restart probability (default 0.85).
            directed: Whether graph traversal is directed.
            resolution: Louvain modularity resolution parameter (global mode).
            community_table: Optional pre-computed materialized community Iceberg table.
            max_nodes: Maximum number of enriched nodes to return.
            device: Optional compute device.
            allowed_relations: Optional list of relationship types to restrict subgraph exploration.
            seed_weights: Optional explicit continuous seed weights for HippoRAG-style PPR.
            search_edges: If True, executes Dual Vector-Graph RAG by searching edge embeddings in parallel.
            edge_vector_column: Embedding column name in edge_table when search_edges=True.
            
        Returns:
            GraphRagResult object with `.nodes`, `.edges`, `.seeds`, and `.format_context()` for prompt injection.
        """
        import pandas as pd

        if hasattr(query, "tolist"):
            query = query.tolist()

        # Resolve doc_table and edge_table
        if edge_table is None:
            if self.is_edge_table():
                edge_table = self
            elif doc_table is not None and doc_table.is_edge_table():
                edge_table = doc_table
                doc_table = self
        if doc_table is None:
            if not self.is_edge_table():
                doc_table = self
            elif edge_table is not None and not edge_table.is_edge_table():
                doc_table = edge_table
                edge_table = self

        if edge_table is None or doc_table is None:
            raise ValueError("graph_rag_search requires both a document table and an edge table.")

        # Auto-detect id_column
        if id_column is None:
            pk = doc_table.primary_key
            if pk and isinstance(pk, str) and pk in doc_table.columns:
                id_column = pk
            elif pk and isinstance(pk, (list, tuple)) and len(pk) > 0 and pk[0] in doc_table.columns:
                id_column = pk[0]
            else:
                for cand in ["id", "doc_id", "node_id", "node", "key"]:
                    if cand in doc_table.columns:
                        id_column = cand
                        break
                if not id_column:
                    id_column = doc_table.columns[0]

        # 1. Seed Discovery via vector or keyword search
        if isinstance(query, str) and vector_column not in doc_table._embedding_configs and not isinstance(query, list):
            seed_res = doc_table.vector_search(column=vector_column, query=query, k=top_k, device=device)
        else:
            seed_res = doc_table.search(column=vector_column, query=query, k=top_k, device=device)

        seed_df = seed_res if isinstance(seed_res, pd.DataFrame) else (seed_res.to_pandas() if hasattr(seed_res, "to_pandas") else pd.DataFrame(seed_res))

        seed_weight_map = {}
        if not seed_df.empty and id_column in seed_df.columns:
            for _, r in seed_df.iterrows():
                try:
                    s_id = int(r[id_column])
                    w = 1.0
                    if "_distance" in r and pd.notna(r["_distance"]):
                        w = 1.0 / (1.0 + max(float(r["_distance"]), 0.0))
                    elif "score" in r and pd.notna(r["score"]):
                        w = float(r["score"])
                    seed_weight_map[s_id] = max(seed_weight_map.get(s_id, 0.0), w)
                except (ValueError, TypeError):
                    pass

        # Dual Vector-Graph RAG (Semantic Edge Search)
        if search_edges and edge_table is not None and edge_vector_column in edge_table.columns:
            try:
                edge_search_res = edge_table.search(column=edge_vector_column, query=query, k=top_k, device=device)
                edge_search_df = edge_search_res if isinstance(edge_search_res, pd.DataFrame) else (edge_search_res.to_pandas() if hasattr(edge_search_res, "to_pandas") else pd.DataFrame(edge_search_res))
                source_col, target_col = edge_table.edge_endpoints() if edge_table.is_edge_table() else ("source", "target")
                if not edge_search_df.empty and source_col in edge_search_df.columns and target_col in edge_search_df.columns:
                    for _, er in edge_search_df.iterrows():
                        try:
                            u = int(er[source_col])
                            v = int(er[target_col])
                            ew = 1.0
                            if "_distance" in er and pd.notna(er["_distance"]):
                                ew = 1.0 / (1.0 + max(float(er["_distance"]), 0.0))
                            elif "score" in er and pd.notna(er["score"]):
                                ew = float(er["score"])
                            seed_weight_map[u] = max(seed_weight_map.get(u, 0.0), ew)
                            seed_weight_map[v] = max(seed_weight_map.get(v, 0.0), ew)
                        except (ValueError, TypeError):
                            pass
            except Exception:
                pass

        if seed_weights is not None:
            seeds = [int(s) for s in seed_df[id_column].tolist() if id_column in seed_df.columns] if not seed_df.empty else []
            active_weights = seed_weights
        elif seed_weight_map:
            seeds = list(seed_weight_map.keys())
            active_weights = [seed_weight_map[s] for s in seeds]
        else:
            seeds = []
            active_weights = None

        if mode.lower() == "local":
            if not seeds:
                return GraphRagResult(mode="local", nodes=pd.DataFrame(), edges=pd.DataFrame(), seeds=[])

            # 2. Multi-hop Induced Subgraph
            sub_res = edge_table.subgraph(
                seeds,
                hops=hops,
                directed=directed,
                allowed_relations=allowed_relations,
                time_column=time_column,
                time_start=time_start,
                time_end=time_end,
            )
            subgraph_df = sub_res if isinstance(sub_res, pd.DataFrame) else (sub_res.to_pandas() if hasattr(sub_res, "to_pandas") else pd.DataFrame(sub_res))

            neighborhood_nodes = set(seeds)
            if not subgraph_df.empty:
                if "source" in subgraph_df.columns:
                    neighborhood_nodes.update([int(x) for x in subgraph_df["source"]])
                if "target" in subgraph_df.columns:
                    neighborhood_nodes.update([int(x) for x in subgraph_df["target"]])

            # 3. Personalized PageRank Topological Grounding
            try:
                ppr_res = edge_table.personalized_pagerank(seeds, alpha=alpha, directed=directed, seed_weights=active_weights)
                ppr_df = ppr_res if isinstance(ppr_res, pd.DataFrame) else (ppr_res.to_pandas() if hasattr(ppr_res, "to_pandas") else pd.DataFrame(ppr_res))
                if not ppr_df.empty and "node" in ppr_df.columns and "score" in ppr_df.columns:
                    ppr_df = ppr_df[ppr_df["node"].isin(neighborhood_nodes)]
                    ppr_df = ppr_df.sort_values(by="score", ascending=False)
                else:
                    ppr_df = pd.DataFrame({"node": list(neighborhood_nodes), "score": [1.0 if n in seeds else 0.5 for n in neighborhood_nodes]})
            except Exception:
                ppr_df = pd.DataFrame({"node": list(neighborhood_nodes), "score": [1.0 if n in seeds else 0.5 for n in neighborhood_nodes]})

            ranked_nodes = ppr_df["node"].head(max_nodes).tolist() if not ppr_df.empty else list(neighborhood_nodes)[:max_nodes]

            # 4. Context Enrichment
            if ranked_nodes:
                nodes_sql = ", ".join(map(str, ranked_nodes))
                doc_res = doc_table.execute_sql(f"SELECT * FROM t WHERE {id_column} IN ({nodes_sql})")
                nodes_df = doc_res if isinstance(doc_res, pd.DataFrame) else (doc_res.to_pandas() if hasattr(doc_res, "to_pandas") else pd.DataFrame(doc_res))
                
                # Merge PPR scores
                if not ppr_df.empty and id_column in nodes_df.columns:
                    nodes_df[id_column] = nodes_df[id_column].astype(int)
                    ppr_df["node"] = ppr_df["node"].astype(int)
                    nodes_df = nodes_df.merge(ppr_df[["node", "score"]], left_on=id_column, right_on="node", how="left")
                    nodes_df = nodes_df.rename(columns={"score": "pagerank"})
                    if "node" in nodes_df.columns and id_column != "node":
                        nodes_df = nodes_df.drop(columns=["node"])
                    nodes_df = nodes_df.sort_values(by="pagerank", ascending=False)
            else:
                nodes_df = pd.DataFrame()

            # 5. Extract Paths
            paths = []
            if len(seeds) > 1:
                try:
                    path_res = edge_table.connecting_paths(seeds, directed=directed)
                    path_df = path_res if isinstance(path_res, pd.DataFrame) else (path_res.to_pandas() if hasattr(path_res, "to_pandas") else pd.DataFrame(path_res))
                    if not path_df.empty:
                        # Find relation/label mappings
                        node_map = {}
                        if not nodes_df.empty and id_column in nodes_df.columns:
                            cnt_col = next((c for c in ["content", "text", "body", "summary", "description", "title"] if c in nodes_df.columns), None)
                            if cnt_col:
                                for _, row in nodes_df.iterrows():
                                    val = str(row[cnt_col]).strip()
                                    node_map[int(row[id_column])] = val[:30] + "..." if len(val) > 30 else val
                        
                        # connecting_paths returns columns like: path_id, step, source, target, (relation)
                        s_col = "source" if "source" in path_df.columns else None
                        t_col = "target" if "target" in path_df.columns else None
                        r_col = "relation" if "relation" in path_df.columns else None
                        
                        if s_col and t_col:
                            if "path_id" in path_df.columns:
                                for path_id, group in path_df.groupby("path_id"):
                                    group = group.sort_values(by="step") if "step" in group.columns else group
                                    cur_path = []
                                    for _, r in group.iterrows():
                                        src = int(r[s_col])
                                        tgt = int(r[t_col])
                                        rel = str(r[r_col]) if r_col and pd.notna(r.get(r_col)) else ""
                                        cur_path.append({
                                            "source": node_map.get(src, f"Node {src}"),
                                            "target": node_map.get(tgt, f"Node {tgt}"),
                                            "relation": rel
                                        })
                                    if cur_path:
                                        paths.append(cur_path)
                            else:
                                # Fallback if path_id is missing
                                cur_path = []
                                for _, r in path_df.iterrows():
                                    src = int(r[s_col])
                                    tgt = int(r[t_col])
                                    rel = str(r[r_col]) if r_col and pd.notna(r.get(r_col)) else ""
                                    cur_path.append({
                                        "source": node_map.get(src, f"Node {src}"),
                                        "target": node_map.get(tgt, f"Node {tgt}"),
                                        "relation": rel
                                    })
                                if cur_path:
                                    paths.append(cur_path)
                except Exception:
                    pass

            return GraphRagResult(mode="local", nodes=nodes_df, edges=subgraph_df, seeds=seeds, paths=paths)

        elif mode.lower() == "global":
            # Global Search Mode
            if community_table is not None:
                comm_res = community_table.execute_sql("SELECT * FROM t")
                comm_df = comm_res if isinstance(comm_res, pd.DataFrame) else (comm_res.to_pandas() if hasattr(comm_res, "to_pandas") else pd.DataFrame(comm_res))
            else:
                louvain_res = edge_table.louvain_communities(resolution=resolution)
                louvain_df = louvain_res if isinstance(louvain_res, pd.DataFrame) else (louvain_res.to_pandas() if hasattr(louvain_res, "to_pandas") else pd.DataFrame(louvain_res))
                deg_res = edge_table.degree_centrality()
                deg_df = deg_res if isinstance(deg_res, pd.DataFrame) else (deg_res.to_pandas() if hasattr(deg_res, "to_pandas") else pd.DataFrame(deg_res))
                deg_map = dict(zip(deg_df["node"].astype(int), deg_df["degree"])) if not deg_df.empty else {}

                comm_records = []
                for c_idx, row in louvain_df.iterrows():
                    members = [int(m) for m in row["community"]]
                    sorted_members = sorted(members, key=lambda m: deg_map.get(m, 0), reverse=True)
                    seed_hits = sum(1 for m in members if m in seeds)
                    comm_records.append({
                        "community_id": c_idx,
                        "member_count": len(members),
                        "members": members,
                        "top_entities": sorted_members[:5],
                        "seed_overlap": seed_hits,
                        "title": f"Community {c_idx} ({len(members)} entities)",
                        "summary": f"Community {c_idx} consisting of {len(members)} entities centered around {sorted_members[:3]}."
                    })
                comm_df = pd.DataFrame(comm_records)

            if not comm_df.empty:
                if "seed_overlap" in comm_df.columns and comm_df["seed_overlap"].sum() > 0:
                    comm_df = comm_df.sort_values(by=["seed_overlap", "member_count"], ascending=[False, False])
                else:
                    comm_df = comm_df.sort_values(by="member_count", ascending=False)

            top_comm_df = comm_df.head(top_k)
            top_comm_ids = top_comm_df["community_id"].tolist() if not top_comm_df.empty else []

            selected_nodes = []
            for _, row in top_comm_df.iterrows():
                ents = row.get("top_entities", row.get("members", []))
                selected_nodes.extend([int(x) for x in ents])
            selected_nodes = list(dict.fromkeys(selected_nodes))[:max_nodes]

            if selected_nodes:
                nodes_sql = ", ".join(map(str, selected_nodes))
                doc_res = doc_table.execute_sql(f"SELECT * FROM t WHERE {id_column} IN ({nodes_sql})")
                nodes_df = doc_res if isinstance(doc_res, pd.DataFrame) else (doc_res.to_pandas() if hasattr(doc_res, "to_pandas") else pd.DataFrame(doc_res))
            else:
                nodes_df = pd.DataFrame()

            if len(selected_nodes) > 1:
                sub_res = edge_table.subgraph(
                    selected_nodes,
                    hops=1,
                    directed=directed,
                    time_column=time_column,
                    time_start=time_start,
                    time_end=time_end,
                )
                edges_df = sub_res if isinstance(sub_res, pd.DataFrame) else (sub_res.to_pandas() if hasattr(sub_res, "to_pandas") else pd.DataFrame(sub_res))
            else:
                edges_df = pd.DataFrame()

            return GraphRagResult(
                mode="global",
                nodes=nodes_df,
                edges=edges_df,
                seeds=top_comm_ids,
                communities=top_comm_df,
            )
        else:
            raise ValueError(f"Unknown graph RAG mode: '{mode}'. Supported modes: 'local', 'global'.")

    def summarize_communities(
        self,
        doc_table: 'Table',
        target_uri: str,
        id_column: Optional[str] = None,
        content_column: Optional[str] = None,
        resolution: float = 1.0,
        resolutions: Optional[List[float]] = None,
        hierarchical: bool = False,
        top_entities_per_comm: int = 5,
        device: Optional[Any] = None,
    ) -> 'Table':
        """
        Summarize Louvain communities in this edge table using text context from doc_table,
        and materialize the results as an Apache Iceberg table at target_uri.
        
        Supports both single-level clustering and hierarchical multi-resolution Louvain pyramids
        (with `level` and `parent_community_id` tracking for Microsoft GraphRAG parity).
        
        Args:
            doc_table: Document/entity Table containing text descriptions.
            target_uri: Target URI where the community Iceberg table will be materialized.
            id_column: Node/entity identifier column in doc_table. Auto-detected if None.
            content_column: Text content column in doc_table. Auto-detected if None.
            resolution: Single-resolution Louvain modularity parameter (used when hierarchical=False).
            resolutions: List of resolutions across hierarchical levels (defaults to [0.5, 1.0, 2.0] if hierarchical=True).
            hierarchical: Whether to build a multi-level hierarchical community tree.
            top_entities_per_comm: Number of top central entities to feature per community.
            device: Optional compute device.
            
        Returns:
            Table instance pointing to the newly materialized Iceberg community table.
        """
        import pyarrow as pa
        import pandas as pd

        target_uri = _resolve_uri(target_uri)

        # Identify id_column
        if id_column is None:
            pk = doc_table.primary_key
            if pk and isinstance(pk, str) and pk in doc_table.columns:
                id_column = pk
            elif pk and isinstance(pk, (list, tuple)) and len(pk) > 0 and pk[0] in doc_table.columns:
                id_column = pk[0]
            else:
                for cand in ["id", "doc_id", "node_id", "node", "key"]:
                    if cand in doc_table.columns:
                        id_column = cand
                        break
                if not id_column:
                    id_column = doc_table.columns[0]

        # Identify content_column
        if content_column is None:
            for cand in ["content", "text", "body", "summary", "description", "title"]:
                if cand in doc_table.columns:
                    content_column = cand
                    break
            if not content_column:
                content_column = doc_table.columns[-1]

        # Degree Centrality for ranking entities within communities
        deg_res = self.degree_centrality()
        deg_df = deg_res if isinstance(deg_res, pd.DataFrame) else (deg_res.to_pandas() if hasattr(deg_res, "to_pandas") else pd.DataFrame(deg_res))
        deg_map = dict(zip(deg_df["node"].astype(int), deg_df["degree"])) if not deg_df.empty else {}

        # Fetch doc text
        docs_res = doc_table.execute_sql("SELECT * FROM t")
        docs_df = docs_res if isinstance(docs_res, pd.DataFrame) else (docs_res.to_pandas() if hasattr(docs_res, "to_pandas") else pd.DataFrame(docs_res))

        doc_map = {}
        title_map = {}
        if not docs_df.empty and id_column in docs_df.columns:
            for _, r in docs_df.iterrows():
                try:
                    nid = int(r[id_column])
                    doc_map[nid] = str(r[content_column]) if content_column in r and pd.notna(r[content_column]) else ""
                    if "title" in r and pd.notna(r["title"]):
                        title_map[nid] = str(r["title"])
                except Exception:
                    pass

        active_resolutions = (resolutions or [0.5, 1.0, 2.0]) if hierarchical else [resolution]

        records = []
        global_comm_id = 0
        prev_level_comms = []

        for lvl, res in enumerate(active_resolutions):
            louvain_res = self.louvain_communities(resolution=res)
            louvain_df = louvain_res if isinstance(louvain_res, pd.DataFrame) else (louvain_res.to_pandas() if hasattr(louvain_res, "to_pandas") else pd.DataFrame(louvain_res))
            current_level_comms = []

            for _, row in louvain_df.iterrows():
                members = [int(m) for m in row["community"]]
                member_set = set(members)
                sorted_members = sorted(members, key=lambda m: deg_map.get(m, 0), reverse=True)
                top_ent = sorted_members[:top_entities_per_comm]

                ent_titles = [title_map.get(m, f"Entity {m}") for m in top_ent]
                title = f"L{lvl} Comm {global_comm_id}: {', '.join(ent_titles[:3])}" if hierarchical else f"Community {global_comm_id}: {', '.join(ent_titles[:3])}"

                contexts = [f"[{title_map.get(m, f'Node {m}')}]: {doc_map.get(m, '')}" for m in top_ent if m in doc_map and doc_map.get(m)]
                summary_text = f"Community {global_comm_id} (Level {lvl}) contains {len(members)} entities. Key entities: {', '.join(map(str, top_ent))}. " + " | ".join(contexts)

                parent_id = 0
                if lvl > 0 and prev_level_comms:
                    best_overlap = -1
                    for p_id, p_members in prev_level_comms:
                        overlap = len(member_set.intersection(p_members))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            parent_id = p_id

                records.append({
                    "community_id": global_comm_id,
                    "title": title,
                    "member_count": len(members),
                    "members": members,
                    "top_entities": top_ent,
                    "summary": summary_text,
                    "level": lvl,
                    "parent_community_id": parent_id,
                })
                current_level_comms.append((global_comm_id, member_set))
                global_comm_id += 1

            prev_level_comms = current_level_comms

        schema = pa.schema([
            ("community_id", pa.uint64()),
            ("title", pa.string()),
            ("member_count", pa.uint32()),
            ("members", pa.list_(pa.uint64())),
            ("top_entities", pa.list_(pa.uint64())),
            ("summary", pa.string()),
            ("level", pa.uint32()),
            ("parent_community_id", pa.uint64()),
        ])

        arrow_table = pa.Table.from_pylist(records, schema=schema)
        comm_table = Table.create(target_uri, schema, device=device)
        comm_table.insert(arrow_table)
        comm_table.commit()
        return comm_table

    def resolve_entities(
        self,
        relation: str = "same_as",
        source_col: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> Dict[int, int]:
        """
        Compute transitive equivalence closure over alias / same_as edges in this edge table.
        
        Uses Disjoint Set Union (Union-Find) with path compression to partition alias nodes
        into connected components, electing the minimal node ID as the canonical representative.
        
        Args:
            relation: Edge relationship string to treat as equivalence (default: 'same_as').
            source_col: Source node column name. Auto-detected if None.
            target_col: Target node column name. Auto-detected if None.
            
        Returns:
            Dict mapping each node ID to its canonical resolved node ID {node_id: canonical_id}.
        """
        import pandas as pd
        
        if source_col is None or target_col is None:
            if self.is_edge_table():
                s_col, t_col = self.edge_endpoints()
                source_col = source_col or s_col
                target_col = target_col or t_col
            else:
                source_col = source_col or "source"
                target_col = target_col or "target"
                
        rel_col = next((c for c in ["relation", "predicate", "type", "rel", "edge_type"] if c in self.columns), None)
        if rel_col:
            sql = f"SELECT {source_col}, {target_col} FROM t WHERE {rel_col} = '{relation}'"
        else:
            sql = f"SELECT {source_col}, {target_col} FROM t"
            
        res = self.execute_sql(sql)
        df = res if isinstance(res, pd.DataFrame) else (res.to_pandas() if hasattr(res, "to_pandas") else pd.DataFrame(res))
        
        parent: Dict[int, int] = {}
        
        def find(u: int) -> int:
            path = []
            while u in parent and parent[u] != u:
                path.append(u)
                u = parent[u]
            for node in path:
                parent[node] = u
            return u
            
        def union(u: int, v: int):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if root_u < root_v:
                    parent[root_v] = root_u
                else:
                    parent[root_u] = root_v

        if not df.empty and source_col in df.columns and target_col in df.columns:
            for _, row in df.iterrows():
                try:
                    u = int(row[source_col])
                    v = int(row[target_col])
                    if u not in parent:
                        parent[u] = u
                    if v not in parent:
                        parent[v] = v
                    union(u, v)
                except (ValueError, TypeError):
                    pass
                    
        return {node: find(node) for node in parent}

    def hybrid_search(
        self,
        text_column: str,
        query_text: str,
        vector_column: str,
        query_vector: List[float],
        k: int = 10,
        rrf_k: int = 60,
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
        device: Optional[Any] = None,
    ) -> Any:
        """
        Perform hybrid search combining vector similarity and BM25 keyword search using Reciprocal Rank Fusion (RRF).
        
        RRF Score: score = sum(1.0 / (rrf_k + rank + 1.0))
        
        Args:
            text_column: Text column for BM25 keyword search.
            query_text: Keyword or query string.
            vector_column: Embedding column for dense vector search.
            query_vector: Dense float vector.
            k: Number of fused top results to return.
            rrf_k: RRF smoothing constant (default: 60).
            filter: Optional SQL filter expression.
            columns: Optional list of columns to return.
            device: Optional compute device.
            
        Returns:
            pandas.DataFrame of top results sorted by `rrf_score` descending.
        """
        import pandas as pd

        try:
            vec_res = self.vector_search(
                column=vector_column,
                query=query_vector,
                k=k * 2,
                filter=filter,
                columns=columns,
                device=device,
            )
            vec_df = vec_res if isinstance(vec_res, pd.DataFrame) else (vec_res.to_pandas() if hasattr(vec_res, "to_pandas") else pd.DataFrame(vec_res))
        except Exception:
            vec_df = pd.DataFrame()

        try:
            text_res = self.vector_search(
                column=text_column,
                query=query_text,
                k=k * 2,
                filter=filter,
                columns=columns,
                device=device,
            )
            text_df = text_res if isinstance(text_res, pd.DataFrame) else (text_res.to_pandas() if hasattr(text_res, "to_pandas") else pd.DataFrame(text_res))
        except Exception:
            # Fallback to SQL text matching if inverted index is not built on column
            keywords = [w.strip() for w in query_text.split() if len(w.strip()) > 0]
            if keywords:
                like_clauses = ["LOWER(" + text_column + ") LIKE '%" + w.lower().replace("'", "''") + "%'" for w in keywords]
                where_clause = " OR ".join(like_clauses)
                if filter:
                    where_clause = f"({filter}) AND ({where_clause})"
                try:
                    sql_res = self.execute_sql(f"SELECT * FROM t WHERE {where_clause}")
                    text_df = sql_res if isinstance(sql_res, pd.DataFrame) else (sql_res.to_pandas() if hasattr(sql_res, "to_pandas") else pd.DataFrame(sql_res))
                except Exception:
                    text_df = pd.DataFrame()
            else:
                text_df = pd.DataFrame()

        # Identity column
        id_col = self.primary_key
        if id_col and isinstance(id_col, (list, tuple)) and len(id_col) == 1:
            id_col = id_col[0]
        if not id_col or (not vec_df.empty and id_col not in vec_df.columns):
            for cand in ["id", "doc_id", "node_id", "key", "uuid"]:
                if (not vec_df.empty and cand in vec_df.columns) or (not text_df.empty and cand in text_df.columns):
                    id_col = cand
                    break
            if not id_col:
                cols = vec_df.columns if not vec_df.empty else text_df.columns
                id_col = cols[0] if len(cols) > 0 else "id"

        scores = {}
        rows_by_id = {}

        if not vec_df.empty and id_col in vec_df.columns:
            for rank, (_, row) in enumerate(vec_df.iterrows()):
                key = row[id_col]
                scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank + 1.0))
                rows_by_id[key] = row.to_dict()

        if not text_df.empty and id_col in text_df.columns:
            for rank, (_, row) in enumerate(text_df.iterrows()):
                key = row[id_col]
                scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank + 1.0))
                if key not in rows_by_id:
                    rows_by_id[key] = row.to_dict()

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        fused_rows = []
        for key, score in sorted_items:
            r = dict(rows_by_id[key])
            r["rrf_score"] = score
            fused_rows.append(r)

        return pd.DataFrame(fused_rows)

    def is_edge_table(self) -> bool:
        """Check whether this table conforms to the standard edge table convention."""
        cols = self.columns
        has_source = any(c in cols for c in ["source", "source_id", "src", "u"])
        has_target = any(c in cols for c in ["target", "target_id", "dst", "v"])
        return has_source and has_target

    def edge_endpoints(self) -> tuple:
        """Return the (source, target) column names for this edge table."""
        cols = self.columns
        source_col = next((c for c in ["source", "source_id", "src", "u"] if c in cols), None)
        target_col = next((c for c in ["target", "target_id", "dst", "v"] if c in cols), None)
        if not source_col or not target_col:
            raise ValueError(f"Table does not have standard edge endpoints. Columns: {cols}")
        return (source_col, target_col)

    @classmethod
    def create_edge_table(
        cls,
        uri: str,
        schema: Optional[Any] = None,
        directed: bool = True,
        node_id_type: str = "uint64",
        with_relation: bool = True,
        with_weight: bool = True,
        embedding_dim: Optional[int] = None,
        index_endpoints: bool = True,
        index_embedding: bool = True,
        partition_by_relation: bool = False,
        device: Optional[Any] = None,
    ) -> 'Table':
        """
        Create a new edge table following the standard Iceberg edge table schema convention:
        - source: UInt64 or String node identifier
        - target: UInt64 or String node identifier
        - relation: String relationship type (e.g. 'cites', 'mentions', 'related_to')
        - weight: Float64 edge weight (defaults to 1.0)
        - embedding: Optional List of Float32 for semantic edge search
        
        Args:
            uri: Table location.
            schema: Optional explicit pyarrow.Schema.
            directed: Whether graph is directed.
            node_id_type: 'uint64' (default, optimal for graph UDAFs) or 'string'.
            with_relation: Include 'relation' string column.
            with_weight: Include 'weight' float64 column.
            embedding_dim: Optional vector dimension for edge embeddings.
            index_endpoints: Automatically configure sidecar Bitmap indexes on 'source' and 'target'.
            index_embedding: Automatically configure HNSW index on 'embedding' if present.
            partition_by_relation: Partition Iceberg table by relation type.
            device: Compute device.
        """
        import pyarrow as pa
        uri = _resolve_uri(uri)
        
        if schema is None:
            id_type = pa.uint64() if node_id_type.lower() == "uint64" else pa.string()
            fields = [
                pa.field("source", id_type, nullable=False),
                pa.field("target", id_type, nullable=False),
            ]
            if with_relation:
                fields.append(pa.field("relation", pa.string(), nullable=True))
            if with_weight:
                fields.append(pa.field("weight", pa.float64(), nullable=False))
            if embedding_dim is not None and embedding_dim > 0:
                fields.append(pa.field("embedding", pa.list_(pa.float32(), embedding_dim), nullable=True))
            schema = pa.schema(fields)
            
        if partition_by_relation and "relation" in [f.name for f in schema]:
            spec = {
                'fields': [
                    {'name': 'relation', 'transform': 'identity', 'source_id': 1, 'field_id': 1000}
                ]
            }
            table = cls.create_partitioned(uri, schema, spec, device=device)
        else:
            table = cls.create(uri, schema, device=device)
            
        if index_endpoints:
            try:
                table.add_index("source")
                table.add_index("target")
            except Exception:
                pass
                
        if index_embedding and embedding_dim is not None and "embedding" in [f.name for f in schema]:
            try:
                table.add_index("embedding", "hnsw")
            except Exception:
                pass
                
        return table

    @classmethod
    def from_networkx(
        cls,
        uri: str,
        graph,
        device: Optional[Any] = None,
        mode: str = "append",
        index_endpoints: bool = True,
        index_embedding: bool = True,
        partition_by_relation: bool = False,
    ) -> 'Table':
        """
        Create or append to an edge table from a NetworkX graph (Graph, DiGraph, MultiGraph, MultiDiGraph).
        
        Converts NetworkX graph edges and edge attributes to columnar Arrow/Iceberg format.
        Preserves edge weights, relations, embeddings, and arbitrary edge attributes.
        Automatically enables sidecar Roaring Bitmap indexes on 'source' and 'target',
        and HNSW vector index on 'embedding' if present.
        """
        import networkx as nx
        import pandas as pd
        import pyarrow as pa
        
        uri = _resolve_uri(uri)
        df = nx.to_pandas_edgelist(graph)
        
        # Ensure standard column names
        if "source" not in df.columns and "u" in df.columns:
            df = df.rename(columns={"u": "source"})
        if "target" not in df.columns and "v" in df.columns:
            df = df.rename(columns={"v": "target"})
            
        if "weight" not in df.columns:
            df["weight"] = 1.0
        else:
            df["weight"] = df["weight"].astype(float)
            
        # Ensure relation column exists if type or key is present
        if "relation" not in df.columns:
            if "type" in df.columns:
                df["relation"] = df["type"]
            elif "key" in df.columns:
                df["relation"] = df["key"].astype(str)
            else:
                df["relation"] = "rel"

        # Check for integer vs string node IDs
        for col in ["source", "target"]:
            if not pd.api.types.is_integer_dtype(df[col]):
                try:
                    df[col] = df[col].astype("uint64")
                except (ValueError, TypeError):
                    df[col] = df[col].astype(str)
                    
        emb_col = "embedding" if "embedding" in df.columns else ("embeddings" if "embeddings" in df.columns else None)
        
        arrow_table = pa.Table.from_pandas(df, preserve_index=False)
        try:
            table = cls(uri, device=device)
            table.write(arrow_table, mode=mode)
        except Exception:
            if partition_by_relation and "relation" in df.columns:
                spec = {
                    'fields': [
                        {'name': 'relation', 'transform': 'identity', 'source_id': 1, 'field_id': 1000}
                    ]
                }
                table = cls.create_partitioned(uri, arrow_table.schema, spec, device=device)
            else:
                table = cls.create(uri, arrow_table.schema, device=device)
            table.write(arrow_table, mode="append")
            
        if index_endpoints:
            try:
                table.add_index("source")
                table.add_index("target")
            except Exception:
                pass
                
        if index_embedding and emb_col:
            try:
                table.add_index(emb_col, "hnsw")
            except Exception:
                pass
                
        return table

    def to_networkx(
        self,
        create_using=None,
        directed: bool = True,
        multigraph: bool = False,
        source: str = "source",
        target: str = "target",
        weight: str = "weight",
        edge_attr: Union[bool, List[str]] = True,
        filter_expr: Optional[str] = None,
    ):
        """
        Export this HyperStreamDB edge table to a NetworkX Graph, DiGraph, MultiGraph, or MultiDiGraph.
        
        Args:
            create_using: NetworkX graph class or instance. If None, uses nx.MultiDiGraph,
                          nx.DiGraph, nx.MultiGraph, or nx.Graph based on directed & multigraph args.
            directed: Whether the output graph is directed (default: True). Ignored if create_using is provided.
            multigraph: Whether the output graph allows multiple edges between node pairs (default: False).
            source: Source node column name (default 'source', auto-checks 'source_id' or 'u').
            target: Target node column name (default 'target', auto-checks 'target_id' or 'v').
            weight: Edge weight column name (default 'weight').
            edge_attr: List of column names to include as edge attributes, or True for all other columns.
            filter_expr: Optional SQL WHERE clause to filter edges before export (e.g. "weight > 0.5").
            
        Returns:
            networkx.Graph, networkx.DiGraph, networkx.MultiGraph, or networkx.MultiDiGraph.
        """
        import networkx as nx
        import pandas as pd
        
        sql = "SELECT * FROM t"
        if filter_expr:
            sql = f"SELECT * FROM t WHERE {filter_expr}"
            
        res = self.execute_sql(sql)
        df = res.to_pandas()
        
        actual_source = source
        if actual_source not in df.columns:
            for cand in ["source_id", "src", "u", "from"]:
                if cand in df.columns:
                    actual_source = cand
                    break
                    
        actual_target = target
        if actual_target not in df.columns:
            for cand in ["target_id", "dst", "v", "to"]:
                if cand in df.columns:
                    actual_target = cand
                    break
                    
        if actual_source not in df.columns or actual_target not in df.columns:
            raise ValueError(f"Edge table must contain source and target columns. Found columns: {list(df.columns)}")
            
        if create_using is None:
            if directed:
                create_using = nx.MultiDiGraph if multigraph else nx.DiGraph
            else:
                create_using = nx.MultiGraph if multigraph else nx.Graph
                
        if edge_attr is True:
            attrs = [col for col in df.columns if col not in (actual_source, actual_target)]
        elif isinstance(edge_attr, (list, tuple)):
            attrs = list(edge_attr)
        else:
            attrs = None
            
        return nx.from_pandas_edgelist(
            df,
            source=actual_source,
            target=actual_target,
            edge_attr=attrs,
            create_using=create_using
        )

    def define_embedding(self, column: str, function: Union[str, EmbeddingFunction], vector_column: Optional[str] = None):
        """
        Link a source column to an embedding function for automatic vectorization.
        
        Args:
            column: The source text column.
            function: Registered function name or EmbeddingFunction instance.
            vector_column: Target vector column name (defaults to {column}_vector).
        """
        self._embedding_configs[column] = {
            "function": function,
            "vector_column": vector_column or f"{column}_vector"
        }

    def write(self, data: Any, device: Optional[Any] = None, mode: str = "append"):
        """
        Write data to the table, automatically generating embeddings for configured columns.
        
        Args:
            data: pandas.DataFrame, pyarrow.Table, polars.DataFrame, numpy.ndarray, torch.Tensor, or List[Dict].
            device: Optional Device for GPU acceleration.
            mode: 'append' (default) or 'overwrite' (clears table first).
        """
        if mode == "overwrite":
            self.truncate()

        if isinstance(data, pd.DataFrame):
            return self._write_pandas(data, device=device)
        elif pa and isinstance(data, pa.Table):
            return self._write_arrow(data, device=device)
        elif pl and isinstance(data, pl.DataFrame):
            return self._write_polars(data, device=device)
        elif isinstance(data, list):
            return self._write_list(data, device=device)
        else:
            try:
                import numpy as np
                if isinstance(data, np.ndarray):
                    return self.write(pd.DataFrame(data), device=device)
                
                import torch
                if isinstance(data, torch.Tensor):
                    return self.write(pd.DataFrame(data.detach().cpu().numpy()), device=device)
            except ImportError:
                pass
            raise TypeError(f"Unsupported data type for write: {type(data)}")

    def insert(self, data: Any, device: Optional[Any] = None):
        """Alias for write() for compatibility with common vector DB APIs."""
        return self.write(data, device=device)

    def write_pandas(self, df: pd.DataFrame, device: Optional[Any] = None):
        """High-level Pandas ingestion with auto-vectorization."""
        return self._write_pandas(df, device=device)

    def write_arrow(self, table: 'pa.Table', device: Optional[Any] = None):
        """High-level Arrow ingestion with auto-vectorization."""
        return self._write_arrow(table, device=device)

    def upsert(self, data: Any, key_column: Union[str, List[str]], mode: str = "merge_on_read", device: Optional[Any] = None):
        """Update or insert data using a key column (or list of columns) to avoid duplicates."""
        from .hyperstreamdb import PyMergeMode
        
        # Map string mode to Enum
        enum_mode = PyMergeMode.MergeOnRead
        if mode.lower() == "merge_on_write":
            enum_mode = PyMergeMode.MergeOnWrite
            
        if isinstance(data, pd.DataFrame):
            processed_df = self._auto_vectorize(data)
            # If key_column is a list, join it with commas for the Rust side (or update Rust to take list)
            if isinstance(key_column, list):
                key_str = ",".join(key_column)
            else:
                key_str = key_column
            return self._inner.merge_pandas(processed_df, key_str, enum_mode, device=device)
        
        df = pd.DataFrame(data)
        return self.upsert(df, key_column, mode, device=device)

    def commit(self):
        """Commit temporary segments to the table."""
        return self._inner.commit()

    def truncate(self):
        """Clear all data from the table while keeping the schema."""
        return self._inner.truncate()

    def vacuum(self, retention_versions: int = 1):
        """
        Physically delete unreferenced data and manifest files to reclaim space.
        
        Args:
            retention_versions: Number of snapshots to keep (default 1).
        """
        return self._inner.vacuum(retention_versions)

    @property
    def autocommit(self) -> bool:
        """Get or set the autocommit state of the table."""
        return self._inner.autocommit

    @autocommit.setter
    def autocommit(self, value: bool):
        self._inner.autocommit = value

    def wait_for_background_tasks(self):
        """Wait for all background tasks (like index building) to complete."""
        return self._inner.wait_for_background_tasks()

    def delete(self, filter: str):
        """Delete rows matching the filter expression."""
        return self._inner.delete(filter)

    def _write_pandas(self, df: pd.DataFrame, device: Optional[Any] = None):
        processed_df = self._auto_vectorize(df)
        return self._inner.write_pandas(processed_df, device=device)

    def _write_arrow(self, table: 'pa.Table', device: Optional[Any] = None):
        if self._embedding_configs:
            df = table.to_pandas()
            return self._write_pandas(df, device=device)
            
        if pa and isinstance(table, pa.RecordBatch):
            from pyarrow import Table as paTable
            table = paTable.from_batches([table])
            
        return self._inner.write_arrow(table, device=device)

    def _write_polars(self, df: 'pl.DataFrame', device: Optional[Any] = None):
        if self._embedding_configs:
            pandas_df = df.to_pandas()
            return self._write_pandas(pandas_df, device=device)
        return self._inner.write_arrow(df.to_arrow(), device=device)

    def _write_list(self, data: List[Any], device: Optional[Any] = None):
        if not data:
            return
        
        # Check if first element is an Arrow object
        first = data[0]
        if (pa and isinstance(first, (pa.RecordBatch, pa.Table))):
            from pyarrow import Table as paTable
            if isinstance(first, pa.RecordBatch):
                combined = paTable.from_batches(data)
            else:
                combined = pa.concat_tables(data)
            return self._write_arrow(combined, device=device)
            
        # Default to pandas for List[Dict] or other types
        df = pd.DataFrame(data)
        return self._write_pandas(df, device=device)

    def _auto_vectorize(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]):
        if not self._embedding_configs:
            return data
            
        if isinstance(data, pd.DataFrame):
            import numpy as np
            df = data.copy()
            for col, config in self._embedding_configs.items():
                if col in df.columns:
                    func = config["function"]
                    if isinstance(func, str):
                        func = registry.get(func)
                    
                    if func:
                        vector_col = config["vector_column"]
                        embeddings = func(df[col].tolist())
                        # Enforce Float32 for vector compatibility
                        if isinstance(embeddings, np.ndarray):
                            embeddings = embeddings.astype(np.float32)
                        df[vector_col] = list(embeddings)
            return df
        
        # Large list branch omitted for brevity, logic is similar (use pandas path)
        return data

    def _prepare_vector_filter(self, vector_filter: Optional[Union[Dict[str, Any], List[float]]], **kwargs) -> Optional[Dict[str, Any]]:
        if vector_filter is None:
            return None
            
        # 1. Handle vector_filter as a list (simplified search)
        if not isinstance(vector_filter, dict):
            column = "embedding"
            if self._embedding_configs:
                column = list(self._embedding_configs.values())[0]["vector_column"]
            vector_filter = {"column": column, "query": vector_filter}
            
        # 2. Add extra kwargs (k, n_probe) to vector_filter if present
        if kwargs:
            vector_filter.update(kwargs)
            
        if "k" not in vector_filter:
            vector_filter["k"] = 10
            
        # Ensure column is set (e.g. if fluent API sent column=None)
        if vector_filter.get("column") is None:
            column = "embedding"
            if self._embedding_configs:
                column = list(self._embedding_configs.values())[0]["vector_column"]
            vector_filter["column"] = column
            
        # Auto-vectorize string query
        if "query" in vector_filter and isinstance(vector_filter["query"], str):
            # Try to find a matching embedding function
            target_col = vector_filter.get("column")
            func = None
            
            # 1. Check if we have an explicit config for this vector column
            for src_col, config in self._embedding_configs.items():
                if config["vector_column"] == target_col:
                    func = config["function"]
                    break
            
            # 2. If not, check if any registered function matches the column name
            if not func:
                func = registry.get(target_col)
            
            if func:
                if isinstance(func, str):
                    func = registry.get(func)
                if func:
                    # Vectorize the query string
                    vector_filter["query"] = func([vector_filter["query"]])[0].tolist()
                    if self.explain:
                        print(f"[Explain] Vectorized query using device: {target_col}")
                    
        return vector_filter

    def to_pandas(self, filter: Optional[str] = None, vector_filter: Optional[Union[Dict[str, Any], List[float]]] = None, columns: Optional[List[str]] = None, device: Optional[Any] = None, **kwargs):
        """
        Read table to Pandas with auto-vectorization of search queries and flexible parameters.
        
        Parameters:
            filter: Optional scalar WHERE clause (e.g., "category = 'news'")
            vector_filter: Dict with vector search params:
                - column: str (required) - vector column name
                - query: list (required) - query vector
                - k: int (required) - number of results  
                - metric: str (optional) - 'l2'|'cosine'|'innerproduct'|'l1'|'hamming'|'jaccard' (default: l2)
                - ef_search: int (optional) - HNSW ef parameter for tuning
                - probes: int (optional) - IVF probes parameter for tuning
            columns: Optional list of column names to select
            device: Optional compute device (GPU/CPU)
            **kwargs: Extra params (merged into vector_filter if present)
        
        Example::

            # Vector search with cosine metric
            df = table.to_pandas(vector_filter={
                "column": "embedding",
                "query": [1.0, 2.0, 3.0],
                "k": 5,
                "metric": "cosine",
                "ef_search": 200  # Tune HNSW search quality
            })
        """
        vf = self._prepare_vector_filter(vector_filter, **kwargs)
        if self.explain:
            # Call native Rust explain logic
            print(self._inner.explain(filter, vf))
            
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["k", "n_probe", "column"]}
        return self._inner.to_pandas(filter, vf, columns, device=device, **filtered_kwargs)

    def to_arrow(self, filter: Optional[str] = None, vector_filter: Optional[Union[Dict[str, Any], List[float]]] = None, columns: Optional[List[str]] = None, device: Optional[Any] = None, **kwargs):
        """
        Read table to Arrow Table with auto-vectorization of search queries and flexible parameters.
        
        Parameters:
            filter: Optional scalar WHERE clause (e.g., "category = 'news'")
            vector_filter: Dict with vector search params:
                - column: str (required) - vector column name
                - query: list (required) - query vector
                - k: int (required) - number of results  
                - metric: str (optional) - 'l2'|'cosine'|'innerproduct'|'l1'|'hamming'|'jaccard' (default: l2)
                - ef_search: int (optional) - HNSW ef parameter for tuning
                - probes: int (optional) - IVF probes parameter for tuning
            columns: Optional list of column names to select
            device: Optional compute device (GPU/CPU)
            **kwargs: Extra params (merged into vector_filter if present)
        """
        if "filter" in kwargs and filter is None:
            filter = kwargs.pop("filter")
            
        vf = self._prepare_vector_filter(vector_filter, **kwargs)
        # to_arrow in Rust doesn't currently take **kwargs
        return self._inner.to_arrow(filter, vf, columns, device=device)

    def sql(self, query: str) -> Any:
        """
        Execute a SQL query against the table.
        The table is registered as 't'.
        """
        return self._inner.execute_sql(query)

    def query(self) -> Query:
        """Start a fluent query."""
        return Query(self)

    def read(self, filter: Optional[str] = None, vector_filter: Optional[Union[Dict[str, Any], List[float]]] = None, columns: Optional[List[str]] = None, device: Optional[Any] = None, **kwargs):
        """
        Read table to Arrow Table (alias for to_arrow).
        """
        return self.to_arrow(filter, vector_filter, columns, device=device, **kwargs)

    def vector_search(self, column: str, query: List[float], k: int = 10, filter: Optional[str] = None, columns: Optional[List[str]] = None, device: Optional[Any] = None, **kwargs):
        """Backward compatibility alias for to_pandas with vector filter."""
        vf = {"column": column, "query": query, "k": k}
        vf.update(kwargs)
        return self.to_pandas(filter=filter, vector_filter=vf, columns=columns, device=device)

    def search(self, column: str, query: List[float], k: int = 10, filter: Optional[str] = None, columns: Optional[List[str]] = None, device: Optional[Any] = None, **kwargs):
        """Alias for vector_search."""
        return self.vector_search(column, query, k, filter, columns, device, **kwargs)

    def filter(self, expr: Optional[str] = None, vector_filter: Optional[Union[Dict[str, Any], List[float]]] = None, **kwargs) -> 'Query':
        """
        Start a fluent query or apply immediate filters.
        """
        if "filter" in kwargs and expr is None:
            expr = kwargs.pop("filter")
            
        q = Query(self, expr)
        if vector_filter is not None:
            if isinstance(vector_filter, list):
                q.vector_search(vector_filter, **kwargs)
            elif isinstance(vector_filter, dict):
                # Merge dict into Query state
                q._vector_filter = vector_filter
                if kwargs:
                    q._vector_filter.update(kwargs)
        elif kwargs:
            # Assume kwargs refer to search params if vector_filter was missing but k was provided?
            # Actually better to be explicit: table.filter(vector_filter=v, k=5)
            pass
        return q

    @property
    def primary_key(self):
        """Get the current primary key column."""
        return self._inner.get_primary_key()

    @primary_key.setter
    def primary_key(self, columns: Union[str, List[str]]):
        """Set the primary key column(s)."""
        if isinstance(columns, str):
            self._inner.set_primary_key([columns])
        else:
            self._inner.set_primary_key(list(columns))

    @property
    def index_all(self):
        """
        Whether to build HNSW/BM25 indexes for all compatible columns on commit.

        Defaults to ``False`` (opt-in). Setting this to ``True`` triggers
        background index builds after every ``commit()`` call — useful when
        you want fast ANN search but be aware of the additional commit latency
        (~15 s per 100 K rows with 768-dim vectors on CPU).

        For selective indexing, prefer ``table.add_index(column, 'hnsw')``.
        """
        return self._inner.get_index_all()

    @index_all.setter
    def index_all(self, value):
        self._inner.set_index_all(value)

    @property
    def row_count(self) -> int:
        """Get total row count in the table."""
        return self._inner.get_table_statistics().row_count

    @property
    def statistics(self):
        """Get full table statistics."""
        return self._inner.get_table_statistics()

    def add_index_columns(self, columns: List[str], tokenizer: Optional[str] = None):
        """
        Add columns to the indexing configuration.
        
        Args:
            columns: List of column names to index.
            tokenizer: Optional tokenizer name from the registry.
        """
        return self._inner.add_index_columns(columns, tokenizer)

    def set_index_config(self, column: str, enabled: bool = True, tokenizer: Optional[str] = None, device: Optional[str] = None):
        """
        Set indexing configuration for a specific column.
        (Legacy compatibility wrapper)
        """
        if not enabled:
            return self.drop_index(column)
            
        config = {"type": "hnsw"}
        if tokenizer: config["tokenizer"] = tokenizer
        if device: config["build_device"] = device
        return self.add_index(column, config)

    def set_index_columns(self, config: Dict[str, Union[str, List[Union[str, Dict[str, Any]]], Dict[str, Any]]]):
        """
        Update indexing specifications for multiple columns at once.
        Supports both simple strings and advanced configuration dictionaries.
        
        Example::

            table.set_index_columns({
                "embedding": IndexType.HNSW,
                "content": ["hnsw", "bm25"],
                "category": "bitmap"
            })
        """
        return self._inner.set_index_columns(config)

    def add_index(self, column: str, algorithm: Union[str, Dict[str, Any]] = "hnsw", **kwargs):
        """
        Add an indexing strategy to a column.
        """
        if isinstance(algorithm, str):
            algorithm = {"type": algorithm}
        
        if kwargs:
            # Map 'device' to 'build_device' for consistency with set_index_config
            if 'device' in kwargs:
                kwargs['build_device'] = kwargs.pop('device')
            algorithm.update(kwargs)
            
        return self._inner.add_index(column, algorithm)

    def drop_index(self, column: str):
        """
        Remove all indexing strategies from a column.
        """
        return self._inner.drop_index(column)

    def add_primary_key(self, column: str):
        """
        Atomically add a column to the primary key.
        This performs a validation check for duplicates across all existing data.
        If validation fails, the change is NOT committed.
        """
        return self._inner.add_primary_key(column)

    def drop_primary_key(self, column: str):
        """
        Atomically remove a column from the primary key.
        """
        return self._inner.drop_primary_key(column)

    def add_column(self, name: str, data_type: Union[str, Any]):
        """
        Add a new column to the table.
        data_type can be a string (e.g., 'int32', 'float64', 'string') or a pyarrow.DataType.
        """
        import pyarrow as pa
        if isinstance(data_type, str):
            # Helper to parse common string types
            dt_map = {
                'int8': pa.int8(), 'int16': pa.int16(), 'int32': pa.int32(), 'int64': pa.int64(),
                'uint8': pa.uint8(), 'uint16': pa.uint16(), 'uint32': pa.uint32(), 'uint64': pa.uint64(),
                'float16': pa.float16(), 'float32': pa.float32(), 'float64': pa.float64(),
                'string': pa.string(), 'utf8': pa.string(), 'large_string': pa.large_string(),
                'bool': pa.bool_(), 'boolean': pa.bool_(),
                'date32': pa.date32(), 'date64': pa.date64(),
            }
            if data_type.lower() in dt_map:
                data_type = dt_map[data_type.lower()]
            else:
                raise ValueError(f"Unsupported or unrecognized string data_type: {data_type}. Please pass a pyarrow.DataType explicitly.")
                
        if isinstance(data_type, pa.DataType):
            # Create a single-field schema to pass the type down safely via FFI
            schema = pa.schema([(name, data_type)])
            return self._inner.add_column(name, schema)
        else:
            raise TypeError("data_type must be a string or pyarrow.DataType")

    def drop_column(self, name: str):
        """Drop an existing column from the table."""
        return self._inner.drop_column(name)

    def rename_column(self, old_name: str, new_name: str):
        """Rename an existing column."""
        return self._inner.rename_column(old_name, new_name)

    def update_column_type(self, name: str, new_type: str):
        """
        Update the type of a column (Type Promotion).
        Example: update_column_type('id', 'long')
        """
        return self._inner.update_column_type(name, new_type)

    def move_column(self, name: str, new_index: int):
        """Move a column to a new 0-based index position."""
        return self._inner.move_column(name, new_index)

    def set_sort_order(self, columns: List[str], ascending: List[bool]):
        """Set the table's default sort order for future data writes."""
        return self._inner.replace_sort_order(columns, ascending)

    def set_partition_spec(self, spec: List[Dict[str, Any]]):
        """
        Update the table's partition specification.
        
        Args:
            spec: List of partition fields, each being a dict with:
                - source_id: int (or source_ids: List[int])
                - name: str
                - transform: str
                - field_id: int (optional)
        """
        from .hyperstreamdb import PartitionField
        
        fields = []
        for item in spec:
            if isinstance(item, dict):
                # Handle both 'source_id' (singular) and 'source_ids' (plural) for flexibility
                source_ids = item.get("source_ids")
                if source_ids is None:
                    sid = item.get("source_id")
                    source_ids = [sid] if sid is not None else []
                
                fields.append(PartitionField(
                    source_ids=source_ids,
                    name=item["name"],
                    transform=item["transform"],
                    field_id=item.get("field_id")
                ))
            else:
                fields.append(item)
                
        return self._inner.update_spec(fields)

    def __getattr__(self, name):
        """Delegate other calls to the Rust implementation."""
        return getattr(self._inner, name)

    def __repr__(self):
        return f"HyperStreamTable(uri={self._inner.table_uri()})"

class Session:
    """
    HyperStreamDB Query Session with integration for Python Table objects.
    """
    def __init__(self, memory_mb: Optional[int] = None):
        self._inner = _RustSession(memory_mb)

    def register(self, name: str, table: Union[Table, _RustTable]):
        """Register a table in the session for SQL queries."""
        if hasattr(table, "_inner"):
            # Unwrap Python Table to get the Rust implementation
            return self._inner.register(name, table._inner)
        return self._inner.register(name, table)

    def sql(self, query: str) -> Any:
        """Execute a SQL query against the table (registered as 't')."""
        return self._inner.sql(query)
