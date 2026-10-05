import numpy as np
import cudf
import cugraph
import networkx as nx

rng = np.random.default_rng(42)
n = 10000
edges = np.stack(
    [rng.integers(0, n, 50000), rng.integers(0, n, 50000)], axis=1
).astype(np.int64)

# cuGraph
gdf = cudf.DataFrame({"src": edges[:, 0], "dst": edges[:, 1]})
G = cugraph.Graph(directed=True)
G.from_cudf_edgelist(gdf, source="src", destination="dst")
cg = cugraph.pagerank(G, alpha=0.85, max_iter=30).to_pandas().set_index("vertex")["pagerank"]

# NetworkX reference
Gx = nx.DiGraph()
Gx.add_nodes_from(range(n))
Gx.add_edges_from(edges.tolist())
nxp = nx.pagerank(Gx, alpha=0.85, max_iter=30)

a = cg.reindex(range(n)).fillna(0.0).to_numpy()
b = np.array([nxp.get(i, 0.0) for i in range(n)])
print("max abs diff:", float(np.abs(a - b).max()))
print("corr:", float(np.corrcoef(a, b)[0, 1]))
print("cugraph top5:", list(cg.sort_values(ascending=False).head(5).index))
print("networkx top5:", sorted(nxp, key=nxp.get, reverse=True)[:5])
