# BenoStreamDB Workflow Tutorials

One runnable tutorial per core workflow. The **Python** versions are Jupyter
notebooks (open them in JupyterLab); the **SQL / connector** versions are
markdown pages under [`docs/tutorials/`](../../docs/tutorials/).

| Workflow | Python notebook | SQL / connector guide |
| :--- | :--- | :--- |
| AI / semantic retrieval via MCP | [`01_ai_retrieval_mcp.ipynb`](01_ai_retrieval_mcp.ipynb) | [`docs/tutorials/01_ai_retrieval_mcp.md`](../../docs/tutorials/01_ai_retrieval_mcp.md) |
| JSON / document retrieval with a JSON-path index | [`02_json_document_retrieval.ipynb`](02_json_document_retrieval.ipynb) | [`docs/tutorials/02_json_document_retrieval.md`](../../docs/tutorials/02_json_document_retrieval.md) |
| Graph traversal via `graph_neighbors(...)` | [`03_graph_traversal.ipynb`](03_graph_traversal.ipynb) | [`docs/tutorials/03_graph_traversal.md`](../../docs/tutorials/03_graph_traversal.md) |

## Running the notebooks

```bash
pip install jupyterlab pyarrow
pip install benostreamdb            # or: maturin develop --release
jupyter lab examples/tutorials
```

Each notebook is self-contained (no external datasets or model downloads) and
writes to `/tmp/bsdb_tutorial*`.

## Kernels

- **Python** — first-class: the `benostreamdb` extension is a normal Python
  module, so the standard IPython kernel works.
- **Rust** — use the community [`evcxr_jupyter`](https://github.com/evcxr/evcxr)
  kernel and `:dep benostreamdb` to drive the core API.
- **Scala** — use the [Almond](https://almond.sh/) kernel with the Spark
  connector on the classpath.

Rust and Scala kernels are third-party and not shipped in this repository; the
markdown guides cover their usage.
