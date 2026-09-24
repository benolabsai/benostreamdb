import os
import argparse
import pandas as pd
import benostreamdb
import shutil

# Dumps (full wiki parquets) live on the 14 TB HDD by default; override with
# --dumps-dir or BENOSTREAM_DATA.
DEFAULT_DUMPS = os.environ.get(
    "BENOSTREAM_DATA",
    os.path.join(os.path.expanduser("~"), "data", "benostreamdb"),
)

def main():
    parser = argparse.ArgumentParser(description="Prune the wiki graph and export for GitHub releases")
    parser.add_argument("--dumps-dir", type=str, default=DEFAULT_DUMPS,
                        help="directory holding the wiki parquets (default: $HOME/data/benostreamdb)")
    parser.add_argument("--nodes", type=str, default=None, help="default: <dumps-dir>/nodes_with_embeddings.parquet")
    parser.add_argument("--edges", type=str, default=None, help="default: <dumps-dir>/edges.parquet")
    parser.add_argument("--out_nodes", type=str, default=None, help="default: <dumps-dir>/demo_nodes.parquet")
    parser.add_argument("--out_edges", type=str, default=None, help="default: <dumps-dir>/demo_edges.parquet")
    args = parser.parse_args()

    dumps = os.path.abspath(os.path.expanduser(args.dumps_dir))
    args.nodes = args.nodes or os.path.join(dumps, "nodes_with_embeddings.parquet")
    args.edges = args.edges or os.path.join(dumps, "edges.parquet")
    args.out_nodes = args.out_nodes or os.path.join(dumps, "demo_nodes.parquet")
    args.out_edges = args.out_edges or os.path.join(dumps, "demo_edges.parquet")

    print(f"Loading {args.nodes} and {args.edges}...")
    nodes_df = pd.read_parquet(args.nodes)
    edges_df = pd.read_parquet(args.edges)
    
    # Ensure id/source/target are strings
    nodes_df['id'] = nodes_df['id'].astype(str)
    edges_df['source'] = edges_df['source'].astype(str)
    edges_df['target'] = edges_df['target'].astype(str)

    # Initialize a temporary BenoStreamDB to run connected components
    tmp_uri = "file:///tmp/benostreamdb_export_tmp"
    if os.path.exists("/tmp/benostreamdb_export_tmp"):
        shutil.rmtree("/tmp/benostreamdb_export_tmp")

    print("Loading into BenoStreamDB for graph algorithms...")
    table = benostreamdb.Table(tmp_uri)
    table.add_index("source", {"type": "graph", "src_column": "source", "dst_column": "target"})
    table.insert(edges_df.to_dict('records'))
    table.commit()
    table.wait_for_background_tasks()

    print("Computing connected components...")
    cc_result = table.connected_components()
    cc_df = cc_result.to_pandas()

    print("Finding the largest component...")
    component_counts = cc_df['component'].value_counts()
    largest_component_id = component_counts.idxmax()
    largest_size = component_counts.max()
    print(f"Largest component is {largest_component_id} with {largest_size} edges/nodes.")

    valid_nodes = cc_df[cc_df['component'] == largest_component_id]['node'].values
    
    print(f"Filtering nodes... (original: {len(nodes_df)})")
    pruned_nodes_df = nodes_df[nodes_df['id'].isin(valid_nodes)].copy()
    print(f"Pruned nodes: {len(pruned_nodes_df)}")
    
    print(f"Filtering edges... (original: {len(edges_df)})")
    pruned_edges_df = edges_df[edges_df['source'].isin(valid_nodes) & edges_df['target'].isin(valid_nodes)].copy()
    print(f"Pruned edges: {len(pruned_edges_df)}")

    print(f"Saving pruned datasets to {args.out_nodes} and {args.out_edges}...")
    pruned_nodes_df.to_parquet(args.out_nodes)
    pruned_edges_df.to_parquet(args.out_edges)

    # Clean up
    shutil.rmtree("/tmp/benostreamdb_export_tmp")
    print("Done! Datasets are ready for GitHub Releases.")

if __name__ == "__main__":
    main()
