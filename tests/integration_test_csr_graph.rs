// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use hyperstreamdb::core::index::csr_graph::MmapCsrGraph;
use hyperstreamdb::core::sql::graph_udf::drift_search::DriftGraph;
use std::fs::File;
use std::io::Write;

#[tokio::test]
async fn test_csr_graph_out_of_core() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let offsets_path = temp_dir.path().join("test.graph.csr.offsets");
    let edges_path = temp_dir.path().join("test.graph.csr.edges");

    // Create a simple graph:
    // 0 -> 1, 2
    // 1 -> 2
    // 2 -> 0, 1, 3
    // 3 -> 0
    let num_nodes = 4;
    let offsets: Vec<u64> = vec![0, 2, 3, 6, 7];

    use hyperstreamdb::core::index::csr_graph::CsrEdge;
    let edges: Vec<CsrEdge> = vec![
        CsrEdge {
            dst_id: 1,
            row_id: 10,
        },
        CsrEdge {
            dst_id: 2,
            row_id: 11,
        },
        CsrEdge {
            dst_id: 2,
            row_id: 20,
        },
        CsrEdge {
            dst_id: 0,
            row_id: 30,
        },
        CsrEdge {
            dst_id: 1,
            row_id: 31,
        },
        CsrEdge {
            dst_id: 3,
            row_id: 32,
        },
        CsrEdge {
            dst_id: 0,
            row_id: 40,
        },
    ];

    // Write offsets manually
    {
        let mut f = File::create(&offsets_path)?;
        for o in offsets {
            f.write_all(bytemuck::bytes_of(&o))?;
        }
    }

    // Write edges manually
    {
        let mut f = File::create(&edges_path)?;
        for e in edges {
            f.write_all(bytemuck::bytes_of(&e))?;
        }
    }

    // Create dummy dict
    let dict_path = temp_dir.path().join("graph.dict");
    {
        let mut f = File::create(&dict_path)?;
        let dict: Vec<u64> = vec![0, 1, 2, 3];
        for d in dict {
            f.write_all(bytemuck::bytes_of(&d))?;
        }
    }

    // Load out-of-core
    let graph = MmapCsrGraph::load(&offsets_path, &edges_path, &dict_path)?;

    assert_eq!(graph.num_nodes, num_nodes);
    assert_eq!(graph.num_edges, 7);

    // Test DriftGraph trait methods
    assert_eq!(graph.get_degree(0), 2);
    assert_eq!(graph.get_degree(1), 1);
    assert_eq!(graph.get_degree(2), 3);
    assert_eq!(graph.get_degree(3), 1);
    assert_eq!(graph.get_degree(4), 0); // Out of bounds

    assert_eq!(graph.get_neighbors(0), vec![1, 2]);
    assert_eq!(graph.get_neighbors(1), vec![2]);
    assert_eq!(graph.get_neighbors(2), vec![0, 1, 3]);
    assert_eq!(graph.get_neighbors(3), vec![0]);
    assert!(graph.get_neighbors(4).is_empty());

    Ok(())
}
