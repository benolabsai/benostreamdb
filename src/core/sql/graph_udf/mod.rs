// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

pub mod all_shortest_paths;
pub mod betweenness_centrality;
pub mod closeness_centrality;
pub mod connected_components;
pub mod degree_centrality;
pub mod drift_search;
pub mod graph_neighbors;
pub mod graph_view;
pub mod jaccard_coefficient;
pub mod label_propagation;
pub mod leiden_communities;
pub mod louvain_communities;
pub mod modularity;
pub mod pagerank;
pub mod personalized_pagerank;
pub mod preferential_attachment;
pub mod shortest_path;
pub mod strongly_connected_components;
pub mod subgraph;
pub mod triangle_count;
use datafusion::logical_expr::AggregateUDF;

pub use all_shortest_paths::AllShortestPathsUDF;
pub use betweenness_centrality::BetweennessCentralityUDF;
pub use closeness_centrality::ClosenessCentralityUDF;
pub use connected_components::ConnectedComponentsUDF;
pub use degree_centrality::DegreeCentralityUDF;
pub use drift_search::{DriftSearchUDF, RegionalDriftUDF};
pub use graph_neighbors::GraphNeighborsUDF;
pub use graph_view::{
    CachingGraph, GraphAccumulatorBase, GraphMode, GraphView, SimpleGraph, SubgraphView,
};
pub use jaccard_coefficient::JaccardCoefficientUDF;
pub use label_propagation::LabelPropagationUDF;
pub use leiden_communities::LeidenCommunitiesUDF;
pub use louvain_communities::LouvainCommunitiesUDF;
pub use modularity::ModularityUDF;
pub use pagerank::PageRankUDF;
pub use personalized_pagerank::PersonalizedPageRankUDF;
pub use preferential_attachment::PreferentialAttachmentUDF;
pub use shortest_path::ShortestPathUDF;
pub use strongly_connected_components::StronglyConnectedComponentsUDF;
pub use subgraph::SubgraphUDF;
pub use triangle_count::TriangleCountUDF;

/// Returns a list of all custom Graph UDAFs to be registered in DataFusion
pub fn all_graph_aggregates() -> Vec<AggregateUDF> {
    vec![
        AggregateUDF::new_from_impl(label_propagation::LabelPropagationUDF::new()),
        AggregateUDF::new_from_impl(graph_neighbors::GraphNeighborsUDF::new()),
        AggregateUDF::new_from_impl(subgraph::SubgraphUDF::new()),
        AggregateUDF::new_from_impl(subgraph::ConnectingPathsUDF::new()),
        AggregateUDF::new_from_impl(shortest_path::ShortestPathUDF::new()),
        AggregateUDF::new_from_impl(drift_search::DriftSearchUDF::new()),
        AggregateUDF::new_from_impl(drift_search::RegionalDriftUDF::new()),
        AggregateUDF::new_from_impl(connected_components::ConnectedComponentsUDF::new()),
        AggregateUDF::new_from_impl(degree_centrality::DegreeCentralityUDF::new()),
        AggregateUDF::new_from_impl(jaccard_coefficient::JaccardCoefficientUDF::new()),
        AggregateUDF::new_from_impl(louvain_communities::LouvainCommunitiesUDF::new()),
        AggregateUDF::new_from_impl(leiden_communities::LeidenCommunitiesUDF::new()),
        AggregateUDF::new_from_impl(modularity::ModularityUDF::new()),
        AggregateUDF::new_from_impl(pagerank::PageRankUDF::new()),
        AggregateUDF::new_from_impl(personalized_pagerank::PersonalizedPageRankUDF::new()),
        AggregateUDF::new_from_impl(
            strongly_connected_components::StronglyConnectedComponentsUDF::new(),
        ),
        AggregateUDF::new_from_impl(preferential_attachment::PreferentialAttachmentUDF::new()),
        AggregateUDF::new_from_impl(triangle_count::TriangleCountUDF::new()),
        AggregateUDF::new_from_impl(all_shortest_paths::AllShortestPathsUDF::new()),
        AggregateUDF::new_from_impl(closeness_centrality::ClosenessCentralityUDF::new()),
        AggregateUDF::new_from_impl(betweenness_centrality::BetweennessCentralityUDF::new()),
    ]
}
