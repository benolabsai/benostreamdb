// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod connected_components;
pub mod connecting_paths;
pub mod degree_centrality;
pub mod louvain_communities;
pub mod modularity;
pub mod neighbors;
pub mod pagerank;
pub mod personalized_pagerank;
pub mod shortest_path;
pub mod subgraph;
pub mod to_graphviz;
use datafusion::logical_expr::AggregateUDF;
pub mod adamic_adar;
pub mod clustering_coefficient;
pub mod jaccard_coefficient;
pub mod label_propagation;
pub mod preferential_attachment;
pub mod resource_allocation;
pub mod strongly_connected_components;
pub mod topological_sort;

pub use adamic_adar::AdamicAdarUDF;
pub use clustering_coefficient::ClusteringCoefficientUDF;
pub use connected_components::ConnectedComponentsUDF;
pub use connecting_paths::ConnectingPathsUDF;
pub use degree_centrality::DegreeCentralityUDF;
pub use jaccard_coefficient::JaccardCoefficientUDF;
pub use label_propagation::LabelPropagationUDF;
pub use louvain_communities::LouvainCommunitiesUDF;
pub use modularity::ModularityUDF;
pub use neighbors::GraphNeighborsUDF;
pub use pagerank::PageRankUDF;
pub use personalized_pagerank::PersonalizedPageRankUDF;
pub use preferential_attachment::PreferentialAttachmentUDF;
pub use resource_allocation::ResourceAllocationUDF;
pub use shortest_path::ShortestPathUDF;
pub use strongly_connected_components::StronglyConnectedComponentsUDF;
pub use subgraph::SubgraphUDF;
pub use to_graphviz::ToGraphvizUDF;
pub use topological_sort::TopologicalSortUDF;

/// Returns a list of all custom Graph UDAFs to be registered in DataFusion
pub fn all_graph_aggregates() -> Vec<AggregateUDF> {
    vec![
        AggregateUDF::new_from_impl(pagerank::PageRankUDF::new()),
        AggregateUDF::new_from_impl(personalized_pagerank::PersonalizedPageRankUDF::new()),
        AggregateUDF::new_from_impl(shortest_path::ShortestPathUDF::new()),
        AggregateUDF::new_from_impl(connected_components::ConnectedComponentsUDF::new()),
        AggregateUDF::new_from_impl(neighbors::GraphNeighborsUDF::new()),
        AggregateUDF::new_from_impl(subgraph::SubgraphUDF::new()),
        AggregateUDF::new_from_impl(connecting_paths::ConnectingPathsUDF::new()),
        AggregateUDF::new_from_impl(degree_centrality::DegreeCentralityUDF::new()),
        AggregateUDF::new_from_impl(to_graphviz::ToGraphvizUDF::new()),
        AggregateUDF::new_from_impl(louvain_communities::LouvainCommunitiesUDF::new()),
        AggregateUDF::new_from_impl(modularity::ModularityUDF::new()),
        AggregateUDF::new_from_impl(label_propagation::LabelPropagationUDF::new()),
        AggregateUDF::new_from_impl(label_propagation::LabelPropagationUDF::new_alias()),
        AggregateUDF::new_from_impl(adamic_adar::AdamicAdarUDF::new()),
        AggregateUDF::new_from_impl(preferential_attachment::PreferentialAttachmentUDF::new()),
        AggregateUDF::new_from_impl(jaccard_coefficient::JaccardCoefficientUDF::new()),
        AggregateUDF::new_from_impl(resource_allocation::ResourceAllocationUDF::new()),
        AggregateUDF::new_from_impl(clustering_coefficient::ClusteringCoefficientUDF::new()),
        AggregateUDF::new_from_impl(
            strongly_connected_components::StronglyConnectedComponentsUDF::new(),
        ),
        AggregateUDF::new_from_impl(topological_sort::TopologicalSortUDF::new()),
    ]
}
