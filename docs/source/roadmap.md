# Project Roadmap

For the latest status on implementation and planned features, please refer to the [ROADMAP.md](https://github.com/rla3rd/hyperstreamdb/blob/main/ROADMAP.md) in the GitHub repository.

## Future Phases

### Multi-Column Indexes
We are currently researching the implementation of **Composite Scalar Indexes** (pre-computed roaring bitmaps for common multi-column filters) and **Multi-Vector Search** (simultaneous ranking across multiple embedding columns).

### Distributed Query Engine
Future versions of HyperStreamDB will include a distributed query planner capable of coordinating multiple serverless execution nodes for ultra-large datasets.

### Native Spark and Trino Connectors
Native integration for distributed SQL engines to leverage HyperStreamDB sidecar indexing.
