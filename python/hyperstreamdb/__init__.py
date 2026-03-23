from .hyperstreamdb import *
from .hyperstreamdb import Table as _RustTable
from .embeddings import registry, EmbeddingFunction
import pandas as pd
from typing import List, Optional, Union, Dict, Any

class Table:
    """
    Enhanced HyperStreamDB Table with auto-vectorization and embedding registry support.
    """
    def __init__(self, uri: str, inner_table: Optional[_RustTable] = None):
        if inner_table:
            self._inner = inner_table
        else:
            self._inner = _RustTable(uri)
        self._embedding_configs = {}

    @classmethod
    def create(cls, uri: str, schema) -> 'Table':
        """Create a new table with an explicit schema."""
        return cls(uri, inner_table=_RustTable.create(uri, schema))

    @classmethod
    def register_external(cls, uri: str, iceberg_metadata_uri: str) -> 'Table':
        """Register an existing Iceberg table."""
        return cls(uri, inner_table=_RustTable.register_external(uri, iceberg_metadata_uri))

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

    def write(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]):
        """
        Write data to the table, automatically generating embeddings for configured columns.
        """
        processed_data = self._auto_vectorize(data)
        return self._inner.write(processed_data)

    def _auto_vectorize(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]):
        if not self._embedding_configs:
            return data
            
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            for col, config in self._embedding_configs.items():
                if col in df.columns:
                    func = config["function"]
                    if isinstance(func, str):
                        func = registry.get(func)
                    
                    if func:
                        vector_col = config["vector_column"]
                        # Generate embeddings for the entire column
                        embeddings = func(df[col].tolist())
                        # Convert to list of lists for Arrow/Pandas compatibility
                        df[vector_col] = embeddings.tolist()
            return df
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Batch vectorization for performance
            for col, config in self._embedding_configs.items():
                texts = [item.get(col) for item in data if col in item]
                if texts:
                    func = config["function"]
                    if isinstance(func, str):
                        func = registry.get(func)
                    if func:
                        vector_col = config["vector_column"]
                        embeddings = func(texts)
                        # Distribute back to dicts
                        emb_list = embeddings.tolist()
                        idx = 0
                        for item in data:
                            if col in item:
                                item[vector_col] = emb_list[idx]
                                idx += 1
            return data
            
        return data

    def to_pandas(self, filter: Optional[str] = None, vector_filter: Optional[Dict[str, Any]] = None, columns: Optional[List[str]] = None):
        """
        Read table to Pandas with auto-vectorization of search queries.
        """
        if vector_filter and "vector" in vector_filter and isinstance(vector_filter["vector"], str):
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
                    vector_filter["vector"] = func([vector_filter["vector"]])[0].tolist()
        
        return self._inner.to_pandas(filter, vector_filter, columns)

    def __getattr__(self, name):
        """Delegate other calls to the Rust implementation."""
        return getattr(self._inner, name)

    def __repr__(self):
        return f"HyperStreamTable(uri={self._inner.table_uri()})"
