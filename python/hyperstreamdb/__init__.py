from .hyperstreamdb import *
from .hyperstreamdb import Table as _RustTable
from .embeddings import registry, EmbeddingFunction
import pandas as pd
from typing import List, Optional, Union, Dict, Any
import os

def _resolve_uri(uri: str) -> str:
    """Resolve a URI to an absolute path if it's a local relative path."""
    if "://" not in uri and not uri.startswith("/"):
        return os.path.abspath(uri)
    return uri

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

    def vector_search(self, vector: Union[List[float], str], column: Optional[str] = None, k: int = 10, **kwargs) -> 'Query':
        """
        Apply a vector search filter.
        
        Args:
            vector: The query vector (list of floats) or a string to be vectorized.
            column: The vector column to search against.
            k: Number of nearest neighbors to return.
            **kwargs: Additional parameters (e.g., n_probe).
        """
        self._vector_filter = {
            "column": column,
            "vector": vector,
            "k": k,
            **kwargs
        }
        return self

    def select(self, columns: List[str]) -> 'Query':
        """Select specific columns to return."""
        self._columns = columns
        return self

    def to_pandas(self, context: Optional[Any] = None):
        """Execute the query and return results as a Pandas DataFrame."""
        return self._table.to_pandas(
            filter=self._filter, 
            vector_filter=self._vector_filter, 
            columns=self._columns, 
            context=context
        )

    def to_arrow(self, context: Optional[Any] = None):
        """Execute the query and return results as an Arrow Table."""
        return self._table.to_arrow(
            filter=self._filter, 
            vector_filter=self._vector_filter, 
            columns=self._columns, 
            context=context
        )

    def execute(self, context: Optional[Any] = None):
        """Execute the query and return results as a Pandas DataFrame (alias for to_pandas)."""
        return self.to_pandas(context)

class Table:
    """
    Enhanced HyperStreamDB Table with auto-vectorization and embedding registry support.
    """
    def __init__(self, uri: str, inner_table: Optional[_RustTable] = None, context: Optional[Any] = None):
        uri = _resolve_uri(uri)
        if inner_table:
            self._inner = inner_table
        else:
            self._inner = _RustTable(uri, context=context)
        self._embedding_configs = {}

    @classmethod
    def create(cls, uri: str, schema, context: Optional[Any] = None) -> 'Table':
        """Create a new table with an explicit schema."""
        uri = _resolve_uri(uri)
        return cls(uri, inner_table=_RustTable.create(uri, schema, context=context))

    @classmethod
    def register_external(cls, uri: str, iceberg_metadata_uri: str, context: Optional[Any] = None) -> 'Table':
        """Register an existing Iceberg table."""
        uri = _resolve_uri(uri)
        return cls(uri, inner_table=_RustTable.register_external(uri, iceberg_metadata_uri, context=context))

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

    def write(self, data: Union[pd.DataFrame, List[Dict[str, Any]]], context: Optional[Any] = None):
        """
        Write data to the table, automatically generating embeddings for configured columns.
        """
        processed_data = self._auto_vectorize(data)
        return self._inner.write(processed_data, context=context)

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

    def _prepare_vector_filter(self, vector_filter: Optional[Union[Dict[str, Any], List[float]]], **kwargs) -> Optional[Dict[str, Any]]:
        if vector_filter is None:
            return None
            
        # 1. Handle vector_filter as a list (simplified search)
        if not isinstance(vector_filter, dict):
            column = "embedding"
            if self._embedding_configs:
                column = list(self._embedding_configs.values())[0]["vector_column"]
            vector_filter = {"column": column, "vector": vector_filter}
            
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
        if "vector" in vector_filter and isinstance(vector_filter["vector"], str):
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
                    
        return vector_filter

    def to_pandas(self, filter: Optional[str] = None, vector_filter: Optional[Union[Dict[str, Any], List[float]]] = None, columns: Optional[List[str]] = None, context: Optional[Any] = None, **kwargs):
        """
        Read table to Pandas with auto-vectorization of search queries and flexible parameters.
        """
        if "filter" in kwargs and filter is None:
            filter = kwargs.pop("filter")
            
        vf = self._prepare_vector_filter(vector_filter, **kwargs)
        return self._inner.to_pandas(filter, vf, columns, context=context)

    def to_arrow(self, filter: Optional[str] = None, vector_filter: Optional[Union[Dict[str, Any], List[float]]] = None, columns: Optional[List[str]] = None, context: Optional[Any] = None, **kwargs):
        """
        Read table to Arrow Table with auto-vectorization of search queries and flexible parameters.
        """
        if "filter" in kwargs and filter is None:
            filter = kwargs.pop("filter")
            
        vf = self._prepare_vector_filter(vector_filter, **kwargs)
        return self._inner.to_arrow(filter, vf, columns, context=context)

    def query(self) -> Query:
        """Start a fluent query."""
        return Query(self)

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

    def __getattr__(self, name):
        """Delegate other calls to the Rust implementation."""
        return getattr(self._inner, name)

    def __repr__(self):
        return f"HyperStreamTable(uri={self._inner.table_uri()})"
