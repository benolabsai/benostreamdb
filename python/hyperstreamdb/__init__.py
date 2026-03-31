from .hyperstreamdb import *
from .hyperstreamdb import Table as _RustTable
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
    def __init__(self, uri: str, inner_table: Optional[_RustTable] = None, context: Optional[Any] = None, index_all: bool = True, primary_key: Optional[str] = None):
        uri = _resolve_uri(uri)
        if inner_table:
            self._inner = inner_table
        else:
            self._inner = _RustTable(uri, context=context)
        self._inner.set_index_all(index_all)
        if primary_key:
            if isinstance(primary_key, str):
                self._inner.set_primary_key([primary_key])
            else:
                self._inner.set_primary_key(list(primary_key))
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

    def write(self, data: Any, context: Optional[Any] = None, mode: str = "append"):
        """
        Write data to the table, automatically generating embeddings for configured columns.
        
        Args:
            data: pandas.DataFrame, pyarrow.Table, polars.DataFrame, or List[Dict].
            context: Optional ComputeContext for GPU acceleration.
            mode: 'append' (default) or 'overwrite' (clears table first).
        """
        if mode == "overwrite":
            self.truncate()

        if isinstance(data, pd.DataFrame):
            return self._write_pandas(data, context=context)
        elif pa and isinstance(data, pa.Table):
            return self._write_arrow(data, context=context)
        elif pl and isinstance(data, pl.DataFrame):
            return self._write_polars(data, context=context)
        elif isinstance(data, list):
            return self._write_list(data, context=context)
        else:
            try:
                import numpy as np
                if isinstance(data, np.ndarray):
                    return self.write(pd.DataFrame(data), context=context)
                
                import torch
                if isinstance(data, torch.Tensor):
                    return self.write(pd.DataFrame(data.detach().cpu().numpy()), context=context)
            except ImportError:
                pass
            raise TypeError(f"Unsupported data type for write: {type(data)}")

    def write_pandas(self, df: pd.DataFrame, context: Optional[Any] = None):
        """High-level Pandas ingestion with auto-vectorization."""
        return self._write_pandas(df, context=context)

    def write_arrow(self, table: 'pa.Table', context: Optional[Any] = None):
        """High-level Arrow ingestion with auto-vectorization."""
        return self._write_arrow(table, context=context)

    def upsert(self, data: Any, key_column: Union[str, List[str]], mode: str = "merge_on_read", context: Optional[Any] = None):
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
            return self._inner.merge_pandas(processed_df, key_str, enum_mode, context=context)
        
        df = pd.DataFrame(data)
        return self.upsert(df, key_column, mode, context=context)

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

    def _write_pandas(self, df: pd.DataFrame, context: Optional[Any] = None):
        processed_df = self._auto_vectorize(df)
        return self._inner.write_pandas(processed_df, context=context)

    def _write_arrow(self, table: 'pa.Table', context: Optional[Any] = None):
        if self._embedding_configs:
            df = table.to_pandas()
            return self._write_pandas(df, context=context)
        return self._inner.write_arrow(table, context=context)

    def _write_polars(self, df: 'pl.DataFrame', context: Optional[Any] = None):
        if self._embedding_configs:
            pandas_df = df.to_pandas()
            return self._write_pandas(pandas_df, context=context)
        return self._inner.write_arrow(df.to_arrow(), context=context)

    def _write_list(self, data: List[Dict[str, Any]], context: Optional[Any] = None):
        # Convert to pandas first to handle vectorization and type enforcement
        df = pd.DataFrame(data)
        return self._write_pandas(df, context=context)

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
                    print(f"DEBUG: Vectorized query to {vector_filter['query']}")
                    
        return vector_filter

    def to_pandas(self, filter: Optional[str] = None, vector_filter: Optional[Union[Dict[str, Any], List[float]]] = None, columns: Optional[List[str]] = None, context: Optional[Any] = None, **kwargs):
        """
        Read table to Pandas with auto-vectorization of search queries and flexible parameters.
        """
        if "filter" in kwargs and filter is None:
            filter = kwargs.pop("filter")
            
        vf = self._prepare_vector_filter(vector_filter, **kwargs)
        # Filter out HyperStreamDB-specific kwargs before passing to inner to_pandas
        # which passes them to pyarrow.Table.to_pandas
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["k", "n_probe", "column"]}
        return self._inner.to_pandas(filter, vf, columns, context=context, **filtered_kwargs)

    def to_arrow(self, filter: Optional[str] = None, vector_filter: Optional[Union[Dict[str, Any], List[float]]] = None, columns: Optional[List[str]] = None, context: Optional[Any] = None, **kwargs):
        """
        Read table to Arrow Table with auto-vectorization of search queries and flexible parameters.
        """
        if "filter" in kwargs and filter is None:
            filter = kwargs.pop("filter")
            
        vf = self._prepare_vector_filter(vector_filter, **kwargs)
        # to_arrow in Rust doesn't currently take **kwargs
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
        """Whether to index all compatible columns by default."""
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
        
        Args:
            column: Name of the column to configure.
            enabled: Whether to enable indexing for this column (default: True).
            tokenizer: Tokenizer name from the registry ('identity', 'whitespace', 'standard').
            device: Compute device ('cpu', 'cuda', 'mps') if specific processing is needed.
        """
        self._inner.set_index_config(column, enabled, tokenizer, device)

    def __getattr__(self, name):
        """Delegate other calls to the Rust implementation."""
        return getattr(self._inner, name)

    def __repr__(self):
        return f"HyperStreamTable(uri={self._inner.table_uri()})"
