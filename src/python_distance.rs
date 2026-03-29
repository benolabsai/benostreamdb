// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Python bindings for vector distance functions
/// 
/// This module provides Python bindings for all 6 distance metrics with GPU acceleration support.
/// It uses PyO3 for zero-copy NumPy integration and supports batch operations.

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use crate::core::index::{VectorMetric, distance};
use crate::core::index::gpu::{compute_distance, compute_distance_batch, ComputeBackend};
use crate::python_gpu_context::PyComputeContext;

/// Helper function to validate vector dimensions
fn validate_dimensions(a_len: usize, b_len: usize) -> PyResult<()> {
    if a_len != b_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Vector dimension mismatch: a has dimension {}, b has dimension {}", a_len, b_len)
        ));
    }
    Ok(())
}

/// Helper function to validate for NaN/inf values
fn validate_values(values: &[f32], name: &str) -> PyResult<()> {
    for (i, &val) in values.iter().enumerate() {
        if val.is_nan() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid input: {} contains NaN at index {}", name, i)
            ));
        }
        if val.is_infinite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid input: {} contains infinite value at index {}", name, i)
            ));
        }
    }
    Ok(())
}

/// Helper function to compute single-pair distance
fn compute_single_distance(
    a: &[f32],
    b: &[f32],
    metric: VectorMetric,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    // Validate dimensions
    validate_dimensions(a.len(), b.len())?;
    
    // Validate values
    validate_values(a, "vector a")?;
    validate_values(b, "vector b")?;

    
    // Compute distance using GPU if context provided, otherwise CPU
    if let Some(ctx) = context {
        // Use GPU acceleration
        let gpu_context = ctx.get_context();
        let mut vectors = Vec::with_capacity(a.len() * 2);
        vectors.extend_from_slice(a);
        vectors.extend_from_slice(b);
        
        let distances = compute_distance(a, b, a.len(), metric, gpu_context)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(distances[0])
    } else {
        // Use CPU computation
        let dist = match metric {
            VectorMetric::L2 => distance::l2_distance(a, b),
            VectorMetric::Cosine => distance::cosine_distance(a, b),
            VectorMetric::InnerProduct => distance::dot_product(a, b),
            VectorMetric::L1 => distance::l1_distance(a, b),
            VectorMetric::Hamming => distance::hamming_distance(a, b),
            VectorMetric::Jaccard => distance::jaccard_distance(a, b),
        };
        Ok(dist)
    }
}

// ============================================================================
// Single-pair distance functions
// ============================================================================

/// Compute L2 (Euclidean) distance between two vectors
/// 
/// The L2 distance is the square root of the sum of squared differences between
/// corresponding elements. Also known as Euclidean distance.
/// 
/// Parameters
/// ----------
/// a : array_like
///     First vector (NumPy array, list, or array-like)
/// b : array_like
///     Second vector (NumPy array, list, or array-like)
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration
/// 
/// Returns
/// -------
/// float
///     L2 distance between vectors
/// 
/// Raises
/// ------
/// ValueError
///     If vectors have different dimensions or contain NaN/inf
/// TypeError
///     If inputs are not numeric arrays
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> a = np.array([1.0, 2.0, 3.0])
/// >>> b = np.array([4.0, 5.0, 6.0])
/// >>> distance = hdb.l2(a, b)
/// >>> print(f"{distance:.3f}")
/// 5.196
#[pyfunction]
#[pyo3(name = "l2", signature = (a, b, context=None))]
pub fn py_l2(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    compute_single_distance(a_slice, b_slice, VectorMetric::L2, context)
}

/// Compute cosine distance between two vectors
/// 
/// Cosine distance = 1 - cosine_similarity, where cosine_similarity is the dot product
/// of normalized vectors. This metric is useful for comparing vector directions
/// regardless of magnitude.
/// 
/// Parameters
/// ----------
/// a : array_like
///     First vector (NumPy array, list, or array-like)
/// b : array_like
///     Second vector (NumPy array, list, or array-like)
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration
/// 
/// Returns
/// -------
/// float
///     Cosine distance between vectors (0 = identical direction, 2 = opposite direction)
/// 
/// Raises
/// ------
/// ValueError
///     If vectors have different dimensions or contain NaN/inf
/// TypeError
///     If inputs are not numeric arrays
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> a = np.array([1.0, 0.0, 0.0])
/// >>> b = np.array([0.0, 1.0, 0.0])
/// >>> distance = hdb.cosine(a, b)
/// >>> print(distance)
/// 1.0
#[pyfunction]
#[pyo3(name = "cosine", signature = (a, b, context=None))]
pub fn py_cosine(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    compute_single_distance(a_slice, b_slice, VectorMetric::Cosine, context)
}

/// Compute inner product (dot product) between two vectors
/// 
/// The inner product is the sum of element-wise products. Unlike cosine similarity,
/// this metric is sensitive to vector magnitudes.
/// 
/// Parameters
/// ----------
/// a : array_like
///     First vector (NumPy array, list, or array-like)
/// b : array_like
///     Second vector (NumPy array, list, or array-like)
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration
/// 
/// Returns
/// -------
/// float
///     Inner product of vectors
/// 
/// Raises
/// ------
/// ValueError
///     If vectors have different dimensions or contain NaN/inf
/// TypeError
///     If inputs are not numeric arrays
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> a = np.array([1.0, 2.0, 3.0])
/// >>> b = np.array([4.0, 5.0, 6.0])
/// >>> product = hdb.inner_product(a, b)
/// >>> print(product)
/// 32.0
#[pyfunction]
#[pyo3(name = "inner_product", signature = (a, b, context=None))]
pub fn py_inner_product(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    compute_single_distance(a_slice, b_slice, VectorMetric::InnerProduct, context)
}

/// Compute L1 (Manhattan) distance between two vectors
/// 
/// L1 distance is the sum of absolute differences between corresponding elements.
/// Also known as Manhattan distance or taxicab distance.
/// 
/// Parameters
/// ----------
/// a : array_like
///     First vector (NumPy array, list, or array-like)
/// b : array_like
///     Second vector (NumPy array, list, or array-like)
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration
/// 
/// Returns
/// -------
/// float
///     L1 distance between vectors
/// 
/// Raises
/// ------
/// ValueError
///     If vectors have different dimensions or contain NaN/inf
/// TypeError
///     If inputs are not numeric arrays
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> a = np.array([1.0, 2.0, 3.0])
/// >>> b = np.array([4.0, 5.0, 6.0])
/// >>> distance = hdb.l1(a, b)
/// >>> print(distance)
/// 9.0
#[pyfunction]
#[pyo3(name = "l1", signature = (a, b, context=None))]
pub fn py_l1(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    compute_single_distance(a_slice, b_slice, VectorMetric::L1, context)
}

/// Compute Hamming distance between two vectors
/// 
/// Hamming distance counts the number of positions where corresponding elements differ.
/// For continuous vectors, elements are considered different if they are not exactly equal.
/// 
/// Parameters
/// ----------
/// a : array_like
///     First vector (NumPy array, list, or array-like)
/// b : array_like
///     Second vector (NumPy array, list, or array-like)
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration
/// 
/// Returns
/// -------
/// float
///     Hamming distance (count of differing elements)
/// 
/// Raises
/// ------
/// ValueError
///     If vectors have different dimensions or contain NaN/inf
/// TypeError
///     If inputs are not numeric arrays
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> a = np.array([1.0, 2.0, 3.0, 4.0])
/// >>> b = np.array([1.0, 0.0, 3.0, 5.0])
/// >>> distance = hdb.hamming(a, b)
/// >>> print(distance)
/// 2.0
#[pyfunction]
#[pyo3(name = "hamming", signature = (a, b, context=None))]
pub fn py_hamming(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    compute_single_distance(a_slice, b_slice, VectorMetric::Hamming, context)
}

/// Compute Jaccard distance between two vectors
/// 
/// Jaccard distance = 1 - (intersection / union), where intersection and union
/// are computed based on non-zero elements. This metric is useful for comparing
/// set-like representations.
/// 
/// Parameters
/// ----------
/// a : array_like
///     First vector (NumPy array, list, or array-like)
/// b : array_like
///     Second vector (NumPy array, list, or array-like)
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration
/// 
/// Returns
/// -------
/// float
///     Jaccard distance (0 = identical sets, 1 = completely different)
/// 
/// Raises
/// ------
/// ValueError
///     If vectors have different dimensions or contain NaN/inf
/// TypeError
///     If inputs are not numeric arrays
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> a = np.array([1.0, 1.0, 0.0, 0.0])
/// >>> b = np.array([1.0, 0.0, 1.0, 0.0])
/// >>> distance = hdb.jaccard(a, b)
/// >>> print(f"{distance:.3f}")
/// 0.667
#[pyfunction]
#[pyo3(name = "jaccard", signature = (a, b, context=None))]
pub fn py_jaccard(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    compute_single_distance(a_slice, b_slice, VectorMetric::Jaccard, context)
}


// ============================================================================
// Batch distance functions
// ============================================================================

/// Helper function to compute batch distances
fn compute_batch_distances(
    query: &[f32],
    vectors: &[f32],
    dim: usize,
    metric: VectorMetric,
    context: Option<&PyComputeContext>,
) -> PyResult<Vec<f32>> {
    // Validate query dimension
    if query.len() != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Query vector length {} does not match dimension {}", query.len(), dim)
        ));
    }
    
    // Validate vectors array
    if vectors.len() % dim != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Vectors array length {} is not a multiple of dimension {}", vectors.len(), dim)
        ));
    }
    
    // Validate values
    validate_values(query, "query vector")?;
    validate_values(vectors, "database vectors")?;
    
    // Compute distances using GPU if context provided, otherwise CPU
    if let Some(ctx) = context {
        let gpu_context = ctx.get_context();
        let start = std::time::Instant::now();
        
        let results = compute_distance_batch(query, vectors, dim, metric, gpu_context)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
        let duration = start.elapsed();
        
        // Update stats
        {
            let tracker = ctx.get_stats_tracker();
            let mut stats = tracker.lock().unwrap();
            if gpu_context.backend == ComputeBackend::Cpu {
                 stats.total_cpu_time_ms += duration.as_secs_f64() * 1000.0;
            } else {
                 stats.total_kernel_launches += 1;
                 stats.total_gpu_time_ms += duration.as_secs_f64() * 1000.0;
                 let bytes = (query.len() + vectors.len() + results.len()) * std::mem::size_of::<f32>();
                 stats.memory_transfers_mb += bytes as f64 / 1024.0 / 1024.0;
            }
            stats.total_vectors_processed += (vectors.len() / dim) as u64;
        }
        
        Ok(results)
    } else {
        // Use CPU computation
        let n_vectors = vectors.len() / dim;
        let mut distances = Vec::with_capacity(n_vectors);
        
        for i in 0..n_vectors {
            let start_idx = i * dim;
            let end_idx = start_idx + dim;
            let vector = &vectors[start_idx..end_idx];
            
            let dist = match metric {
                VectorMetric::L2 => distance::l2_distance(query, vector),
                VectorMetric::Cosine => distance::cosine_distance(query, vector),
                VectorMetric::InnerProduct => distance::dot_product(query, vector),
                VectorMetric::L1 => distance::l1_distance(query, vector),
                VectorMetric::Hamming => distance::hamming_distance(query, vector),
                VectorMetric::Jaccard => distance::jaccard_distance(query, vector),
            };
            distances.push(dist);
        }
        
        // Note: CPU path without context doesn't track stats since there's no tracker
        
        Ok(distances)
    }
}

/// Compute L2 distances between a query vector and multiple database vectors
/// 
/// This function efficiently computes L2 (Euclidean) distances between one query vector
/// and many database vectors in a single operation. GPU acceleration provides significant
/// speedup for large databases.
/// 
/// Parameters
/// ----------
/// query : array_like
///     Query vector (1D NumPy array)
/// vectors : array_like
///     Database vectors (2D NumPy array, shape: [n_vectors, dim])
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration
/// 
/// Returns
/// -------
/// ndarray
///     1D array of distances, one per database vector
/// 
/// Raises
/// ------
/// ValueError
///     If query dimension doesn't match vectors dimension, or if inputs contain NaN/inf
/// TypeError
///     If inputs are not numeric arrays
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> query = np.array([1.0, 2.0, 3.0])
/// >>> vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
/// >>> distances = hdb.l2_batch(query, vectors)
/// >>> print(distances)
/// [0.0, 5.196...]
#[pyfunction]
#[pyo3(name = "l2_batch", signature = (query, vectors, context=None))]
pub fn py_l2_batch<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f32>,
    vectors: PyReadonlyArray2<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query_slice = query.as_slice()?;
    let vectors_array = vectors.as_array();
    let shape = vectors_array.shape();
    let dim = shape[1];
    
    // Flatten the 2D array to 1D for processing
    let vectors_slice = vectors.as_slice()?;
    
    let distances = compute_batch_distances(query_slice, vectors_slice, dim, VectorMetric::L2, context)?;
    
    Ok(PyArray1::from_vec(py, distances))
}

/// Compute cosine distances between a query vector and multiple database vectors
/// 
/// This function efficiently computes cosine distances between one query vector and
/// many database vectors in a single operation. When using GPU acceleration, this
/// provides significant performance benefits over computing distances individually.
/// 
/// Args:
///     query: Query vector (1D NumPy array)
///     vectors: Database vectors (2D NumPy array, shape: [n_vectors, dim])
///     context: Optional ComputeContext for GPU acceleration
/// 
/// Returns:
///     np.ndarray: 1D array of cosine distances, one per database vector
/// 
/// Raises:
///     ValueError: If query dimension doesn't match vectors dimension, or if inputs contain NaN/inf
///     TypeError: If inputs are not numeric arrays
/// 
/// Example:
///     >>> import hyperstreamdb as hdb
///     >>> import numpy as np
///     >>> query = np.array([1.0, 0.0, 0.0])
///     >>> vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
///     >>> distances = hdb.cosine_batch(query, vectors)
///     >>> print(distances)  # [0.0, 1.0, 1.0]
#[pyfunction]
#[pyo3(name = "cosine_batch", signature = (query, vectors, context=None))]
pub fn py_cosine_batch<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f32>,
    vectors: PyReadonlyArray2<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query_slice = query.as_slice()?;
    let vectors_array = vectors.as_array();
    let shape = vectors_array.shape();
    let dim = shape[1];
    let vectors_slice = vectors.as_slice()?;
    
    let distances = compute_batch_distances(query_slice, vectors_slice, dim, VectorMetric::Cosine, context)?;
    
    Ok(PyArray1::from_vec(py, distances))
}

/// Compute inner product between a query vector and multiple database vectors
/// 
/// This function efficiently computes inner products between one query vector and
/// many database vectors in a single operation. Useful for similarity search where
/// higher inner products indicate more similar vectors.
/// 
/// Args:
///     query: Query vector (1D NumPy array)
///     vectors: Database vectors (2D NumPy array, shape: [n_vectors, dim])
///     context: Optional ComputeContext for GPU acceleration
/// 
/// Returns:
///     np.ndarray: 1D array of inner products, one per database vector
/// 
/// Raises:
///     ValueError: If query dimension doesn't match vectors dimension, or if inputs contain NaN/inf
///     TypeError: If inputs are not numeric arrays
/// 
/// Example:
///     >>> import hyperstreamdb as hdb
///     >>> import numpy as np
///     >>> query = np.array([1.0, 2.0, 3.0])
///     >>> vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
///     >>> products = hdb.inner_product_batch(query, vectors)
///     >>> print(products)  # [14.0, 32.0]
#[pyfunction]
#[pyo3(name = "inner_product_batch", signature = (query, vectors, context=None))]
pub fn py_inner_product_batch<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f32>,
    vectors: PyReadonlyArray2<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query_slice = query.as_slice()?;
    let vectors_array = vectors.as_array();
    let shape = vectors_array.shape();
    let dim = shape[1];
    let vectors_slice = vectors.as_slice()?;
    
    let distances = compute_batch_distances(query_slice, vectors_slice, dim, VectorMetric::InnerProduct, context)?;
    
    Ok(PyArray1::from_vec(py, distances))
}

/// Compute L1 distances between a query vector and multiple database vectors
/// 
/// This function efficiently computes L1 (Manhattan) distances between one query vector
/// and many database vectors in a single operation. GPU acceleration provides significant
/// speedup for large databases.
/// 
/// Args:
///     query: Query vector (1D NumPy array)
///     vectors: Database vectors (2D NumPy array, shape: [n_vectors, dim])
///     context: Optional ComputeContext for GPU acceleration
/// 
/// Returns:
///     np.ndarray: 1D array of L1 distances, one per database vector
/// 
/// Raises:
///     ValueError: If query dimension doesn't match vectors dimension, or if inputs contain NaN/inf
///     TypeError: If inputs are not numeric arrays
/// 
/// Example:
///     >>> import hyperstreamdb as hdb
///     >>> import numpy as np
///     >>> query = np.array([0.0, 0.0, 0.0])
///     >>> vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
///     >>> distances = hdb.l1_batch(query, vectors)
///     >>> print(distances)  # [6.0, 15.0]
#[pyfunction]
#[pyo3(name = "l1_batch", signature = (query, vectors, context=None))]
pub fn py_l1_batch<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f32>,
    vectors: PyReadonlyArray2<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query_slice = query.as_slice()?;
    let vectors_array = vectors.as_array();
    let shape = vectors_array.shape();
    let dim = shape[1];
    let vectors_slice = vectors.as_slice()?;
    
    let distances = compute_batch_distances(query_slice, vectors_slice, dim, VectorMetric::L1, context)?;
    
    Ok(PyArray1::from_vec(py, distances))
}

/// Compute Hamming distances between a query vector and multiple database vectors
/// 
/// This function efficiently computes Hamming distances (count of differing elements)
/// between one query vector and many database vectors in a single operation.
/// 
/// Args:
///     query: Query vector (1D NumPy array)
///     vectors: Database vectors (2D NumPy array, shape: [n_vectors, dim])
///     context: Optional ComputeContext for GPU acceleration
/// 
/// Returns:
///     np.ndarray: 1D array of Hamming distances, one per database vector
/// 
/// Raises:
///     ValueError: If query dimension doesn't match vectors dimension, or if inputs contain NaN/inf
///     TypeError: If inputs are not numeric arrays
/// 
/// Example:
///     >>> import hyperstreamdb as hdb
///     >>> import numpy as np
///     >>> query = np.array([1.0, 0.0, 1.0, 0.0])
///     >>> vectors = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
///     >>> distances = hdb.hamming_batch(query, vectors)
///     >>> print(distances)  # [0.0, 4.0]
#[pyfunction]
#[pyo3(name = "hamming_batch", signature = (query, vectors, context=None))]
pub fn py_hamming_batch<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f32>,
    vectors: PyReadonlyArray2<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query_slice = query.as_slice()?;
    let vectors_array = vectors.as_array();
    let shape = vectors_array.shape();
    let dim = shape[1];
    let vectors_slice = vectors.as_slice()?;
    
    let distances = compute_batch_distances(query_slice, vectors_slice, dim, VectorMetric::Hamming, context)?;
    
    Ok(PyArray1::from_vec(py, distances))
}

/// Compute Jaccard distances between a query vector and multiple database vectors
/// 
/// This function efficiently computes Jaccard distances (1 - intersection/union)
/// between one query vector and many database vectors in a single operation.
/// 
/// Args:
///     query: Query vector (1D NumPy array)
///     vectors: Database vectors (2D NumPy array, shape: [n_vectors, dim])
///     context: Optional ComputeContext for GPU acceleration
/// 
/// Returns:
///     np.ndarray: 1D array of Jaccard distances, one per database vector
/// 
/// Raises:
///     ValueError: If query dimension doesn't match vectors dimension, or if inputs contain NaN/inf
///     TypeError: If inputs are not numeric arrays
/// 
/// Example:
///     >>> import hyperstreamdb as hdb
///     >>> import numpy as np
///     >>> query = np.array([1.0, 1.0, 0.0, 0.0])
///     >>> vectors = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
///     >>> distances = hdb.jaccard_batch(query, vectors)
///     >>> print(distances)  # [0.0, 0.667]
#[pyfunction]
#[pyo3(name = "jaccard_batch", signature = (query, vectors, context=None))]
pub fn py_jaccard_batch<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f32>,
    vectors: PyReadonlyArray2<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query_slice = query.as_slice()?;
    let vectors_array = vectors.as_array();
    let shape = vectors_array.shape();
    let dim = shape[1];
    let vectors_slice = vectors.as_slice()?;
    
    let distances = compute_batch_distances(query_slice, vectors_slice, dim, VectorMetric::Jaccard, context)?;
    
    Ok(PyArray1::from_vec(py, distances))
}

// ============================================================================
// Sparse Vector Support
// ============================================================================

/// Sparse vector representation storing only non-zero indices and values
/// 
/// This class provides an efficient representation for high-dimensional sparse vectors
/// by storing only the non-zero elements. It validates that indices are sorted and
/// within dimension bounds.
/// 
/// Parameters
/// ----------
/// indices : array_like
///     Array of non-zero indices (must be sorted, dtype: uint32)
/// values : array_like
///     Array of non-zero values (same length as indices, dtype: float32)
/// dim : int
///     Total dimension of the vector
/// 
/// Raises
/// ------
/// ValueError
///     If indices are not sorted, out of bounds, or length mismatch with values
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> # Create a sparse vector with non-zero values at indices 0, 5, 10
/// >>> sparse = hdb.SparseVector([0, 5, 10], [1.0, 2.0, 3.0], 100)
/// >>> print(sparse.dim)
/// 100
/// >>> dense = sparse.to_dense()  # Convert to dense representation
/// >>> print(dense[5])
/// 2.0
#[pyclass(name = "SparseVector")]
pub struct PySparseVector {
    indices: Vec<u32>,
    values: Vec<f32>,
    dim: usize,
}

#[pymethods]
impl PySparseVector {
    /// Create a new sparse vector
    /// 
    /// Parameters
    /// ----------
    /// indices : array_like
    ///     Non-zero indices (must be sorted and within [0, dim), dtype: uint32)
    /// values : array_like
    ///     Non-zero values (same length as indices, dtype: float32)
    /// dim : int
    ///     Total vector dimension
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If indices/values length mismatch, indices not sorted, or indices out of bounds
    #[new]
    pub fn new(
        indices: PyReadonlyArray1<u32>,
        values: PyReadonlyArray1<f32>,
        dim: usize,
    ) -> PyResult<Self> {
        let indices_slice = indices.as_slice()?;
        let values_slice = values.as_slice()?;
        
        // Validate lengths match
        if indices_slice.len() != values_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Length mismatch: indices has {} elements, values has {} elements",
                    indices_slice.len(), values_slice.len())
            ));
        }
        
        // Validate indices are sorted and within bounds
        validate_sparse_indices(indices_slice, dim)?;
        
        // Validate values don't contain NaN/inf
        validate_values(values_slice, "sparse vector values")?;
        
        Ok(PySparseVector {
            indices: indices_slice.to_vec(),
            values: values_slice.to_vec(),
            dim,
        })
    }
    
    /// Get the non-zero indices
    /// 
    /// Returns
    /// -------
    /// ndarray
    ///     Array of non-zero indices (dtype: uint32)
    #[getter]
    pub fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_vec(py, self.indices.clone())
    }
    
    /// Get the non-zero values
    /// 
    /// Returns
    /// -------
    /// ndarray
    ///     Array of non-zero values (dtype: float32)
    #[getter]
    pub fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.values.clone())
    }
    
    /// Get the total vector dimension
    /// 
    /// Returns
    /// -------
    /// int
    ///     Total dimension of the vector
    #[getter]
    pub fn dim(&self) -> usize {
        self.dim
    }
    
    /// Convert sparse vector to dense representation
    /// 
    /// Returns
    /// -------
    /// ndarray
    ///     Dense vector with zeros at non-specified indices (dtype: float32)
    /// 
    /// Examples
    /// --------
    /// >>> import hyperstreamdb as hdb
    /// >>> sparse = hdb.SparseVector([0, 2], [1.0, 3.0], 5)
    /// >>> dense = sparse.to_dense()
    /// >>> print(dense)
    /// [1.0, 0.0, 3.0, 0.0, 0.0]
    pub fn to_dense<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut dense = vec![0.0f32; self.dim];
        for (idx, val) in self.indices.iter().zip(self.values.iter()) {
            dense[*idx as usize] = *val;
        }
        PyArray1::from_vec(py, dense)
    }
}

/// Helper function to validate sparse vector indices
fn validate_sparse_indices(indices: &[u32], dim: usize) -> PyResult<()> {
    // Check indices are sorted and within bounds
    for i in 0..indices.len() {
        let idx = indices[i] as usize;
        
        // Check bounds
        if idx >= dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Sparse vector index {} is out of bounds for dimension {}", idx, dim)
            ));
        }
        
        // Check sorted (each index should be greater than previous)
        if i > 0 && indices[i] <= indices[i - 1] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Sparse vector indices must be sorted. Found index {} after index {}",
                    indices[i], indices[i - 1])
            ));
        }
    }
    Ok(())
}

/// Compute L2 distance between two sparse vectors
/// 
/// This function computes L2 (Euclidean) distance efficiently for sparse vectors by only
/// processing non-zero elements. The result is mathematically equivalent to converting
/// to dense and computing L2 distance, but much more efficient for high-dimensional
/// sparse data.
/// 
/// Parameters
/// ----------
/// a : SparseVector
///     First sparse vector
/// b : SparseVector
///     Second sparse vector
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration (currently unused for sparse)
/// 
/// Returns
/// -------
/// float
///     L2 distance between sparse vectors
/// 
/// Raises
/// ------
/// ValueError
///     If vectors have different dimensions
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> a = hdb.SparseVector([0, 5], [1.0, 2.0], 10)
/// >>> b = hdb.SparseVector([0, 3], [1.0, 3.0], 10)
/// >>> distance = hdb.l2_sparse(a, b)
/// >>> print(f"{distance:.3f}")
/// 3.606
#[pyfunction]
#[pyo3(name = "l2_sparse", signature = (a, b, context=None))]
pub fn py_l2_sparse(
    a: &PySparseVector,
    b: &PySparseVector,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    // Validate dimensions match
    if a.dim != b.dim {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Vector dimension mismatch: a has dimension {}, b has dimension {}",
                a.dim, b.dim)
        ));
    }
    
    // Note: GPU acceleration for sparse vectors is not yet implemented
    // Always use CPU computation
    let _ = context; // Suppress unused warning
    
    let dist_squared = distance::sparse_l2_distance_squared(
        &a.indices, &a.values,
        &b.indices, &b.values
    );
    
    Ok(dist_squared.sqrt())
}

/// Compute cosine distance between two sparse vectors
/// 
/// This function computes cosine distance efficiently for sparse vectors by only
/// processing non-zero elements. The result is mathematically equivalent to converting
/// to dense and computing cosine distance, but much more efficient for high-dimensional
/// sparse data.
/// 
/// Args:
///     a: First sparse vector
///     b: Second sparse vector
///     context: Optional ComputeContext for GPU acceleration (currently unused for sparse)
/// 
/// Returns:
///     float: Cosine distance between sparse vectors (0 = identical direction, 2 = opposite)
/// 
/// Raises:
///     ValueError: If vectors have different dimensions
/// 
/// Example:
///     >>> import hyperstreamdb as hdb
///     >>> import numpy as np
///     >>> # Two sparse vectors in 1000-dimensional space
///     >>> a = hdb.SparseVector([0, 100, 500], [1.0, 2.0, 3.0], 1000)
///     >>> b = hdb.SparseVector([0, 200, 500], [1.0, 1.0, 3.0], 1000)
///     >>> distance = hdb.cosine_sparse(a, b)
#[pyfunction]
#[pyo3(name = "cosine_sparse", signature = (a, b, context=None))]
pub fn py_cosine_sparse(
    a: &PySparseVector,
    b: &PySparseVector,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    // Validate dimensions match
    if a.dim != b.dim {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Vector dimension mismatch: a has dimension {}, b has dimension {}",
                a.dim, b.dim)
        ));
    }
    
    let _ = context; // Suppress unused warning
    
    // Compute dot product
    let dot = distance::sparse_dot_product(
        &a.indices, &a.values,
        &b.indices, &b.values
    );
    
    // Compute norms
    let norm_a_sq = distance::sparse_dot_product(
        &a.indices, &a.values,
        &a.indices, &a.values
    );
    let norm_b_sq = distance::sparse_dot_product(
        &b.indices, &b.values,
        &b.indices, &b.values
    );
    
    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(1.0); // Maximum cosine distance
    }
    
    let similarity = dot / (norm_a * norm_b);
    Ok(1.0 - similarity)
}

/// Compute inner product between two sparse vectors
/// 
/// This function computes the inner product (dot product) efficiently for sparse vectors
/// by only processing non-zero elements. The result is mathematically equivalent to
/// converting to dense and computing inner product.
/// 
/// Args:
///     a: First sparse vector
///     b: Second sparse vector
///     context: Optional ComputeContext for GPU acceleration (currently unused for sparse)
/// 
/// Returns:
///     float: Inner product of sparse vectors
/// 
/// Raises:
///     ValueError: If vectors have different dimensions
/// 
/// Example:
///     >>> import hyperstreamdb as hdb
///     >>> import numpy as np
///     >>> # Sparse vectors with some overlapping non-zero indices
///     >>> a = hdb.SparseVector([0, 2, 4], [1.0, 2.0, 3.0], 10)
///     >>> b = hdb.SparseVector([0, 2, 5], [2.0, 3.0, 1.0], 10)
///     >>> product = hdb.inner_product_sparse(a, b)
///     >>> print(product)  # 8.0 (1*2 + 2*3 + 0)
#[pyfunction]
#[pyo3(name = "inner_product_sparse", signature = (a, b, context=None))]
pub fn py_inner_product_sparse(
    a: &PySparseVector,
    b: &PySparseVector,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    // Validate dimensions match
    if a.dim != b.dim {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Vector dimension mismatch: a has dimension {}, b has dimension {}",
                a.dim, b.dim)
        ));
    }
    
    let _ = context; // Suppress unused warning
    
    let dot = distance::sparse_dot_product(
        &a.indices, &a.values,
        &b.indices, &b.values
    );
    
    Ok(dot)
}

// ============================================================================
// Binary Vector Support
// ============================================================================

/// Helper function to check if a float array is binary (all values are 0.0 or 1.0)
fn is_binary_vector(values: &[f32]) -> bool {
    values.iter().all(|&x| x == 0.0 || x == 1.0)
}

/// Helper function to pack a binary vector from f32 array to u8 array
/// Each byte stores 8 bits
fn pack_binary_vector(values: &[f32]) -> Vec<u8> {
    let num_bytes = (values.len() + 7) / 8; // Round up to nearest byte
    let mut packed = vec![0u8; num_bytes];
    
    for (i, &val) in values.iter().enumerate() {
        if val != 0.0 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            packed[byte_idx] |= 1 << (7 - bit_idx); // MSB first
        }
    }
    
    packed
}

/// Compute Hamming distance between two bit-packed binary vectors
/// 
/// This function computes the Hamming distance (count of differing bits) between
/// two binary vectors represented as packed uint8 arrays. Each byte stores 8 bits.
/// 
/// If the input vectors are unpacked (f32 arrays with 0.0/1.0 values), they will
/// be automatically packed before computing the distance.
/// 
/// Parameters
/// ----------
/// a : array_like
///     First binary vector (NumPy uint8 array, bit-packed)
/// b : array_like
///     Second binary vector (NumPy uint8 array, bit-packed)
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration (currently unused)
/// 
/// Returns
/// -------
/// int
///     Number of differing bits between the vectors
/// 
/// Raises
/// ------
/// ValueError
///     If vectors have different lengths
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> # Two binary vectors: 10110101 and 10101100
/// >>> a = np.array([0b10110101], dtype=np.uint8)
/// >>> b = np.array([0b10101100], dtype=np.uint8)
/// >>> distance = hdb.hamming_packed(a, b)
/// >>> print(distance)
/// 3
#[pyfunction]
#[pyo3(name = "hamming_packed", signature = (a, b, context=None))]
pub fn py_hamming_packed(
    a: PyReadonlyArray1<u8>,
    b: PyReadonlyArray1<u8>,
    context: Option<&PyComputeContext>,
) -> PyResult<u32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    
    // Validate dimensions
    if a_slice.len() != b_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Binary vector length mismatch: a has {} bytes, b has {} bytes",
                a_slice.len(), b_slice.len())
        ));
    }
    
    // Note: GPU acceleration for binary vectors is not yet implemented
    // Always use CPU computation
    let _ = context; // Suppress unused warning
    
    let distance = distance::hamming_distance_packed(a_slice, b_slice);
    Ok(distance)
}

/// Compute Hamming distance with auto-packing support for unpacked binary vectors
/// 
/// This function accepts either packed (uint8) or unpacked (f32 with 0.0/1.0 values)
/// binary vectors and automatically packs them if needed before computing distance.
/// This provides a convenient interface when working with binary data that hasn't
/// been pre-packed.
/// 
/// Parameters
/// ----------
/// a : array_like
///     First binary vector (NumPy array, uint8 or f32 with 0.0/1.0 values)
/// b : array_like
///     Second binary vector (NumPy array, uint8 or f32 with 0.0/1.0 values)
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration (currently unused)
/// 
/// Returns
/// -------
/// int
///     Number of differing bits between the vectors
/// 
/// Raises
/// ------
/// ValueError
///     If vectors have different lengths or contain non-binary values
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> # Unpacked binary vectors (will be auto-packed)
/// >>> a = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
/// >>> b = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
/// >>> distance = hdb.hamming_auto(a, b)
/// >>> print(distance)
/// 3
#[pyfunction]
#[pyo3(name = "hamming_auto", signature = (a, b, context=None))]
pub fn py_hamming_auto(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<u32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    
    // Validate dimensions
    validate_dimensions(a_slice.len(), b_slice.len())?;
    
    // Check if vectors are binary
    if !is_binary_vector(a_slice) || !is_binary_vector(b_slice) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Vectors must contain only 0.0 or 1.0 values for binary Hamming distance"
        ));
    }
    
    // Pack the vectors
    let a_packed = pack_binary_vector(a_slice);
    let b_packed = pack_binary_vector(b_slice);
    
    // Note: GPU acceleration for binary vectors is not yet implemented
    let _ = context; // Suppress unused warning
    
    let distance = distance::hamming_distance_packed(&a_packed, &b_packed);
    Ok(distance)
}

/// Compute Jaccard distance between two bit-packed binary vectors
/// 
/// This function computes the Jaccard distance (1 - intersection/union) between
/// two binary vectors represented as packed uint8 arrays. Each byte stores 8 bits.
/// 
/// Parameters
/// ----------
/// a : array_like
///     First binary vector (NumPy uint8 array, bit-packed)
/// b : array_like
///     Second binary vector (NumPy uint8 array, bit-packed)
/// context : ComputeContext, optional
///     Optional ComputeContext for GPU acceleration (currently unused)
/// 
/// Returns
/// -------
/// float
///     Jaccard distance (0.0 = identical, 1.0 = completely different)
/// 
/// Raises
/// ------
/// ValueError
///     If vectors have different lengths
/// 
/// Examples
/// --------
/// >>> import hyperstreamdb as hdb
/// >>> import numpy as np
/// >>> a = np.array([0b10110101], dtype=np.uint8)
/// >>> b = np.array([0b10101100], dtype=np.uint8)
/// >>> distance = hdb.jaccard_packed(a, b)
/// >>> print(f"{distance:.3f}")
/// 0.429
#[pyfunction]
#[pyo3(name = "jaccard_packed", signature = (a, b, context=None))]
pub fn py_jaccard_packed(
    a: PyReadonlyArray1<u8>,
    b: PyReadonlyArray1<u8>,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    
    // Validate dimensions
    if a_slice.len() != b_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Binary vector length mismatch: a has {} bytes, b has {} bytes",
                a_slice.len(), b_slice.len())
        ));
    }
    
    // Note: GPU acceleration for binary vectors is not yet implemented
    // Always use CPU computation
    let _ = context; // Suppress unused warning
    
    // Compute intersection and union using bit operations
    let mut intersection = 0u32;
    let mut union = 0u32;
    
    for (&byte_a, &byte_b) in a_slice.iter().zip(b_slice.iter()) {
        intersection += (byte_a & byte_b).count_ones();
        union += (byte_a | byte_b).count_ones();
    }
    
    if union == 0 {
        return Ok(0.0); // Both vectors are all zeros
    }
    
    let jaccard_similarity = intersection as f32 / union as f32;
    Ok(1.0 - jaccard_similarity)
}

/// Compute Jaccard distance with auto-packing support for unpacked binary vectors
/// 
/// This function accepts either packed (uint8) or unpacked (f32 with 0.0/1.0 values)
/// binary vectors and automatically packs them if needed before computing distance.
/// Jaccard distance measures dissimilarity between sets, computed as 1 - (intersection/union).
/// 
/// Args:
///     a: First binary vector (NumPy array, uint8 or f32 with 0.0/1.0 values)
///     b: Second binary vector (NumPy array, uint8 or f32 with 0.0/1.0 values)
///     context: Optional ComputeContext for GPU acceleration (currently unused)
/// 
/// Returns:
///     float: Jaccard distance (0.0 = identical sets, 1.0 = completely different)
/// 
/// Raises:
///     ValueError: If vectors have different lengths or contain non-binary values
/// 
/// Example:
///     >>> import hyperstreamdb as hdb
///     >>> import numpy as np
///     >>> # Unpacked binary vectors representing sets
///     >>> a = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
///     >>> b = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
///     >>> distance = hdb.jaccard_auto(a, b)  # Auto-packs and computes
///     >>> print(distance)  # Jaccard distance based on set overlap
#[pyfunction]
#[pyo3(name = "jaccard_auto", signature = (a, b, context=None))]
pub fn py_jaccard_auto(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    context: Option<&PyComputeContext>,
) -> PyResult<f32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    
    // Validate dimensions
    validate_dimensions(a_slice.len(), b_slice.len())?;
    
    // Check if vectors are binary
    if !is_binary_vector(a_slice) || !is_binary_vector(b_slice) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Vectors must contain only 0.0 or 1.0 values for binary Jaccard distance"
        ));
    }
    
    // Pack the vectors
    let a_packed = pack_binary_vector(a_slice);
    let b_packed = pack_binary_vector(b_slice);
    
    // Note: GPU acceleration for binary vectors is not yet implemented
    let _ = context; // Suppress unused warning
    
    // Compute intersection and union using bit operations
    let mut intersection = 0u32;
    let mut union = 0u32;
    
    for (&byte_a, &byte_b) in a_packed.iter().zip(b_packed.iter()) {
        intersection += (byte_a & byte_b).count_ones();
        union += (byte_a | byte_b).count_ones();
    }
    
    if union == 0 {
        return Ok(0.0); // Both vectors are all zeros
    }
    
    let jaccard_similarity = intersection as f32 / union as f32;
    Ok(1.0 - jaccard_similarity)
}

