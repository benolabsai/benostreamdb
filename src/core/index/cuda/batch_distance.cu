// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// High-Performance 2D-Tiled Batched Distance Kernels for M queries vs N vectors.

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// ---------------------------------------------------------------------------
// Batched L2 Distance Kernel
// Computes M x N Euclidean distances: distances[q * n_vectors + v] = ||q - v||
// ---------------------------------------------------------------------------
extern "C" __global__ void batch_l2_distance_kernel(
    const float* __restrict__ queries,  // [n_queries, dim]
    const float* __restrict__ vectors,  // [n_vectors, dim]
    float* __restrict__ distances,      // [n_queries, n_vectors]
    int dim,
    int n_queries,
    int n_vectors
) {
    // 2D thread block: (TILE_N, TILE_M) = (16, 16) = 256 threads
    int tx = threadIdx.x; // 0..15: vector column in tile
    int ty = threadIdx.y; // 0..15: query row in tile

    int v_idx = blockIdx.x * TILE_N + tx;
    int q_idx = blockIdx.y * TILE_M + ty;

    // Shared memory tiles (+1 column padding to eliminate 32-way bank conflicts)
    __shared__ float s_q[TILE_M][TILE_K + 1];
    __shared__ float s_v[TILE_N][TILE_K + 1];

    float acc = 0.0f;

    // Loop over dimension in chunks of TILE_K
    for (int k_step = 0; k_step < dim; k_step += TILE_K) {
        int k_tx = k_step + tx;
        int k_ty = k_step + ty;

        // Cooperative coalesced load of query tile
        // Consecutive threads tx read consecutive memory addresses: queries[q_idx * dim + k_tx]
        if (q_idx < n_queries && k_tx < dim) {
            s_q[ty][tx] = queries[q_idx * dim + k_tx];
        } else {
            s_q[ty][tx] = 0.0f;
        }

        // Cooperative coalesced load of vector tile
        // Transposed indexing so threads tx read consecutive vector elements
        int v_load = blockIdx.x * TILE_N + ty;
        if (v_load < n_vectors && k_tx < dim) {
            s_v[ty][tx] = vectors[v_load * dim + k_tx];
        } else {
            s_v[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Accumulate squared differences from fast shared memory
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float diff = s_q[ty][k] - s_v[tx][k];
            acc += diff * diff;
        }

        __syncthreads();
    }

    if (q_idx < n_queries && v_idx < n_vectors) {
        distances[q_idx * n_vectors + v_idx] = sqrtf(acc);
    }
}

// ---------------------------------------------------------------------------
// Batched Cosine Distance Kernel
// Computes 1 - (q . v) / (||q|| * ||v||)
// ---------------------------------------------------------------------------
extern "C" __global__ void batch_cosine_distance_kernel(
    const float* __restrict__ queries,  // [n_queries, dim]
    const float* __restrict__ vectors,  // [n_vectors, dim]
    float* __restrict__ distances,      // [n_queries, n_vectors]
    int dim,
    int n_queries,
    int n_vectors
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int v_idx = blockIdx.x * TILE_N + tx;
    int q_idx = blockIdx.y * TILE_M + ty;

    __shared__ float s_q[TILE_M][TILE_K + 1];
    __shared__ float s_v[TILE_N][TILE_K + 1];

    float dot = 0.0f;
    float norm_q = 0.0f;
    float norm_v = 0.0f;

    for (int k_step = 0; k_step < dim; k_step += TILE_K) {
        int k_tx = k_step + tx;

        if (q_idx < n_queries && k_tx < dim) {
            s_q[ty][tx] = queries[q_idx * dim + k_tx];
        } else {
            s_q[ty][tx] = 0.0f;
        }

        int v_load = blockIdx.x * TILE_N + ty;
        if (v_load < n_vectors && k_tx < dim) {
            s_v[ty][tx] = vectors[v_load * dim + k_tx];
        } else {
            s_v[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float q = s_q[ty][k];
            float v = s_v[tx][k];
            dot += q * v;
            norm_q += q * q;
            norm_v += v * v;
        }

        __syncthreads();
    }

    if (q_idx < n_queries && v_idx < n_vectors) {
        float denom = sqrtf(norm_q) * sqrtf(norm_v);
        distances[q_idx * n_vectors + v_idx] = (denom > 1e-10f) ? (1.0f - (dot / denom)) : 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Batched Inner Product Kernel
// Computes - (q . v)
// ---------------------------------------------------------------------------
extern "C" __global__ void batch_inner_product_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ vectors,
    float* __restrict__ distances,
    int dim,
    int n_queries,
    int n_vectors
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int v_idx = blockIdx.x * TILE_N + tx;
    int q_idx = blockIdx.y * TILE_M + ty;

    __shared__ float s_q[TILE_M][TILE_K + 1];
    __shared__ float s_v[TILE_N][TILE_K + 1];

    float dot = 0.0f;

    for (int k_step = 0; k_step < dim; k_step += TILE_K) {
        int k_tx = k_step + tx;

        if (q_idx < n_queries && k_tx < dim) {
            s_q[ty][tx] = queries[q_idx * dim + k_tx];
        } else {
            s_q[ty][tx] = 0.0f;
        }

        int v_load = blockIdx.x * TILE_N + ty;
        if (v_load < n_vectors && k_tx < dim) {
            s_v[ty][tx] = vectors[v_load * dim + k_tx];
        } else {
            s_v[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            dot += s_q[ty][k] * s_v[tx][k];
        }

        __syncthreads();
    }

    if (q_idx < n_queries && v_idx < n_vectors) {
        distances[q_idx * n_vectors + v_idx] = -dot;
    }
}

// ---------------------------------------------------------------------------
// Batched L1 Distance Kernel
// Computes sum(|q - v|)
// ---------------------------------------------------------------------------
extern "C" __global__ void batch_l1_distance_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ vectors,
    float* __restrict__ distances,
    int dim,
    int n_queries,
    int n_vectors
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int v_idx = blockIdx.x * TILE_N + tx;
    int q_idx = blockIdx.y * TILE_M + ty;

    __shared__ float s_q[TILE_M][TILE_K + 1];
    __shared__ float s_v[TILE_N][TILE_K + 1];

    float acc = 0.0f;

    for (int k_step = 0; k_step < dim; k_step += TILE_K) {
        int k_tx = k_step + tx;

        if (q_idx < n_queries && k_tx < dim) {
            s_q[ty][tx] = queries[q_idx * dim + k_tx];
        } else {
            s_q[ty][tx] = 0.0f;
        }

        int v_load = blockIdx.x * TILE_N + ty;
        if (v_load < n_vectors && k_tx < dim) {
            s_v[ty][tx] = vectors[v_load * dim + k_tx];
        } else {
            s_v[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += fabsf(s_q[ty][k] - s_v[tx][k]);
        }

        __syncthreads();
    }

    if (q_idx < n_queries && v_idx < n_vectors) {
        distances[q_idx * n_vectors + v_idx] = acc;
    }
}

