extern "C" __global__ void cosine_distance_kernel(
    const float* __restrict__ query,
    const float* __restrict__ vectors,
    float* __restrict__ distances,
    int dim,
    int n_vectors
) {
    // Each block handles one vector (row)
    int row = blockIdx.x;
    if (row >= n_vectors) return;

    // Pointer to the start of the current vector
    const float* current_vector = vectors + row * dim;

    // Shared memory for reduction
    // Size should be equal to blockDim.x * 3 * sizeof(float)
    // We need to store: dot_product, norm_query, norm_vector
    extern __shared__ float sdata[];

    // 1. Thread-local partial sums for dot product and norms
    float local_dot = 0.0f;
    float local_norm_query = 0.0f;
    float local_norm_vector = 0.0f;
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float q = query[i];
        float v = current_vector[i];
        local_dot += q * v;
        local_norm_query += q * q;
        local_norm_vector += v * v;
    }
    
    // Store in shared memory (interleaved for better memory access)
    int tid = threadIdx.x;
    sdata[tid * 3 + 0] = local_dot;
    sdata[tid * 3 + 1] = local_norm_query;
    sdata[tid * 3 + 2] = local_norm_vector;
    __syncthreads();

    // 2. Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[tid * 3 + 0] += sdata[(tid + s) * 3 + 0];
            sdata[tid * 3 + 1] += sdata[(tid + s) * 3 + 1];
            sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];
        }
        __syncthreads();
    }

    // 3. Write result to global memory (only thread 0)
    if (threadIdx.x == 0) {
        float dot_product = sdata[0];
        float norm_query = sqrtf(sdata[1]);
        float norm_vector = sqrtf(sdata[2]);
        
        // Cosine similarity = dot_product / (norm_query * norm_vector)
        // Cosine distance = 1 - cosine_similarity
        float cosine_similarity = dot_product / (norm_query * norm_vector + 1e-8f);
        distances[row] = 1.0f - cosine_similarity;
    }
}
