extern "C" __global__ void jaccard_distance_kernel(
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
    // Size should be equal to blockDim.x * 2 * sizeof(float)
    // We need to store: intersection_count, union_count
    extern __shared__ float sdata[];

    // 1. Thread-local partial counts for intersection and union
    float local_intersection = 0.0f;
    float local_union = 0.0f;
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float q = query[i];
        float v = current_vector[i];
        
        // Check if at least one element is non-zero (union)
        if (q > 0.0f || v > 0.0f) {
            local_union += 1.0f;
            
            // Check if both are equal and non-zero (intersection)
            if (q == v && q > 0.0f) {
                local_intersection += 1.0f;
            }
        }
    }
    
    // Store in shared memory (interleaved for better memory access)
    int tid = threadIdx.x;
    sdata[tid * 2 + 0] = local_intersection;
    sdata[tid * 2 + 1] = local_union;
    __syncthreads();

    // 2. Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[tid * 2 + 0] += sdata[(tid + s) * 2 + 0];
            sdata[tid * 2 + 1] += sdata[(tid + s) * 2 + 1];
        }
        __syncthreads();
    }

    // 3. Write result to global memory (only thread 0)
    if (threadIdx.x == 0) {
        float intersection = sdata[0];
        float union_count = sdata[1];
        
        // Jaccard distance = 1 - (intersection / union)
        // Handle edge case where union is 0
        float jaccard_distance = (union_count > 0.0f) ? (1.0f - (intersection / union_count)) : 0.0f;
        distances[row] = jaccard_distance;
    }
}
