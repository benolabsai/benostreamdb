extern "C" __global__ void inner_product_kernel(
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
    // Size should be equal to blockDim.x * sizeof(float)
    extern __shared__ float sdata[];

    // 1. Thread-local partial sum for dot product
    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_dot += query[i] * current_vector[i];
    }
    sdata[threadIdx.x] = local_dot;
    __syncthreads();

    // 2. Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // 3. Write result to global memory (only thread 0)
    // Inner product is just the dot product (no normalization)
    if (threadIdx.x == 0) {
        distances[row] = sdata[0];
    }
}
