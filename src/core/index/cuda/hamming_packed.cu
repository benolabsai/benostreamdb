extern "C" __global__ void hamming_packed_kernel(
    const unsigned char* __restrict__ query,
    const unsigned char* __restrict__ vectors,
    float* __restrict__ distances,
    int dim_bytes,
    int n_vectors
) {
    // Each block handles one packed vector (row); threads reduce over bytes.
    int row = blockIdx.x;
    if (row >= n_vectors) return;

    const unsigned char* v = vectors + (long)row * dim_bytes;

    extern __shared__ float sdata[];

    float local = 0.0f;
    for (int i = threadIdx.x; i < dim_bytes; i += blockDim.x) {
        local += (float)__popc((unsigned int)(query[i] ^ v[i]));
    }
    sdata[threadIdx.x] = local;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[row] = sdata[0];
}
