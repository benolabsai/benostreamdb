extern "C" __global__ void jaccard_packed_kernel(
    const unsigned char* __restrict__ query,
    const unsigned char* __restrict__ vectors,
    float* __restrict__ distances,
    int dim_bytes,
    int n_vectors
) {
    // Each block handles one packed vector (row); threads reduce intersection
    // and union bit counts. Shared memory is interleaved (2 slots per thread).
    int row = blockIdx.x;
    if (row >= n_vectors) return;

    const unsigned char* v = vectors + (long)row * dim_bytes;

    extern __shared__ float sdata[];

    float local_inter = 0.0f;
    float local_union = 0.0f;
    for (int i = threadIdx.x; i < dim_bytes; i += blockDim.x) {
        unsigned int q = (unsigned int)query[i];
        unsigned int w = (unsigned int)v[i];
        local_inter += (float)__popc(q & w);
        local_union += (float)__popc(q | w);
    }
    sdata[threadIdx.x * 2 + 0] = local_inter;
    sdata[threadIdx.x * 2 + 1] = local_union;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x * 2 + 0] += sdata[(threadIdx.x + s) * 2 + 0];
            sdata[threadIdx.x * 2 + 1] += sdata[(threadIdx.x + s) * 2 + 1];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float inter = sdata[0];
        float uni = sdata[1];
        distances[row] = (uni == 0.0f) ? 0.0f : 1.0f - (inter / uni);
    }
}
