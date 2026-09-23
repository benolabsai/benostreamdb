#include <metal_stdlib>
using namespace metal;

// Packed-binary Jaccard distance: one thread per packed vector (row), counting
// intersection/union bits with popcount. Mirrors the CUDA/WGPU packed kernels.
kernel void jaccard_packed_kernel(
    device const uchar* query [[ buffer(0) ]],
    device const uchar* vectors [[ buffer(1) ]],
    device float* distances [[ buffer(2) ]],
    constant uint& dim_bytes [[ buffer(3) ]],
    constant uint& n_vectors [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    // The dispatch rounds the thread count up to a multiple of the threadgroup
    // size, so guard the tail.
    uint row = id;
    if (row >= n_vectors) return;

    device const uchar* v = vectors + row * dim_bytes;

    uint inter = 0;
    uint uni = 0;
    for (uint i = 0; i < dim_bytes; i++) {
        uint q = (uint)query[i];
        uint w = (uint)v[i];
        inter += popcount(q & w);
        uni += popcount(q | w);
    }
    distances[row] = (uni == 0) ? 0.0f : 1.0f - (float(inter) / float(uni));
}
