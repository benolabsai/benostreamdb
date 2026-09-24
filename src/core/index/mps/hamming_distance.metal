#include <metal_stdlib>
using namespace metal;

kernel void hamming_distance_kernel(
    device const float* query [[ buffer(0) ]],
    device const float* vectors [[ buffer(1) ]],
    device float* distances [[ buffer(2) ]],
    constant uint& dim [[ buffer(3) ]],
    constant uint& n_vectors [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    // Each thread handles one vector (row). The dispatch rounds the thread
    // count up to a multiple of the threadgroup size, so guard the tail.
    uint row = id;
    if (row >= n_vectors) return;
    
    // Calculate pointer to the start of the current vector
    device const float* current_vector = vectors + row * dim;
    
    // Compute Hamming distance (count of differing elements). Exact compare to
    // match the CPU/CUDA definition.
    float count = 0.0;
    
    for (uint i = 0; i < dim; i++) {
        if (query[i] != current_vector[i]) {
            count += 1.0;
        }
    }
    
    distances[row] = count;
}
