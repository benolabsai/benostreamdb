#include <metal_stdlib>
using namespace metal;

kernel void l2_distance_kernel(
    device const float* query [[ buffer(0) ]],
    device const float* vectors [[ buffer(1) ]],
    device float* distances [[ buffer(2) ]],
    constant uint& dim [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    // Each thread handles one vector (row)
    uint row = id;
    
    // Calculate pointer to the start of the current vector
    // vectors flat array: [v0_0, v0_1, ..., v1_0, v1_1, ...]
    device const float* current_vector = vectors + row * dim;
    
    float sum = 0.0;
    
    // Unrolled loop for generic dimensions? Compiler often handles this.
    for (uint i = 0; i < dim; i++) {
        float diff = query[i] - current_vector[i];
        sum += diff * diff;
    }
    
    distances[row] = sqrt(sum);
}
