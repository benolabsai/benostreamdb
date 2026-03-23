#include <metal_stdlib>
using namespace metal;

kernel void l1_distance_kernel(
    device const float* query [[ buffer(0) ]],
    device const float* vectors [[ buffer(1) ]],
    device float* distances [[ buffer(2) ]],
    constant uint& dim [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    // Each thread handles one vector (row)
    uint row = id;
    
    // Calculate pointer to the start of the current vector
    device const float* current_vector = vectors + row * dim;
    
    // Compute L1 distance (Manhattan distance)
    float sum = 0.0;
    
    for (uint i = 0; i < dim; i++) {
        float diff = query[i] - current_vector[i];
        sum += abs(diff);
    }
    
    distances[row] = sum;
}
