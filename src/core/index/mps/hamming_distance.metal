#include <metal_stdlib>
using namespace metal;

kernel void hamming_distance_kernel(
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
    
    // Compute Hamming distance (count of differing elements)
    float count = 0.0;
    
    for (uint i = 0; i < dim; i++) {
        // For floating point vectors, count non-equal elements
        // Using a small epsilon for floating point comparison
        if (abs(query[i] - current_vector[i]) > 1e-7) {
            count += 1.0;
        }
    }
    
    distances[row] = count;
}
