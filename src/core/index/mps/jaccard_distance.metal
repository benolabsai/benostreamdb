#include <metal_stdlib>
using namespace metal;

kernel void jaccard_distance_kernel(
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
    
    // Compute Jaccard distance for binary/set-like vectors. Membership is
    // "> 0.0" to match the CPU/CUDA definition.
    // Jaccard similarity = |A ∩ B| / |A ∪ B|
    // Jaccard distance = 1 - Jaccard similarity
    
    float intersection = 0.0;
    float union_count = 0.0;
    
    for (uint i = 0; i < dim; i++) {
        float q = query[i];
        float v = current_vector[i];
        
        if (q > 0.0 || v > 0.0) {
            if (q == v && q > 0.0) {
                intersection += 1.0;
            }
            union_count += 1.0;
        }
    }
    
    // Compute Jaccard distance
    float jaccard_similarity = (union_count > 0.0) ? (intersection / union_count) : 0.0;
    distances[row] = 1.0 - jaccard_similarity;
}
