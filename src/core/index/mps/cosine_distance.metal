#include <metal_stdlib>
using namespace metal;

kernel void cosine_distance_kernel(
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
    
    // Compute dot product and norms
    float dot_product = 0.0;
    float norm_query = 0.0;
    float norm_vector = 0.0;
    
    for (uint i = 0; i < dim; i++) {
        float q = query[i];
        float v = current_vector[i];
        dot_product += q * v;
        norm_query += q * q;
        norm_vector += v * v;
    }
    
    // Compute cosine similarity and distance
    norm_query = sqrt(norm_query);
    norm_vector = sqrt(norm_vector);
    
    // Cosine similarity = dot_product / (norm_query * norm_vector)
    // Cosine distance = 1 - cosine_similarity
    float cosine_similarity = dot_product / (norm_query * norm_vector + 1e-8);
    distances[row] = 1.0 - cosine_similarity;
}
