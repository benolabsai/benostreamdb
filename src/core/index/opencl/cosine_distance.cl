__kernel void cosine_distance_kernel(
    __global const float* query,
    __global const float* vectors,
    __global float* distances,
    const int dim
) {
    int row = get_global_id(0);
    
    // Pointer to current vector
    __global const float* current_vector = vectors + row * dim;
    
    float dot_product = 0.0f;
    float norm_query = 0.0f;
    float norm_vector = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        float q = query[i];
        float v = current_vector[i];
        dot_product += q * v;
        norm_query += q * q;
        norm_vector += v * v;
    }
    
    // Compute cosine similarity
    float similarity = dot_product / (sqrt(norm_query) * sqrt(norm_vector));
    
    // Convert to distance (1 - similarity)
    distances[row] = 1.0f - similarity;
}
