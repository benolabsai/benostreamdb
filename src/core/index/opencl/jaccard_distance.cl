__kernel void jaccard_distance_kernel(
    __global const float* query,
    __global const float* vectors,
    __global float* distances,
    const int dim
) {
    int row = get_global_id(0);
    
    // Pointer to current vector
    __global const float* current_vector = vectors + row * dim;
    
    int intersection = 0;
    int union_count = 0;
    
    for (int i = 0; i < dim; i++) {
        // For binary vectors represented as floats (0.0 or 1.0)
        int q_bit = (query[i] != 0.0f) ? 1 : 0;
        int v_bit = (current_vector[i] != 0.0f) ? 1 : 0;
        
        // Intersection: both are 1
        if (q_bit && v_bit) {
            intersection++;
        }
        
        // Union: at least one is 1
        if (q_bit || v_bit) {
            union_count++;
        }
    }
    
    // Jaccard distance = 1 - (intersection / union)
    // Handle edge case where both vectors are all zeros
    float jaccard_similarity = (union_count > 0) ? ((float)intersection / (float)union_count) : 1.0f;
    distances[row] = 1.0f - jaccard_similarity;
}
