__kernel void jaccard_distance_kernel(
    __global const float* query,
    __global const float* vectors,
    __global float* distances,
    const int dim
) {
    int row = get_global_id(0);
    
    // Pointer to current vector
    __global const float* current_vector = vectors + row * dim;
    
    float intersection = 0.0f;
    float union_count = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        float q_val = query[i];
        float v_val = current_vector[i];
        
        if (q_val > 0.0f || v_val > 0.0f) {
            if (q_val == v_val && q_val > 0.0f) {
                intersection += 1.0f;
            }
            union_count += 1.0f;
        }
    }
    
    // Jaccard distance = 1 - (intersection / union)
    // Handle edge case where both vectors are all zeros
    if (union_count == 0.0f) {
        distances[row] = 0.0f;
    } else {
        distances[row] = 1.0f - (intersection / union_count);
    }
}
