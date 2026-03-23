__kernel void hamming_distance_kernel(
    __global const float* query,
    __global const float* vectors,
    __global float* distances,
    const int dim
) {
    int row = get_global_id(0);
    
    // Pointer to current vector
    __global const float* current_vector = vectors + row * dim;
    
    int hamming_dist = 0;
    
    for (int i = 0; i < dim; i++) {
        // For binary vectors represented as floats (0.0 or 1.0)
        // XOR operation: different values contribute 1 to distance
        if (query[i] != current_vector[i]) {
            hamming_dist++;
        }
    }
    
    distances[row] = (float)hamming_dist;
}
