__kernel void l2_distance_kernel(
    __global const float* query,
    __global const float* vectors,
    __global float* distances,
    const int dim
) {
    int row = get_global_id(0);
    
    // Pointer to current vector
    __global const float* current_vector = vectors + row * dim;
    
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = query[i] - current_vector[i];
        sum += diff * diff;
    }
    
    distances[row] = sqrt(sum);
}
