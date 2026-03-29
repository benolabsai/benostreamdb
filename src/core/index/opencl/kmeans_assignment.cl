__kernel void kmeans_assignment(
    __global const float* vectors,     // [batch_size, dim]
    __global const float* centroids,   // [k, dim]
    __global uint* labels,             // [batch_size]
    const uint batch_size,
    const uint k_val,
    const uint dim
) {
    uint id = get_global_id(0);
    if (id >= batch_size) return;

    float min_dist = 3.402823466e+38F;
    uint best_idx = 0;

    __global const float* vec = &vectors[id * dim];

    for (uint i = 0; i < k_val; i++) {
        __global const float* centroid = &centroids[i * dim];
        float dist = 0.0f;
        
        for (uint j = 0; j < dim; j++) {
            float diff = vec[j] - centroid[j];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }

    labels[id] = best_idx;
}
