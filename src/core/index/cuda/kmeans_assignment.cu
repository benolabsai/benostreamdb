extern "C" __global__ void kmeans_assignment(
    const float* vectors,     // [batch_size, dim]
    const float* centroids,   // [k, dim]
    unsigned int* labels,     // [batch_size]
    unsigned int batch_size,
    unsigned int k,
    unsigned int dim
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch_size) return;

    float min_dist = 1e38f;
    unsigned int best_idx = 0;

    const float* vec = &vectors[id * dim];

    for (unsigned int i = 0; i < k; i++) {
        const float* centroid = &centroids[i * dim];
        float dist = 0.0f;
        
        for (unsigned int j = 0; j < dim; j++) {
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
