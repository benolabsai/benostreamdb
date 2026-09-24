#include <metal_stdlib>
using namespace metal;

// K-Means Assignment Kernel (MPS/Metal)
// This kernel calculates the nearest centroid for each vector in a batch.
// 
// threads_per_grid: number of vectors in the batch
kernel void kmeans_assignment(
    const device float* vectors     [[ buffer(0) ]], // [batch_size, dim]
    const device float* centroids   [[ buffer(1) ]], // [k, dim]
    device uint* labels             [[ buffer(2) ]], // [batch_size]
    constant uint& batch_size       [[ buffer(3) ]],
    constant uint& k                [[ buffer(4) ]],
    constant uint& dim              [[ buffer(5) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= batch_size) return;

    float min_dist = 1e38; // Float max
    uint best_idx = 0;

    const device float* vec = &vectors[id * dim];

    for (uint i = 0; i < k; i++) {
        const device float* centroid = &centroids[i * dim];
        float dist = 0.0;
        
        // L2 distance squared
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
