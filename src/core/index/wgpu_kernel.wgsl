@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> vectors: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Config {
    dim: u32,
    num_vectors: u32,
    metric_type: u32,
}
@group(0) @binding(3) var<uniform> config: Config;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= config.num_vectors) {
        return;
    }
    
    let dim = config.dim;
    let offset = idx * dim;
    
    var dist = 0.0;
    let mt = config.metric_type;
    
    if (mt == 0u) {
        // L2 Distance
        for (var i = 0u; i < dim; i++) {
            let diff = query[i] - vectors[offset + i];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        
    } else if (mt == 1u) {
        // Inner Product
        for (var i = 0u; i < dim; i++) {
            dist += query[i] * vectors[offset + i];
        }
        
    } else if (mt == 2u) {
        // Cosine Distance
        var dot = 0.0;
        var norm_q = 0.0;
        var norm_v = 0.0;
        for (var i = 0u; i < dim; i++) {
            let q = query[i];
            let v = vectors[offset + i];
            dot += q * v;
            norm_q += q * q;
            norm_v += v * v;
        }
        if (norm_q == 0.0 || norm_v == 0.0) {
            dist = 1.0;
        } else {
            let cos_sim = dot / (sqrt(norm_q) * sqrt(norm_v));
            dist = 1.0 - cos_sim;
        }
        
    } else if (mt == 3u) {
        // L1 Distance
        for (var i = 0u; i < dim; i++) {
            dist += abs(query[i] - vectors[offset + i]);
        }
        
    } else if (mt == 4u) {
        // Hamming Distance (float comparison)
        for (var i = 0u; i < dim; i++) {
            if (query[i] != vectors[offset + i]) {
                dist += 1.0;
            }
        }
        
    } else if (mt == 5u) {
        // Jaccard Distance
        var intersection = 0.0;
        var union_count = 0.0;
        for (var i = 0u; i < dim; i++) {
            let q = query[i];
            let v = vectors[offset + i];
            if (q == 1.0 && v == 1.0) {
                intersection += 1.0;
            }
            if (q == 1.0 || v == 1.0) {
                union_count += 1.0;
            }
        }
        if (union_count == 0.0) {
            dist = 0.0;
        } else {
            dist = 1.0 - (intersection / union_count);
        }
    }
    
    output[idx] = dist;
}
