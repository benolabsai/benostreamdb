// Packed-binary distance (Hamming / Jaccard) over u32 words.
//
// The host bit-packs bytes into little-endian u32 words, zero-padded to a
// 4-byte multiple, so `dim_words = ceil(dim_bytes / 4)`.

@group(0) @binding(0) var<storage, read> query: array<u32>;
@group(0) @binding(1) var<storage, read> vectors: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Config {
    dim_words: u32,
    num_vectors: u32,
    metric_type: u32, // 0 = Hamming, 1 = Jaccard
    _pad: u32,
}
@group(0) @binding(3) var<uniform> config: Config;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= config.num_vectors) {
        return;
    }

    let dim_words = config.dim_words;
    let offset = idx * dim_words;

    var ham = 0u;
    var inter = 0u;
    var uni = 0u;
    for (var i = 0u; i < dim_words; i++) {
        let q = query[i];
        let v = vectors[offset + i];
        ham += countOneBits(q ^ v);
        inter += countOneBits(q & v);
        uni += countOneBits(q | v);
    }

    if (config.metric_type == 0u) {
        output[idx] = f32(ham);
    } else if (uni == 0u) {
        output[idx] = 0.0;
    } else {
        output[idx] = 1.0 - f32(inter) / f32(uni);
    }
}
