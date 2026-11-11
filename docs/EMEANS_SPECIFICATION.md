# High-Throughput Vector Indexing Suite: Specification & Working Plan (v0.16.0)

**Document Status:** Approved Architecture Plan  
**Target Release:** BenoStreamDB `v0.16.0`  
**Tracking Issue:** [benolabsai/benostreamdb#124](https://github.com/benolabsai/benostreamdb/issues/124)  
**Feature Branch:** `feature/e-means`  
**Author:** Richard Albright (BenoLabs)  

---

## 1. Executive Summary & Problem Definition

In high-dimensional vector search and hybrid graph-vector database systems, vector indexing throughput is bottlenecked by three distinct algorithmic stages:
1. **The Coarse Quantizer Bottleneck (Index Build):** Clustering $N$ vectors into $k$ Voronoi partitions via classical Lloyd's algorithm requires 20 to 25 full-batch sweeps over dataset vectors. Truncating Lloyd to 5–10 iterations avoids prohibitive build times but degrades cluster boundaries and search recall.
2. **The Inverted List Dynamic Range Problem (Fine Quantization):** Inverted lists storing raw vectors $x \in \mathbb{R}^d$ suffer from large dynamic range spread. Storing vectors relative to their coarse partition centroids as residuals ($r = x - c_j$) compresses dynamic range by 70–85%, enabling tighter quantization bins with lower reconstruction distortion.
3. **The Inverted List Memory Bandwidth Bottleneck (Query Scan):** During search, scanning vectors in probed Voronoi buckets incurs high memory bandwidth and cache latency. Traversal using standard gather instructions or uncompressed floats is bound by memory throughput.

To resolve these bottlenecks in `v0.16.0`, BenoStreamDB integrates a unified high-throughput vector suite:
* **e-means Coarse Quantization:** Meta FAIR mini-batch EMA coarse quantizer achieving true convergence in 1.2–2.5 epochs (~10× wall-clock build acceleration).
* **Residual Inverted List Encoding:** Centering partition vectors around coarse centroids ($r = x - c_j$) with mathematical L2 distance preservation.
* **In-Register 4-Bit SIMD FastScan Engine:** Traversal of 32-vector interleaved blocks using AVX2 `_mm_shuffle_epi8` / ARM NEON `vtbl1_u8` byte shuffles directly in CPU registers without L1/L2 cache lookups.

---

## 2. Mathematical Foundation: e-means Coarse Quantizer

### 2.1 Centroid State & The Convex Hull Invariant

All variants of $k$-means maintain for each cluster $j \in \{1, \dots, k\}$:
* $S_j \in \mathbb{R}^d$: cumulative coordinate sum of vectors assigned to cluster $j$.
* $N_j \in \mathbb{R}^+$: cumulative mass/count of vectors assigned to cluster $j$.
* $c_j \in \mathbb{R}^d$: cluster centroid, defined as the ratio $c_j = S_j / N_j$.

When a mini-batch of size $B$ arrives with assigned batch sum $s_j = \sum_{x \in X_j} x$ and count $n_j = |X_j|$:
$$S_j \leftarrow (1 - \alpha) S_j + \alpha s_j$$
$$N_j \leftarrow (1 - \alpha) N_j + \alpha n_j$$
$$c_j \leftarrow \frac{S_j}{N_j}$$

#### Mathematical Consequence:
Unlike unconstrained gradient updates, the EMA ratio guarantees:
$$c_j = (1 - w_j) c_j^{\text{old}} + w_j \bar{x}_j, \quad \text{where } w_j = \frac{\alpha n_j}{(1 - \alpha) N_j + \alpha n_j} \in [0, 1]$$
Centroid $c_j$ is **strictly a convex combination** of the previous centroid and the batch mean $\bar{x}_j$. Centroids can never escape the convex hull of assigned data points.

---

### 2.2 Automatic Hyperparameter Scaling Laws

Rather than requiring manual tuning of batch size $B$ or learning rate $\alpha$, e-means derives settings directly from dataset dimensions $(N, k, T)$:

#### 1. Batch Size Power Law ($B$):
$$B = \text{round\_pow2}\Big(C_B \cdot k^{b_k} \cdot N^{b_n} \cdot T^{b_T}\Big)$$
Bounded by: $B \le \text{round\_pow2}(N / 2)$ (ensuring $\ge 2$ updates per epoch).

* **Throughput Law (Default for $d > 16$):**
  $$C_B = 0.035134, \quad b_k = 0.134749, \quad b_n = 0.876473, \quad b_T = 0.813344$$
* **Quality Law (Default for $d \le 16$):**
  $$C_B = 0.00439175, \quad b_k = 0.134749, \quad b_n = 0.876473, \quad b_T = 0.813344$$

#### 2. Initial Step Size Power Law ($\alpha_0$):
$$\alpha_0 = \min\Big(C_\alpha \cdot k^{a_k} \cdot B^{a_B} \cdot N^{a_n} \cdot T^{a_T}, \; 1.0\Big)$$
* **Coefficients:**
  $$C_\alpha = 0.8244, \quad a_k = -0.4617, \quad a_B = 0.9003, \quad a_n = -0.4498, \quad a_T = -0.1057$$
Clamped strictly to $[10^{-4}, 1.0]$.

---

### 2.3 Cosine Annealing Schedule

Over total global mini-batch steps $T_{\text{total}} = \lceil (N \cdot T) / B \rceil$, step size $\alpha(t)$ smoothly decays:
$$\alpha(t) = \alpha_0 \left(0.1 + 0.9 \cdot \frac{1 + \cos(\pi \cdot t / T_{\text{total}})}{2}\right)$$
This provides exploratory steps in early epochs to escape local minima, and fine-grained localization in late epochs.

---

### 2.4 Dead-Cluster Revival via Mass-Proportional Donor Splitting

Every $\tau = 5$ batches, clusters with mass $N_j < 0.1 \cdot \bar{N}$ are marked dead. A donor cluster $d$ is sampled with probability proportional to its mass $P(d) \propto N_d$.
* **Perturbation Split:** The donor is split into two offspring along alternating perturbation $\delta = \epsilon \cdot [1, -1, 1, -1, \dots]$ ($\epsilon = 10^{-4}$):
  $$c_{\text{donor}} \leftarrow c_d \cdot (1 - \delta), \qquad c_{\text{dead}} \leftarrow c_d \cdot (1 + \delta)$$
  $$N_{\text{donor}} \leftarrow \frac{N_d}{2}, \qquad N_{\text{dead}} \leftarrow \frac{N_d}{2}$$
This reduces dead centroids to **$<0.3\%$**, maximizing codebook utilization.

---

### 2.5 Exact Mean Finalization

On the final pass, a single exact empirical aggregation is performed:
$$c_j = \frac{1}{|X_j|} \sum_{x \in X_j} x$$
This removes stochastic variance and guarantees centroids equal the true empirical centers of their assigned partitions.

---

## 3. Residual Inverted List Encoding

For an inverted list associated with coarse centroid $c_j$, storing vectors as residuals $r = x - c_j$ provides major numerical advantages:

### 3.1 Mathematical Distance Identity (L2 Metric)
For Euclidean squared distance between query $q$ and indexed vector $x$:
$$\|q - x\|^2 = \|(q - c_j) - (x - c_j)\|^2 = \|q_r - r\|^2$$
where $q_r = q - c_j$ is the **query residual**.

#### Operational Advantage:
* The query residual $q_r = q - c_j$ is computed **only once per probed cluster**, rather than for every vector in the inverted list.
* Fine scanning evaluates $\|q_r - r\|^2$, which is mathematically bit-exact identical to full Euclidean distance $\|q - x\|^2$.
* Coordinate magnitudes drop by $70\%\text{--}85\%$, concentrating values near zero and dramatically reducing quantization distortion under scalar or product quantizers.

---

## 4. In-Register SIMD FastScan Engine

FastScan is an in-register vector look-up engine operating on 4-bit subvector codes ($k = 16$).

### 4.1 Register Saturation Mechanics
* A 128-bit SIMD register (`__m128i` on x86, `uint8x16_t` on ARM NEON) holds exactly 16 bytes.
* A 4-bit quantizer divides vector dimension $d$ into $M$ subvectors ($d_{\text{sub}} = d / M$). Each subvector has 16 centroids.
* For query $q$, the distances to all 16 centroids of subvector $m$ are quantized to fixed-point `u8` and loaded into **a single 16-byte register**.

### 4.2 32-Vector Interleaved Block Packing (`FASTSCAN_BLOCK_SIZE = 32`)
Vectors inside an inverted list are partitioned into contiguous blocks of 32 vectors. For each subvector $m$:
* 16 bytes store 32 4-bit codes:
  * Low nibble of byte $i \in 0..15$: vector $i$ (vectors $0..15$).
  * High nibble of byte $i \in 0..15$: vector $i + 16$ (vectors $16..31$).
* Total block size = $M \times 16$ bytes (e.g. for $M = 32$ subvectors, exactly 512 bytes = 8 cache lines).

### 4.3 AVX2 / ARM NEON In-Register Evaluation
In each iteration over subvector $m$:
1. Load 16-byte query distance LUT: `lut_128`.
2. Load 16 bytes of codes: `codes_128`.
3. Extract nibbles: `low_nibbles = codes_128 & 0x0F`, `high_nibbles = (codes_128 >> 4) & 0x0F`.
4. In-register byte shuffle:
   $$\text{dist\_low} = \text{\_mm\_shuffle\_epi8}(\text{lut\_128}, \text{low\_nibbles})$$
   $$\text{dist\_high} = \text{\_mm\_shuffle\_epi8}(\text{lut\_128}, \text{high\_nibbles})$$
5. Unpack to 16-bit integers and accumulate in registers:
   $$\text{acc}_{0..7} \leftarrow \text{acc}_{0..7} + \text{unpack\_lo}(\text{dist\_low})$$
   $$\text{acc}_{8..15} \leftarrow \text{acc}_{8..15} + \text{unpack\_hi}(\text{dist\_low})$$
   $$\text{acc}_{16..23} \leftarrow \text{acc}_{16..23} + \text{unpack\_lo}(\text{dist\_high})$$
   $$\text{acc}_{24..31} \leftarrow \text{acc}_{24..31} + \text{unpack\_hi}(\text{dist\_high})$$

This executes in **1 CPU cycle per subvector for 32 vector distances**, operating at $>100\text{M}$ vector distances per second per core with zero L1/L2 cache accesses during accumulation.

---

## 5. Compatibility Matrix Across BenoStreamDB Compression Formats

| Compression Format | Bit Depth | FastScan Compatibility | Mechanics & Rationale |
| :--- | :---: | :---: | :--- |
| **BenoQuant™ 4-bit (`HnswTq4`)** | 4-bit | **Native / Optimal** | Isotropic FWHT coordinates quantize to 16 uniform bins. Precomputed 16-byte coordinate LUT evaluates 32 vector distances per shuffle. |
| **Product Quantization 4-bit** | 4-bit | **Native / Optimal** | $M$ subvectors with $k=16$ centroids. Query LUT fits in one 128-bit register per subvector. |
| **Product Quantization 8-bit** | 8-bit | **Incompatible with 1-reg FastScan** | 256 centroids require 256 bytes (16 registers), exceeding single shuffle capacity. SDC/ADC gather remains standard. |
| **Scalar Quantization (SQ4)** | 4-bit | **Native** | 16 coordinate bins per dimension evaluated via 16-byte LUTs. |
| **Scalar Quantization (SQ8)** | 8-bit | **Superseded by SIMD MAD** | 8-bit integers are evaluated faster via direct AVX2 integer multiply-add (`_mm256_maddubs_epi16`) than table lookups. |
| **IVF Inverted Lists (`IvfFastScan`)** | Any | **4x – 6x Speedup** | Contiguous 32-vector block streaming memory layout saturates memory bus. |

---

## 6. Architecture & Deliverables

```
benostreamdb/
├── benostream-gpu-ann/               # Bumped to v0.2.0
│   ├── src/
│   │   ├── e_means.rs                # e-means engine (Meta FAIR scaling laws, donor splits)
│   │   ├── fastscan.rs               # In-register SIMD 4-bit FastScan codebook & AVX2 kernels
│   │   ├── ivf_fastscan.rs           # High-throughput IvfFastScanIndex with 32-vector blocks
│   │   ├── ivf_flat.rs               # Residual inverted list encoding (r = x - c)
│   │   ├── kmeans.rs                 # Gated dispatch to e-means
│   │   └── lib.rs                    # Exported Algorithm::IvfFastScan & VectorIndex variants
│   └── tests/
│       └── test_ivf_fastscan.rs      # Integration tests for IVF FastScan
├── src/core/index/
│   ├── e_means.rs                    # Database engine re-export and context wrapper
│   ├── ivf.rs                        # Gated simple_kmeans_with_seed routing + residual IVF
│   ├── global_centroids.rs           # Table-level global centroid training via e-means
│   └── hnsw_ivf.rs                   # Coarse clustering dispatch in HnswIvfIndex
├── fuzz/
│   └── fuzz_targets/
│       └── e_means.rs                # Fuzz target for degenerate inputs and numerical stability
└── docs/
    ├── EMEANS_SPECIFICATION.md       # This unified architecture specification
    └── ROADMAP.md                    # Theme 12 updated for v0.16.0 release suite
```

---

## 7. Verification & Acceptance Criteria for v0.16.0

1. **Pre-Release Check**: Must pass `./scripts/pre_release_check.sh` with zero warnings, zero clippy lints, zero broken docs, and 100% test pass.
2. **Determinism**: Given an identical PRNG seed, e-means and FastScan produce bit-exact identical centroids and block codes across repeated runs.
3. **SIMD Bit-Exact Parity**: AVX2 `scan_block_avx2` and scalar `scan_block_scalar` must produce bit-exact integer results on all test suites.
4. **Fuzz Invariant**: Fuzz target must run $\ge 100,000$ iterations with zero panics or NaNs on adversarial inputs.
5. **Attribution Preservation**: All source files adapting Meta FAIR algorithms (`e_means.rs`, `fastscan.rs`) maintain MIT License notices.
