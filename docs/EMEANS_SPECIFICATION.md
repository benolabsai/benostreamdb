# E-Means High-Throughput Coarse Quantization: Specification & Working Plan (v0.16.0)

**Document Status:** Approved Architecture Plan  
**Target Release:** BenoStreamDB `v0.16.0`  
**Tracking Issue:** [benolabsai/benostreamdb#124](https://github.com/benolabsai/benostreamdb/issues/124)  
**Feature Branch:** `feature/e-means`  
**Author:** Richard Albright (BenoLabs)  

---

## 1. Executive Summary & Problem Definition

In vector search and graph-vector hybrid systems, the coarse quantizer (clustering $N$ vectors into $k$ Voronoi partitions) is the primary computational and memory bottleneck during index construction:
* **The Lloyd Bottleneck:** Classical Lloyd's algorithm evaluates pairwise distances between all $N$ dataset vectors and all $k$ centroids in full-batch passes. Reaching true convergence typically requires **20 to 25 epochs (full sweeps over the dataset)**.
* **The Truncation Compromise:** Because 25 passes over out-of-core Parquet vector tables is prohibitively slow, current production implementations (including `src/core/index/ivf.rs`) truncate Lloyd to 5–10 iterations. Truncated Lloyd's produces suboptimal Voronoi boundaries, directly hurting retrieval recall at query time.
* **The e-means Breakthrough:** Developed by Meta FAIR (*"e-means: outperforming k-means in one-tenth the epochs"*, Bessa et al., arXiv:2611.09501, Nov 2026), e-means introduces a stochastic mini-batch Exponential Moving Average (EMA) update rule coupled with empirical scaling laws. It reaches lower test loss than 25 iterations of Lloyd in **1.2 to 2.5 epochs**, delivering an **8–12× reduction in memory passes and a ~10× wall-clock acceleration** while improving codebook quality.

---

## 2. Mathematical Foundation & Algorithmic Mechanisms

### 2.1 Centroid State & The Convex Hull Invariant

All variants of $k$-means maintain for each cluster $j \in \{1, \dots, k\}$:
* $S_j \in \mathbb{R}^d$: the cumulative coordinate sum of vectors assigned to cluster $j$.
* $N_j \in \mathbb{R}^+$: the cumulative mass/count of vectors assigned to cluster $j$.
* $c_j \in \mathbb{R}^d$: the cluster centroid, defined as the ratio:
  $$c_j = \frac{S_j}{N_j}$$

When a mini-batch of size $B$ arrives with assigned batch sum $s_j = \sum_{x \in X_j} x$ and count $n_j = |X_j|$:
$$S_j \leftarrow (1 - \alpha) S_j + \alpha s_j$$
$$N_j \leftarrow (1 - \alpha) N_j + \alpha n_j$$
$$c_j \leftarrow \frac{S_j}{N_j}$$

#### Mathematical Consequence:
Unlike Stochastic Gradient Descent (SGD) or Adam (which apply unconstrained displacement vectors $\Delta c = -\eta \nabla \mathcal{L}$ that frequently overshoot cluster boundaries), the EMA ratio representation guarantees:
$$c_j = (1 - w_j) c_j^{\text{old}} + w_j \bar{x}_j, \quad \text{where } w_j = \frac{\alpha n_j}{(1 - \alpha) N_j + \alpha n_j} \in [0, 1]$$
Centroid $c_j$ is **strictly a convex combination** of the previous centroid and the batch mean $\bar{x}_j$. Centroids can never escape the convex hull of assigned data points.

---

### 2.2 Automatic Hyperparameter Scaling Laws

Rather than requiring manual tuning of batch size $B$ or learning rate $\alpha$, e-means derives optimal settings directly from dataset dimensions $(N, k, T)$:

#### 1. Batch Size Power Law ($B$):
$$B = \text{round\_pow2}\Big(C_B \cdot k^{b_k} \cdot N^{b_n} \cdot T^{b_T}\Big)$$
Subject to upper bound: $B \le \text{round\_pow2}(N / 2)$ (guaranteeing $\ge 2$ updates per epoch).

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
This provides large exploratory steps in early epochs to escape local minima, and fine-grained localization in late epochs.

---

### 2.4 Dead-Cluster Revival via Mass-Proportional Donor Splitting

Mini-batch $k$-means frequently produces "dead" clusters (centroids receiving zero points due to stochastic batch sampling). 
* **The Failure of Prior Work (EnCodec):** Resampling dead centroids uniformly from the incoming batch leaves 4–14% dead clusters and high cluster size imbalance.
* **Mass-Proportional Splitting:** Every $\tau = 5$ batches, clusters with mass $N_j < 0.1 \cdot \bar{N}$ are marked dead. A donor cluster $d$ is sampled with probability proportional to its mass $P(d) \propto N_d$.
* **Perturbation Split:** The donor is split into two offspring along an alternating coordinate perturbation vector $\delta$:
  $$\delta = \epsilon \cdot [1, -1, 1, -1, \dots], \quad \epsilon = 10^{-4}$$
  $$c_{\text{donor}} \leftarrow c_d \cdot (1 - \delta), \qquad c_{\text{dead}} \leftarrow c_d \cdot (1 + \delta)$$
  $$N_{\text{donor}} \leftarrow \frac{N_d}{2}, \qquad N_{\text{dead}} \leftarrow \frac{N_d}{2}$$
This reduces dead centroids to **$<0.3\%$**, maximizing codebook capacity.

---

### 2.5 Exact Mean Finalization

On the final epoch (or closing fractional pass), the algorithm executes a single exact aggregation:
$$c_j = \frac{1}{|X_j|} \sum_{x \in X_j} x$$
This removes any residual stochastic variance from the EMA update and guarantees that final centroids equal the true empirical centers of their assigned partitions.

---

## 3. Activation & Gating Matrix

e-means must **only** be activated when statistical density guarantees sufficient updates and stable mini-batch draws:

| Dimension / Metric | Condition | Action | Rationale |
| :--- | :--- | :--- | :--- |
| **Dataset Size ($N$)** | $N < 10,000$ | **Fallback to Lloyd** | Lloyd runs in $<50\text{ms}$. Mini-batching yields $<15$ total steps, causing under-convergence. |
| **Cluster Count ($k$)** | $k < 64$ | **Fallback to Lloyd** | Low cluster counts converge rapidly under full-batch Lloyd. |
| **Density Ratio ($N/k$)** | $N/k < 30$ | **Fallback to Lloyd** | Sparse clusters starve mini-batches, triggering excessive dead-cluster revivals. |
| **Vector Dimension ($d$)** | $d \le 2$ | **Disable Revival** | In $d \le 2$ (e.g. geospatial lat/long), revival causes dimensional collapse and increases error by 7–10%. |
| **Subvector PQ ($k=256$)** | Product Quantization | **Lloyd (5 iters)** | Subspace codebook learning ($d=8..32$) already takes milliseconds; no mini-batch gain. |
| **IVF Coarse Quantizer** | $N \ge 10\text{k}, k \ge 64$ | **e-means Active** | Primary target: 10× speedup on coarse Voronoi partitioning. |

---

## 4. Source Code Attribution & License Compliance

Portions of the mathematical formulas, scaling law parameters, and revival mechanisms in BenoStreamDB are adapted from Meta FAIR / Faiss (`contrib/e_means`).

### Mandatory Source Header Requirement
Every source file implementing or directly adapting these formulas (specifically [`src/core/index/e_means.rs`](file:///home/ralbright/projects/benostreamdb/src/core/index/e_means.rs) and related tests) must carry the following attribution block:

```rust
// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
// Portions of this algorithm, lifecycle orchestration, and empirical scaling law parameters
// are adapted from:
//
//   "e-means: outperforming k-means in one-tenth the epochs"
//   Swann Bessa, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Hugo Touvron, David Picard,
//   Matthijs Douze, Hervé Jégou
//   Meta FAIR (2026), arXiv:2611.09501 (https://arxiv.org/abs/2611.09501)
//
// Reference implementation in Faiss (contrib/e_means):
//   Copyright (c) Meta Platforms, Inc. and affiliates.
//   Licensed under the MIT License (https://opensource.org/licenses/MIT)
```

### Citation Clarification
* In academic literature on mini-batch clustering, Bessa et al. cite *"Staleness-Reduction Mini-Batch K-Means"* (IEEE TNNLS, 2024). Note that the primary author is **Xueying Zhu** (Zhejiang University, working with Zeke Wang), not **Xingquan Zhu** (Professor at Florida Atlantic University). Both publish in IEEE TNNLS.

---

## 5. Implementation Architecture & Deliverables

```
benostreamdb/
├── src/core/index/
│   ├── e_means.rs            # Core e-means engine: EMA state, scaling laws, donor splits
│   ├── ivf.rs                # Gated dispatch: routes N >= 10k, k >= 64 to e-means
│   ├── gpu.rs                # GPU batch-chunked kmeans assignment integration
│   └── mod.rs                # Module export
├── fuzz/
│   └── fuzz_targets/
│       └── e_means.rs        # Fuzz testing for degenerate vectors, NaNs, zero norms
├── tests/
│   └── test_e_means.rs       # Precision/recall oracle against classical Lloyd
└── docs/
    └── ROADMAP.md            # Theme 12 entry tracking v0.16.0 milestones
```

---

## 6. Verification & Acceptance Criteria for v0.16.0

1. **Pre-Release Gate**: Must pass 100% of `./scripts/pre_release_check.sh` locally (zero warnings, zero clippy lints, zero broken docs, zero formatting diffs).
2. **Determinism**: Identical PRNG seeds must produce bit-exact identical centroids across repeated runs on the same platform.
3. **Fuzz Invariant**: Fuzzer must run $\ge 100,000$ iterations with zero panics or NaNs when fed arbitrary byte sequences, zero vectors, extreme norms, or colinear points.
4. **Empirical Performance Target**:
   * **Cohere-10M (768-dim, $k=16,384$):** Coarse quantizer training wall-clock time reduced by $\ge 4\times$ on CPU and $\ge 8\times$ on CUDA/Metal compared to 10 iterations of Lloyd.
   * **Recall@10:** Equal or higher search recall at $n_{\text{probe}} \in \{8, 16, 32\}$ on SIFT-1M and Cohere-10M due to true algorithmic convergence.
