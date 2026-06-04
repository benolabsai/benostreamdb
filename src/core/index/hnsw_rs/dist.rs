//! Some standard distances as L1, L2, Cosine, Jaccard, Hamming
//! and a structure to enable the user to implement its own distances.
//! This implementation uses internal optimized distance functions where possible.

use crate::core::index::distance;

/// The trait describing distance.
pub trait Distance<T: Send + Sync> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32;
}

/// Jeffreys divergence
#[derive(Default, Clone, Copy)]
pub struct DistJeffreys;
impl Distance<f32> for DistJeffreys {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        let mut dist = 0.0;
        for i in 0..va.len() {
            let a = va[i].max(1.0e-30);
            let b = vb[i].max(1.0e-30);
            dist += (a - b) * (a / b).ln();
        }
        dist
    }
}

/// Special forbidden computation distance.
#[derive(Default, Clone, Copy)]
pub struct NoDist;

impl<T: Send + Sync> Distance<T> for NoDist {
    fn eval(&self, _va: &[T], _vb: &[T]) -> f32 {
        log::error!("panic error : cannot call eval on NoDist");
        panic!("cannot call distance with NoDist");
    }
}

/// L1 distance
#[derive(Default, Clone, Copy)]
pub struct DistL1;
impl Distance<f32> for DistL1 {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        distance::l1_distance(va, vb)
    }
}

/// L2 distance
#[derive(Default, Clone, Copy)]
pub struct DistL2;
impl Distance<f32> for DistL2 {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        distance::l2_distance_squared(va, vb).sqrt()
    }
}

/// Cosine distance
#[derive(Default, Clone, Copy)]
pub struct DistCosine;
impl Distance<f32> for DistCosine {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        distance::cosine_distance(va, vb)
    }
}

/// Dot product distance (1.0 - dot)
#[derive(Default, Clone, Copy)]
pub struct DistDot;
impl Distance<f32> for DistDot {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        (1.0 - distance::dot_product(va, vb)).max(0.0)
    }
}

/// Hamming distance
#[derive(Default, Clone, Copy)]
pub struct DistHamming;
impl Distance<f32> for DistHamming {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        distance::hamming_distance(va, vb)
    }
}

/// Jaccard distance
#[derive(Default, Clone, Copy)]
pub struct DistJaccard;
impl Distance<f32> for DistJaccard {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        distance::jaccard_distance(va, vb)
    }
}

/// Scalar fallback macros for other types
macro_rules! implement_scalar (
    ($struct:ident, $ty:ty, $op:expr) => (
        impl Distance<$ty> for $struct {
            fn eval(&self, va: &[$ty], vb: &[$ty]) -> f32 {
                va.iter().zip(vb.iter()).map($op).sum::<f32>()
            }
        }
    )
);

// DistL1 for other types
implement_scalar!(DistL1, f64, |(a, b)| (a - b).abs() as f32);
implement_scalar!(DistL1, i32, |(a, b)| (a - b).abs() as f32);
implement_scalar!(DistL1, u32, |(a, b)| (*a as i64 - *b as i64).abs() as f32);
implement_scalar!(DistL1, u16, |(a, b)| (*a as i32 - *b as i32).abs() as f32);
implement_scalar!(DistL1, u8, |(a, b)| (*a as i32 - *b as i32).abs() as f32);

// DistL2 for other types
implement_scalar!(DistL2, f64, |(a, b)| ((a - b) * (a - b)) as f32);
implement_scalar!(DistL2, i32, |(a, b)| {
    let d = (a - b) as f32;
    d * d
});
implement_scalar!(DistL2, u32, |(a, b)| {
    let d = (*a as i64 - *b as i64) as f32;
    d * d
});
implement_scalar!(DistL2, u16, |(a, b)| {
    let d = (*a as i32 - *b as i32) as f32;
    d * d
});
implement_scalar!(DistL2, u8, |(a, b)| {
    let d = (*a as i32 - *b as i32) as f32;
    d * d
});

// DistHamming for other types
implement_scalar!(DistHamming, i32, |(a, b)| if a != b { 1.0 } else { 0.0 });
implement_scalar!(DistHamming, u32, |(a, b)| if a != b { 1.0 } else { 0.0 });
implement_scalar!(DistHamming, u16, |(a, b)| if a != b { 1.0 } else { 0.0 });
implement_scalar!(DistHamming, u8, |(a, b)| if a != b { 1.0 } else { 0.0 });

// DistJaccard for other types
implement_scalar!(DistJaccard, u32, |(a, b)| if a == b { 0.0 } else { 1.0 });
implement_scalar!(DistJaccard, u16, |(a, b)| if a == b { 0.0 } else { 1.0 });
implement_scalar!(DistJaccard, u8, |(a, b)| if a == b { 0.0 } else { 1.0 });

/// L2 normalization utility
pub fn l2_normalize(va: &mut [f32]) {
    let l2norm = va.iter().map(|t| *t * *t).sum::<f32>().sqrt();
    if l2norm > 0. {
        for i in 0..va.len() {
            va[i] /= l2norm;
        }
    }
}

/// Levenshtein distance for u16
#[derive(Default, Clone, Copy)]
pub struct DistLevenshtein;
impl Distance<u16> for DistLevenshtein {
    fn eval(&self, a: &[u16], b: &[u16]) -> f32 {
        let len_a = a.len();
        let len_b = b.len();
        if len_a < len_b {
            return self.eval(b, a);
        }
        if len_a == 0 {
            return len_b as f32;
        }
        let mut cur = vec![0; len_b + 1];
        for i in 1..=len_b {
            cur[i] = i;
        }
        for (i, ca) in a.iter().enumerate() {
            let mut pre = cur[0];
            cur[0] = i + 1;
            for (j, cb) in b.iter().enumerate() {
                let tmp = cur[j + 1];
                cur[j + 1] = std::cmp::min(
                    tmp + 1,
                    std::cmp::min(cur[j] + 1, pre + if ca == cb { 0 } else { 1 }),
                );
                pre = tmp;
            }
        }
        cur[len_b] as f32
    }
}

/// Hellinger distance fallback
#[derive(Default, Clone, Copy)]
pub struct DistHellinger;
impl Distance<f32> for DistHellinger {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        let dist = va
            .iter()
            .zip(vb.iter())
            .map(|t| t.0.sqrt() * t.1.sqrt())
            .sum::<f32>();
        (1.0 - dist).max(0.0).sqrt()
    }
}

/// Jensen-Shannon distance fallback
#[derive(Default, Clone, Copy)]
pub struct DistJensenShannon;
impl Distance<f32> for DistJensenShannon {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        let mut dist = 0.0;
        for i in 0..va.len() {
            let mean = 0.5 * (va[i] + vb[i]);
            if va[i] > 0. {
                dist += va[i] * (va[i] / mean).ln();
            }
            if vb[i] > 0. {
                dist += vb[i] * (vb[i] / mean).ln();
            }
        }
        (0.5 * dist).sqrt()
    }
}

/// C function pointer distance
pub struct DistCFFI<T: Send + Sync> {
    pub func: extern "C" fn(*const T, *const T, u64) -> f32,
}
impl<T: Send + Sync> DistCFFI<T> {
    pub fn new(f: extern "C" fn(*const T, *const T, u64) -> f32) -> Self {
        DistCFFI { func: f }
    }
}
impl<T: Send + Sync> Distance<T> for DistCFFI<T> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        (self.func)(va.as_ptr(), vb.as_ptr(), va.len() as u64)
    }
}

/// Function pointer distance
pub struct DistPtr<T: Send + Sync, U: Send + Sync = T> {
    pub func: fn(&[T], &[T]) -> f32,
    _marker: std::marker::PhantomData<U>,
}
impl<T: Send + Sync, U: Send + Sync> DistPtr<T, U> {
    pub fn new(f: fn(&[T], &[T]) -> f32) -> Self {
        DistPtr {
            func: f,
            _marker: std::marker::PhantomData,
        }
    }
}
impl<T: Send + Sync, U: Send + Sync> Distance<T> for DistPtr<T, U> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        (self.func)(va, vb)
    }
}
impl<T: Send + Sync, U: Send + Sync> Default for DistPtr<T, U> {
    fn default() -> Self {
        panic!("DistPtr has no default");
    }
}
