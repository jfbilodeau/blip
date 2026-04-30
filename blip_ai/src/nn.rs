//! Neural-network primitives used by the Blip transformer.
//!
//! Each parameterized module exposes:
//!   * `forward`        - returns the output (no caching, used at inference)
//!   * `forward_train`  - returns `(output, cache)` used during training
//!   * `backward`       - given upstream gradient + cache, returns gradient
//!     w.r.t. the input and accumulates parameter gradients in `*_grad`.
//!   * `apply_grads(opt)` - apply accumulated grads via the configured
//!     optimizer (SGD or Adam) and zero them.
//!
//! All linear algebra goes through `ndarray`. Linear backward uses an outer
//! product expressed as a `[out,1] x [1,in]` matmul, which is dramatically
//! faster than the naive double loop for non-trivial dims.

use ndarray::{Array1, Array2, Axis};
use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// Seeded RNG
// ---------------------------------------------------------------------------

thread_local! {
    static RNG: RefCell<ChaCha12Rng> = RefCell::new(ChaCha12Rng::from_os_rng());
}

/// Set the global seed used by all initialization helpers in this module.
pub fn seed(value: u64) {
    RNG.with(|r| *r.borrow_mut() = ChaCha12Rng::seed_from_u64(value));
}

/// Xavier-uniform initialization for a `[out, in]` matrix.
pub fn xavier(rows: usize, cols: usize) -> Array2<f32> {
    let bound = (6.0_f32 / (rows as f32 + cols as f32)).sqrt();
    let dist = Uniform::new(-bound, bound).expect("uniform");
    RNG.with(|r| {
        let mut rng = r.borrow_mut();
        Array2::from_shape_fn((rows, cols), |_| dist.sample(&mut *rng))
    })
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum Optimizer {
    /// Plain stochastic gradient descent. Exposed mostly for testing.
    Sgd { lr: f32 },
    /// Adam. Steps are tracked per call to `step`.
    Adam {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        /// Global L2 grad-norm clip threshold. `None` to disable.
        grad_clip: Option<f32>,
    },
}

impl Optimizer {
    pub fn adam(lr: f32) -> Self {
        Self::Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_clip: Some(1.0),
        }
    }

    pub fn sgd(lr: f32) -> Self {
        Self::Sgd { lr }
    }

    pub fn lr(&self) -> f32 {
        match self {
            Self::Sgd { lr } | Self::Adam { lr, .. } => *lr,
        }
    }
}

/// Adam moment tracking for a single tensor (1D or 2D — we just store flat).
#[derive(Default, Clone)]
pub struct AdamState1 {
    pub m: Array1<f32>,
    pub v: Array1<f32>,
    pub t: u32,
}

#[derive(Default, Clone)]
pub struct AdamState2 {
    pub m: Array2<f32>,
    pub v: Array2<f32>,
    pub t: u32,
}

fn adam_step1(
    param: &mut Array1<f32>,
    grad: &Array1<f32>,
    state: &mut AdamState1,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
) {
    if state.m.len() != param.len() {
        state.m = Array1::zeros(param.raw_dim());
        state.v = Array1::zeros(param.raw_dim());
        state.t = 0;
    }
    state.t += 1;
    let bc1 = 1.0 - b1.powi(state.t as i32);
    let bc2 = 1.0 - b2.powi(state.t as i32);
    for i in 0..param.len() {
        let g = grad[i];
        state.m[i] = b1 * state.m[i] + (1.0 - b1) * g;
        state.v[i] = b2 * state.v[i] + (1.0 - b2) * g * g;
        let m_hat = state.m[i] / bc1;
        let v_hat = state.v[i] / bc2;
        param[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

fn adam_step2(
    param: &mut Array2<f32>,
    grad: &Array2<f32>,
    state: &mut AdamState2,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
) {
    if state.m.shape() != param.shape() {
        state.m = Array2::zeros(param.raw_dim());
        state.v = Array2::zeros(param.raw_dim());
        state.t = 0;
    }
    state.t += 1;
    let bc1 = 1.0 - b1.powi(state.t as i32);
    let bc2 = 1.0 - b2.powi(state.t as i32);
    for ((p, g), (m, v)) in param
        .iter_mut()
        .zip(grad.iter())
        .zip(state.m.iter_mut().zip(state.v.iter_mut()))
    {
        *m = b1 * *m + (1.0 - b1) * *g;
        *v = b2 * *v + (1.0 - b2) * *g * *g;
        let m_hat = *m / bc1;
        let v_hat = *v / bc2;
        *p -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

// ---------------------------------------------------------------------------
// Linear:  y = W x + b
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct Linear {
    /// shape `[out_dim, in_dim]`
    pub weights: Array2<f32>,
    /// shape `[out_dim]`
    pub bias: Array1<f32>,
    #[serde(skip, default = "Linear::zero_w")]
    pub w_grad: Array2<f32>,
    #[serde(skip, default = "Linear::zero_b")]
    pub b_grad: Array1<f32>,
    #[serde(skip, default)]
    pub w_state: AdamState2,
    #[serde(skip, default)]
    pub b_state: AdamState1,
}

impl Linear {
    fn zero_w() -> Array2<f32> {
        Array2::zeros((0, 0))
    }
    fn zero_b() -> Array1<f32> {
        Array1::zeros(0)
    }

    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            weights: xavier(out_dim, in_dim),
            bias: Array1::zeros(out_dim),
            w_grad: Array2::zeros((out_dim, in_dim)),
            b_grad: Array1::zeros(out_dim),
            w_state: AdamState2::default(),
            b_state: AdamState1::default(),
        }
    }

    pub fn ensure_grads(&mut self) {
        if self.w_grad.shape() != self.weights.shape() {
            self.w_grad = Array2::zeros(self.weights.raw_dim());
        }
        if self.b_grad.shape() != self.bias.shape() {
            self.b_grad = Array1::zeros(self.bias.raw_dim());
        }
    }

    pub fn in_dim(&self) -> usize {
        self.weights.shape()[1]
    }
    pub fn out_dim(&self) -> usize {
        self.weights.shape()[0]
    }

    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        self.weights.dot(x) + &self.bias
    }

    pub fn forward_train(&self, x: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        (self.forward(x), x.clone())
    }

    /// Outer-product accumulation: `W_grad += d_y * x^T`.
    pub fn backward(&mut self, d_y: &Array1<f32>, x_cache: &Array1<f32>) -> Array1<f32> {
        self.ensure_grads();
        // Reshape to [out,1] and [1,in], then matmul -> [out,in]
        let dy = d_y.view().insert_axis(Axis(1));
        let xt = x_cache.view().insert_axis(Axis(0));
        // outer product
        let outer = dy.dot(&xt);
        self.w_grad += &outer;
        self.b_grad += d_y;
        self.weights.t().dot(d_y)
    }

    pub fn apply_grads(&mut self, opt: Optimizer) {
        self.ensure_grads();
        match opt {
            Optimizer::Sgd { lr } => {
                self.weights.scaled_add(-lr, &self.w_grad);
                self.bias.scaled_add(-lr, &self.b_grad);
            }
            Optimizer::Adam {
                lr, beta1, beta2, eps, ..
            } => {
                adam_step2(
                    &mut self.weights,
                    &self.w_grad,
                    &mut self.w_state,
                    lr,
                    beta1,
                    beta2,
                    eps,
                );
                adam_step1(
                    &mut self.bias,
                    &self.b_grad,
                    &mut self.b_state,
                    lr,
                    beta1,
                    beta2,
                    eps,
                );
            }
        }
        self.w_grad.fill(0.0);
        self.b_grad.fill(0.0);
    }

    /// Squared L2 norm of accumulated gradients (used for global clipping).
    pub fn grad_sq_norm(&self) -> f32 {
        self.w_grad.iter().map(|v| v * v).sum::<f32>()
            + self.b_grad.iter().map(|v| v * v).sum::<f32>()
    }

    pub fn scale_grads(&mut self, factor: f32) {
        self.w_grad *= factor;
        self.b_grad *= factor;
    }
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct LayerNorm {
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    pub eps: f32,
    #[serde(skip, default = "LayerNorm::zero")]
    pub g_grad: Array1<f32>,
    #[serde(skip, default = "LayerNorm::zero")]
    pub b_grad: Array1<f32>,
    #[serde(skip, default)]
    pub g_state: AdamState1,
    #[serde(skip, default)]
    pub b_state: AdamState1,
}

pub struct LayerNormCache {
    pub x_hat: Array1<f32>,
    pub inv_std: f32,
}

impl LayerNorm {
    fn zero() -> Array1<f32> {
        Array1::zeros(0)
    }

    pub fn new(dim: usize) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps: 1e-5,
            g_grad: Array1::zeros(dim),
            b_grad: Array1::zeros(dim),
            g_state: AdamState1::default(),
            b_state: AdamState1::default(),
        }
    }

    pub fn ensure_grads(&mut self) {
        if self.g_grad.shape() != self.gamma.shape() {
            self.g_grad = Array1::zeros(self.gamma.raw_dim());
        }
        if self.b_grad.shape() != self.beta.shape() {
            self.b_grad = Array1::zeros(self.beta.raw_dim());
        }
    }

    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        self.forward_train(x).0
    }

    pub fn forward_train(&self, x: &Array1<f32>) -> (Array1<f32>, LayerNormCache) {
        let n = x.len() as f32;
        let mean = x.sum() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + self.eps).sqrt();
        let x_hat = (x - mean) * inv_std;
        let y = &x_hat * &self.gamma + &self.beta;
        (y, LayerNormCache { x_hat, inv_std })
    }

    pub fn backward(&mut self, d_y: &Array1<f32>, cache: &LayerNormCache) -> Array1<f32> {
        self.ensure_grads();
        let n = d_y.len() as f32;

        self.g_grad += &(d_y * &cache.x_hat);
        self.b_grad += d_y;

        let d_x_hat = d_y * &self.gamma;
        let sum_dxhat: f32 = d_x_hat.sum();
        let sum_dxhat_xhat: f32 = (&d_x_hat * &cache.x_hat).sum();
        let scale = cache.inv_std / n;
        let term = &d_x_hat * n - sum_dxhat - &cache.x_hat * sum_dxhat_xhat;
        term * scale
    }

    pub fn apply_grads(&mut self, opt: Optimizer) {
        self.ensure_grads();
        match opt {
            Optimizer::Sgd { lr } => {
                self.gamma.scaled_add(-lr, &self.g_grad);
                self.beta.scaled_add(-lr, &self.b_grad);
            }
            Optimizer::Adam {
                lr, beta1, beta2, eps, ..
            } => {
                adam_step1(
                    &mut self.gamma,
                    &self.g_grad,
                    &mut self.g_state,
                    lr,
                    beta1,
                    beta2,
                    eps,
                );
                adam_step1(
                    &mut self.beta,
                    &self.b_grad,
                    &mut self.b_state,
                    lr,
                    beta1,
                    beta2,
                    eps,
                );
            }
        }
        self.g_grad.fill(0.0);
        self.b_grad.fill(0.0);
    }

    pub fn grad_sq_norm(&self) -> f32 {
        self.g_grad.iter().map(|v| v * v).sum::<f32>()
            + self.b_grad.iter().map(|v| v * v).sum::<f32>()
    }

    pub fn scale_grads(&mut self, factor: f32) {
        self.g_grad *= factor;
        self.b_grad *= factor;
    }
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

pub fn relu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.max(0.0))
}

pub fn relu_backward(d_y: &Array1<f32>, pre_activation: &Array1<f32>) -> Array1<f32> {
    Array1::from_iter(
        d_y.iter()
            .zip(pre_activation.iter())
            .map(|(g, z)| if *z > 0.0 { *g } else { 0.0 }),
    )
}

/// GELU activation, tanh approximation (Hendrycks & Gimpel 2016, the variant
/// used by GPT-2). Close enough to the exact GELU but much cheaper.
///   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
const GELU_C: f32 = 0.7978845608028654; // sqrt(2/pi)
const GELU_A: f32 = 0.044715;

pub fn gelu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| {
        let inner = GELU_C * (v + GELU_A * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    })
}

pub fn gelu_backward(d_y: &Array1<f32>, pre_activation: &Array1<f32>) -> Array1<f32> {
    Array1::from_iter(d_y.iter().zip(pre_activation.iter()).map(|(g, z)| {
        let z = *z;
        let z2 = z * z;
        let inner = GELU_C * (z + GELU_A * z * z2);
        let t = inner.tanh();
        let dinner_dz = GELU_C * (1.0 + 3.0 * GELU_A * z2);
        let dgelu_dz = 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * dinner_dz;
        *g * dgelu_dz
    }))
}

// ---------------------------------------------------------------------------
// Dropout (training-only). At inference, modules skip the mask entirely.
// We implement the standard "inverted dropout": during training, surviving
// activations are scaled by 1/(1-p) so the expected value matches inference.
// ---------------------------------------------------------------------------

pub struct DropoutCache {
    pub mask: Array1<f32>,
}

pub fn dropout_forward(x: &Array1<f32>, p: f32) -> (Array1<f32>, DropoutCache) {
    if p <= 0.0 {
        let mask = Array1::ones(x.len());
        return (x.clone(), DropoutCache { mask });
    }
    let keep = 1.0 - p;
    let scale = 1.0 / keep;
    let dist = Uniform::new(0.0_f32, 1.0).expect("uniform");
    let mask = RNG.with(|r| {
        let mut rng = r.borrow_mut();
        Array1::from_shape_fn(x.len(), |_| {
            if dist.sample(&mut *rng) < keep { scale } else { 0.0 }
        })
    });
    let y = x * &mask;
    (y, DropoutCache { mask })
}

pub fn dropout_backward(d_y: &Array1<f32>, cache: &DropoutCache) -> Array1<f32> {
    d_y * &cache.mask
}

// ---------------------------------------------------------------------------
// Numerically stable softmax + cross-entropy
// ---------------------------------------------------------------------------

pub fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits.mapv(|v| (v - max).exp());
    let sum = exps.sum();
    exps / sum
}

pub fn softmax_cross_entropy(logits: &Array1<f32>, target: usize) -> (f32, Array1<f32>) {
    let probs = softmax(logits);
    let loss = -probs[target].max(1e-12).ln();
    let mut d_logits = probs;
    d_logits[target] -= 1.0;
    (loss, d_logits)
}

// ---------------------------------------------------------------------------
// FeedForward: Linear -> ReLU -> Linear
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct FeedForward {
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct FeedForwardCache {
    pub x: Array1<f32>,
    pub z1: Array1<f32>,
    pub h: Array1<f32>,
}

impl FeedForward {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            fc1: Linear::new(dim, hidden),
            fc2: Linear::new(hidden, dim),
        }
    }

    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let z1 = self.fc1.forward(x);
        let h = gelu(&z1);
        self.fc2.forward(&h)
    }

    pub fn forward_train(&self, x: &Array1<f32>) -> (Array1<f32>, FeedForwardCache) {
        let z1 = self.fc1.forward(x);
        let h = gelu(&z1);
        let y = self.fc2.forward(&h);
        (
            y,
            FeedForwardCache {
                x: x.clone(),
                z1,
                h,
            },
        )
    }

    pub fn backward(&mut self, d_y: &Array1<f32>, cache: &FeedForwardCache) -> Array1<f32> {
        let d_h = self.fc2.backward(d_y, &cache.h);
        let d_z1 = gelu_backward(&d_h, &cache.z1);
        self.fc1.backward(&d_z1, &cache.x)
    }

    pub fn apply_grads(&mut self, opt: Optimizer) {
        self.fc1.apply_grads(opt);
        self.fc2.apply_grads(opt);
    }

    pub fn grad_sq_norm(&self) -> f32 {
        self.fc1.grad_sq_norm() + self.fc2.grad_sq_norm()
    }

    pub fn scale_grads(&mut self, factor: f32) {
        self.fc1.scale_grads(factor);
        self.fc2.scale_grads(factor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn linear_backward_decreases_loss() {
        seed(42);
        let mut layer = Linear::new(3, 2);
        let opt = Optimizer::adam(0.05);
        let x = array![1.0, -0.5, 2.0];
        let target = array![0.5, -1.0];
        let mut last = f32::INFINITY;
        for _ in 0..200 {
            let (y, cache) = layer.forward_train(&x);
            let diff = &y - &target;
            let loss: f32 = diff.iter().map(|v| v * v).sum();
            let d_y = diff * 2.0;
            layer.backward(&d_y, &cache);
            layer.apply_grads(opt);
            last = loss;
        }
        assert!(last < 1e-3, "final loss {}", last);
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&array![1.0_f32, 2.0, 3.0]);
        assert!((p.sum() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cross_entropy_grad_sign() {
        let logits = array![1.0_f32, 2.0, 0.5];
        let (_, d) = softmax_cross_entropy(&logits, 2);
        assert!(d[2] < 0.0);
        assert!(d.sum().abs() < 1e-5);
    }

    #[test]
    fn layer_norm_normalizes() {
        let ln = LayerNorm::new(4);
        let x = array![1.0_f32, 2.0, 3.0, 4.0];
        let y = ln.forward(&x);
        let mean: f32 = y.sum() / 4.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn seeded_init_is_deterministic() {
        seed(123);
        let a = xavier(3, 4);
        seed(123);
        let b = xavier(3, 4);
        assert_eq!(a, b);
    }
}
