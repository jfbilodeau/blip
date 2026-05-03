//! Blip transformer language model.
//!
//! Decoder-only architecture:
//!   token_id -> learned embedding (+ sinusoidal positional encoding)
//!     -> N x DecoderBlock(LN -> MultiHeadAttention(causal) -> +residual ->
//!                         LN -> FeedForward -> +residual)
//!     -> final LayerNorm
//!     -> LM head Linear(embed_dim -> vocab_size)
//!     -> softmax + cross-entropy
//!
//! Training uses next-token cross-entropy with Adam (with global L2 grad
//! clipping). Generation supports greedy / temperature / top-k / top-p
//! sampling.
//!
//! Checkpoints support both bincode and JSON. The project defaults now use
//! `.json` paths for easier inspection.

use crate::nn::{
    softmax, softmax_cross_entropy, AdamState2, FeedForward, FeedForwardCache, LayerNorm,
    LayerNormCache, Linear, Optimizer,
};
use crate::tokenizer::default_token_texts;
use crate::version::MODEL_VERSION;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayViewMut1};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

// Special tokens
pub const TOKEN_UNKNOWN: &str = "<unk>";
pub const TOKEN_STOP: &str = "<stop>";
pub const TOKEN_TOOL: &str = "<tool>";
pub const TOKEN_BEGIN: &str = "<bos>";
pub const TOKEN_USER: &str = "<user>";
pub const TOKEN_AI: &str = "<ai>";

#[derive(Serialize, Deserialize, Clone)]
pub struct Token {
    pub text: String,
    pub usage_count: u32,
}

// ---------------------------------------------------------------------------
// Multi-head causal self-attention.
//
// Implemented by reshaping the per-position Q/K/V vectors of length `dim` into
// `n_heads` chunks of size `head_dim = dim / n_heads`. Heads do not share
// parameters because Q/K/V are still single Linear projections of the full
// vector — the head split happens in the activations only, exactly as in the
// original transformer.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct SelfAttention {
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    pub n_heads: usize,
}

pub struct AttentionCache {
    pub xs: Array2<f32>,
    pub qs: Array2<f32>,
    pub ks: Array2<f32>,
    pub vs: Array2<f32>,
    /// Per (head, query position) attention distribution over keys 0..=i.
    /// `attn[(h, i, j)]` = weight from query i to key j in head h.
    /// Shape: (n_heads, seq_len, seq_len)
    pub attn: Array3<f32>,
    /// Per-position concatenated context (input to W_o).
    pub context: Array2<f32>,
}

/// Per-layer KV cache for incremental generation. Stores the full key and
/// value vectors (concatenated across heads) for every position seen so far.
#[derive(Default, Clone)]
pub struct AttnKVCache {
    pub ks: Vec<Array1<f32>>,
    pub vs: Vec<Array1<f32>>,
}

impl AttnKVCache {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.ks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ks.is_empty()
    }
}

impl SelfAttention {
    pub fn new(dim: usize, n_heads: usize) -> Self {
        assert!(
            dim % n_heads == 0,
            "embedding_dim {} must be divisible by n_heads {}",
            dim,
            n_heads
        );
        Self {
            w_q: Linear::new(dim, dim),
            w_k: Linear::new(dim, dim),
            w_v: Linear::new(dim, dim),
            w_o: Linear::new(dim, dim),
            n_heads,
        }
    }

    fn dim(&self) -> usize {
        self.w_q.in_dim()
    }
    fn head_dim(&self) -> usize {
        self.dim() / self.n_heads
    }

    pub fn forward(&self, xs: &Array2<f32>) -> Array2<f32> {
        self.forward_train(xs).0
    }

    pub fn forward_train(&self, xs: &Array2<f32>) -> (Array2<f32>, AttentionCache) {
        let dim = self.dim();
        let head_dim = self.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = xs.nrows();

        let mut qs = Array2::<f32>::zeros((n, dim));
        let mut ks = Array2::<f32>::zeros((n, dim));
        let mut vs = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let x = xs.row(i).to_owned();
            qs.row_mut(i).assign(&self.w_q.forward(&x));
            ks.row_mut(i).assign(&self.w_k.forward(&x));
            vs.row_mut(i).assign(&self.w_v.forward(&x));
        }

        let mut attn = Array3::<f32>::zeros((self.n_heads, n, n));
        let mut context = Array2::<f32>::zeros((n, dim));
        let mut scores = vec![0.0_f32; n];

        for h in 0..self.n_heads {
            let off = h * head_dim;
            for i in 0..n {
                // scores[j] = (q_i_h . k_j_h) * scale, for j in 0..=i (causal)
                // Use ndarray slicing for vectorization.
                for j in 0..=i {
                    let q_slice = qs.slice(s![i, off..off+head_dim]);
                    let k_slice = ks.slice(s![j, off..off+head_dim]);
                    scores[j] = q_slice.dot(&k_slice) * scale;
                }
                let mut score_view = ArrayViewMut1::from(&mut scores[..=i]);
                let max = score_view.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                score_view.mapv_inplace(|v| (v - max).exp());
                let sum = score_view.sum();
                score_view.mapv_inplace(|v| v / sum);

                // ctx[h][i] = sum_j scores[j] * v_j_h
                for j in 0..=i {
                    let w = scores[j];
                    let v_slice = vs.slice(s![j, off..off+head_dim]);
                    let mut ctx_slice = context.slice_mut(s![i, off..off+head_dim]);
                    ctx_slice.scaled_add(w, &v_slice);
                }
                // Store attention weights in Array3.
                for j in 0..=i {
                    attn[[h, i, j]] = scores[j];
                }
            }
        }

        let mut outputs = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let ctx = context.row(i).to_owned();
            outputs.row_mut(i).assign(&self.w_o.forward(&ctx));
        }

        (
            outputs,
            AttentionCache {
                xs: xs.clone(),
                qs,
                ks,
                vs,
                attn,
                context,
            },
        )
    }

    pub fn backward(
        &mut self,
        d_outs: &Array2<f32>,
        cache: &AttentionCache,
    ) -> Array2<f32> {
        let dim = self.dim();
        let head_dim = self.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = d_outs.nrows();

        // d_context per position via W_o backward
        let mut d_context = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let d_out = d_outs.row(i).to_owned();
            let ctx = cache.context.row(i).to_owned();
            d_context.row_mut(i).assign(&self.w_o.backward(&d_out, &ctx));
        }

        let mut d_q = Array2::<f32>::zeros((n, dim));
        let mut d_k = Array2::<f32>::zeros((n, dim));
        let mut d_v = Array2::<f32>::zeros((n, dim));
        let mut d_attn = vec![0.0_f32; n];
        let mut d_score = vec![0.0_f32; n];

        for h in 0..self.n_heads {
            let off = h * head_dim;
            for i in 0..n {
                let dc_slice = d_context.slice(s![i, off..off+head_dim]);
                let q_slice = cache.qs.slice(s![i, off..off+head_dim]);
                // d_attn[j] = sum_d d_context[i][off+d] * v_j[off+d]
                // d_v[j][off+d] += attn[j] * d_context[i][off+d]
                for j in 0..=i {
                    let v_slice = cache.vs.slice(s![j, off..off+head_dim]);
                    d_attn[j] = dc_slice.dot(&v_slice);
                    // Accumulate d_v.
                    let attn_weight = cache.attn[[h, i, j]];
                    let mut d_v_slice = d_v.slice_mut(s![j, off..off+head_dim]);
                    d_v_slice.scaled_add(attn_weight, &dc_slice);
                }
                // softmax backward
                let attn_row = cache.attn.slice(s![h, i, 0..=i]);
                let d_attn_view = ArrayView1::from(&d_attn[..=i]);
                let dot = attn_row.dot(&d_attn_view);
                let mut d_score_view = ArrayViewMut1::from(&mut d_score[..=i]);
                d_score_view.assign(&(&attn_row * &(d_attn_view.to_owned() - dot)));
                // d_q[i][off+d] += scale * sum_k d_score[k] * k_k[off+d]
                // d_k[k][off+d] += scale * d_score[k] * q_i[off+d]
                for k in 0..=i {
                    let s_k = scale * d_score[k];
                    let k_slice = cache.ks.slice(s![k, off..off+head_dim]);
                    let mut d_q_slice = d_q.slice_mut(s![i, off..off+head_dim]);
                    d_q_slice.scaled_add(s_k, &k_slice);

                    let mut d_k_slice = d_k.slice_mut(s![k, off..off+head_dim]);
                    d_k_slice.scaled_add(s_k, &q_slice);
                }
            }
        }

        let mut d_xs = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let x = cache.xs.row(i).to_owned();
            let dq = self.w_q.backward(&d_q.row(i).to_owned(), &x);
            let dk = self.w_k.backward(&d_k.row(i).to_owned(), &x);
            let dv = self.w_v.backward(&d_v.row(i).to_owned(), &x);
            d_xs.row_mut(i).assign(&(dq + dk + dv));
        }
        d_xs
    }

    /// Single-position incremental forward used during generation. Appends the
    /// new key/value to `cache` and returns the attention output for the new
    /// position. Causal: the new query attends to all cached keys.
    pub fn step(&self, x: &Array1<f32>, cache: &mut AttnKVCache) -> Array1<f32> {
        let dim = self.dim();
        let head_dim = self.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = self.w_q.forward(x);
        let k = self.w_k.forward(x);
        let v = self.w_v.forward(x);
        cache.ks.push(k);
        cache.vs.push(v);
        let n = cache.ks.len();

        let mut context = Array1::<f32>::zeros(dim);
        let mut scores = vec![0.0_f32; n];
        for h in 0..self.n_heads {
            let off = h * head_dim;
            for j in 0..n {
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[off + d] * cache.ks[j][off + d];
                }
                scores[j] = dot * scale;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for s in &mut scores {
                *s = (*s - max).exp();
            }
            let sum: f32 = scores.iter().sum();
            for s in &mut scores {
                *s /= sum;
            }
            for j in 0..n {
                let w = scores[j];
                for d in 0..head_dim {
                    context[off + d] += w * cache.vs[j][off + d];
                }
            }
        }
        self.w_o.forward(&context)
    }

    pub fn apply_grads(&mut self, opt: Optimizer) {
        self.w_q.apply_grads(opt);
        self.w_k.apply_grads(opt);
        self.w_v.apply_grads(opt);
        self.w_o.apply_grads(opt);
    }

    pub fn grad_sq_norm(&self) -> f32 {
        self.w_q.grad_sq_norm()
            + self.w_k.grad_sq_norm()
            + self.w_v.grad_sq_norm()
            + self.w_o.grad_sq_norm()
    }

    pub fn scale_grads(&mut self, factor: f32) {
        self.w_q.scale_grads(factor);
        self.w_k.scale_grads(factor);
        self.w_v.scale_grads(factor);
        self.w_o.scale_grads(factor);
    }
}

// ---------------------------------------------------------------------------
// DecoderBlock: pre-norm transformer block
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct DecoderBlock {
    pub ln1: LayerNorm,
    pub attn: SelfAttention,
    pub ln2: LayerNorm,
    pub ffn: FeedForward,
    /// Runtime-only: dropout probability applied to attention and FFN outputs
    /// during `forward_train`. Not serialized — always defaults to 0.0 on load
    /// so existing checkpoints behave identically.
    #[serde(skip, default)]
    pub dropout: f32,
}

pub struct DecoderBlockCache {
    pub ln1_caches: Vec<LayerNormCache>,
    pub ln1_outs: Array2<f32>,
    pub attn_cache: AttentionCache,
    pub attn_drop: Vec<crate::nn::DropoutCache>,
    pub residual1: Array2<f32>,
    pub ln2_caches: Vec<LayerNormCache>,
    pub ln2_outs: Array2<f32>,
    pub ffn_caches: Vec<FeedForwardCache>,
    pub ffn_drop: Vec<crate::nn::DropoutCache>,
}

impl DecoderBlock {
    pub fn new(dim: usize, n_heads: usize, ffn_hidden: usize) -> Self {
        Self {
            ln1: LayerNorm::new(dim),
            attn: SelfAttention::new(dim, n_heads),
            ln2: LayerNorm::new(dim),
            ffn: FeedForward::new(dim, ffn_hidden),
            dropout: 0.0,
        }
    }

    pub fn set_dropout(&mut self, p: f32) {
        self.dropout = p.clamp(0.0, 1.0);
    }

    pub fn forward(&self, xs: &Array2<f32>) -> Array2<f32> {
        let n = xs.nrows();
        let dim = xs.ncols();
        let mut normed1 = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            normed1
                .row_mut(i)
                .assign(&self.ln1.forward(&xs.row(i).to_owned()));
        }
        let attn_out = self.attn.forward(&normed1);
        let mut res1 = xs.clone();
        res1 += &attn_out;
        let mut normed2 = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            normed2
                .row_mut(i)
                .assign(&self.ln2.forward(&res1.row(i).to_owned()));
        }
        let mut ffn_out = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            ffn_out
                .row_mut(i)
                .assign(&self.ffn.forward(&normed2.row(i).to_owned()));
        }
        res1 + ffn_out
    }

    pub fn forward_train(&self, xs: &Array2<f32>) -> (Array2<f32>, DecoderBlockCache) {
        let n = xs.nrows();
        let dim = xs.ncols();
        let mut ln1_caches = Vec::with_capacity(n);
        let mut ln1_outs = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let (y, c) = self.ln1.forward_train(&xs.row(i).to_owned());
            ln1_outs.row_mut(i).assign(&y);
            ln1_caches.push(c);
        }
        let (attn_outs_raw, attn_cache) = self.attn.forward_train(&ln1_outs);
        let mut attn_drop = Vec::with_capacity(n);
        let mut attn_outs = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let (y, c) = crate::nn::dropout_forward(&attn_outs_raw.row(i).to_owned(), self.dropout);
            attn_outs.row_mut(i).assign(&y);
            attn_drop.push(c);
        }
        let mut residual1 = xs.clone();
        residual1 += &attn_outs;

        let mut ln2_caches = Vec::with_capacity(n);
        let mut ln2_outs = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let (y, c) = self.ln2.forward_train(&residual1.row(i).to_owned());
            ln2_outs.row_mut(i).assign(&y);
            ln2_caches.push(c);
        }

        let mut ffn_caches = Vec::with_capacity(n);
        let mut ffn_drop = Vec::with_capacity(n);
        let mut ffn_outs = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let (y_raw, c_ffn) = self.ffn.forward_train(&ln2_outs.row(i).to_owned());
            let (y, c_drop) = crate::nn::dropout_forward(&y_raw, self.dropout);
            ffn_outs.row_mut(i).assign(&y);
            ffn_caches.push(c_ffn);
            ffn_drop.push(c_drop);
        }
        let outs = &residual1 + &ffn_outs;

        (
            outs,
            DecoderBlockCache {
                ln1_caches,
                ln1_outs,
                attn_cache,
                attn_drop,
                residual1,
                ln2_caches,
                ln2_outs,
                ffn_caches,
                ffn_drop,
            },
        )
    }

    pub fn backward(
        &mut self,
        d_outs: &Array2<f32>,
        cache: &DecoderBlockCache,
    ) -> Array2<f32> {
        let n = d_outs.nrows();
        let dim = d_outs.ncols();
        // d_out flows into both branches of the FFN residual.
        let mut d_residual1 = d_outs.clone();

        let mut d_ln2_out = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            // backprop through FFN dropout, then FFN itself.
            let d_ffn_raw =
                crate::nn::dropout_backward(&d_outs.row(i).to_owned(), &cache.ffn_drop[i]);
            d_ln2_out
                .row_mut(i)
                .assign(&self.ffn.backward(&d_ffn_raw, &cache.ffn_caches[i]));
        }
        for i in 0..n {
            let d_r = self
                .ln2
                .backward(&d_ln2_out.row(i).to_owned(), &cache.ln2_caches[i]);
            let mut row = d_residual1.row_mut(i);
            row += &d_r;
        }

        // d_residual1 flows into both branches of the attention residual.
        let mut d_x = d_residual1.clone();
        let mut d_attn_out_raw = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            d_attn_out_raw.row_mut(i).assign(&crate::nn::dropout_backward(
                &d_residual1.row(i).to_owned(),
                &cache.attn_drop[i],
            ));
        }
        let d_ln1_out = self.attn.backward(&d_attn_out_raw, &cache.attn_cache);
        for i in 0..n {
            let d = self
                .ln1
                .backward(&d_ln1_out.row(i).to_owned(), &cache.ln1_caches[i]);
            let mut row = d_x.row_mut(i);
            row += &d;
        }

        let _ = &cache.ln1_outs;
        d_x
    }

    /// Single-position incremental forward used during generation. `cache`
    /// holds the per-layer attention KV state.
    pub fn step(&self, x: &Array1<f32>, cache: &mut AttnKVCache) -> Array1<f32> {
        let n1 = self.ln1.forward(x);
        let attn_out = self.attn.step(&n1, cache);
        let res1 = x + &attn_out;
        let n2 = self.ln2.forward(&res1);
        let ffn_out = self.ffn.forward(&n2);
        res1 + ffn_out
    }

    pub fn apply_grads(&mut self, opt: Optimizer) {
        self.ln1.apply_grads(opt);
        self.attn.apply_grads(opt);
        self.ln2.apply_grads(opt);
        self.ffn.apply_grads(opt);
    }

    pub fn grad_sq_norm(&self) -> f32 {
        self.ln1.grad_sq_norm()
            + self.attn.grad_sq_norm()
            + self.ln2.grad_sq_norm()
            + self.ffn.grad_sq_norm()
    }

    pub fn scale_grads(&mut self, factor: f32) {
        self.ln1.scale_grads(factor);
        self.attn.scale_grads(factor);
        self.ln2.scale_grads(factor);
        self.ffn.scale_grads(factor);
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal positional encoding.
// ---------------------------------------------------------------------------

pub fn positional_encoding(seq_len: usize, embedding_dim: usize) -> Vec<Array1<f32>> {
    let mut out = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let mut v = Array1::<f32>::zeros(embedding_dim);
        let mut i = 0;
        while i < embedding_dim {
            let angle = pos as f32 / 10000_f32.powf(i as f32 / embedding_dim as f32);
            v[i] = angle.sin();
            if i + 1 < embedding_dim {
                v[i + 1] = angle.cos();
            }
            i += 2;
        }
        out.push(v);
    }
    out
}

// ---------------------------------------------------------------------------
// Sampling configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub max_new_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            max_new_tokens: 64,
        }
    }
}

impl SamplingConfig {
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            max_new_tokens,
        }
    }
}

fn sample_from_logits<R: Rng>(logits: &Array1<f32>, cfg: &SamplingConfig, rng: &mut R) -> usize {
    if cfg.temperature <= 0.0 {
        // Greedy
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, v) in logits.iter().enumerate() {
            if *v > bv {
                bv = *v;
                best = i;
            }
        }
        return best;
    }

    // Apply temperature
    let scaled = logits.mapv(|v| v / cfg.temperature);
    let mut probs = softmax(&scaled).to_vec();

    // top-k mask
    if let Some(k) = cfg.top_k {
        if k > 0 && k < probs.len() {
            let mut idxs: Vec<usize> = (0..probs.len()).collect();
            idxs.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let keep: std::collections::HashSet<usize> = idxs.into_iter().take(k).collect();
            for (i, p) in probs.iter_mut().enumerate() {
                if !keep.contains(&i) {
                    *p = 0.0;
                }
            }
        }
    }

    // top-p (nucleus) mask: keep smallest set whose cumulative prob >= p
    if let Some(p_cut) = cfg.top_p {
        let mut idxs: Vec<usize> = (0..probs.len()).collect();
        idxs.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut cum = 0.0_f32;
        let mut keep = std::collections::HashSet::new();
        for &i in &idxs {
            keep.insert(i);
            cum += probs[i];
            if cum >= p_cut {
                break;
            }
        }
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep.contains(&i) {
                *p = 0.0;
            }
        }
    }

    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        // Fallback to argmax of original logits.
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, v) in logits.iter().enumerate() {
            if *v > bv {
                bv = *v;
                best = i;
            }
        }
        return best;
    }
    for p in &mut probs {
        *p /= sum;
    }

    let r: f32 = rng.random();
    let mut cum = 0.0_f32;
    for (i, p) in probs.iter().enumerate() {
        cum += *p;
        if r <= cum {
            return i;
        }
    }
    probs.len() - 1
}

fn argmax_excluding(logits: &Array1<f32>, excluded: usize) -> Option<usize> {
    let mut best_idx: Option<usize> = None;
    let mut best_val = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        if i == excluded {
            continue;
        }
        if *v > best_val {
            best_val = *v;
            best_idx = Some(i);
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct Model {
    version: String,
    embedding_dim: usize,
    depth: usize,
    n_heads: usize,
    pub tokens: Vec<Token>,
    pub embeddings: Array2<f32>,
    #[serde(skip, default)]
    pub embeddings_grad: Array2<f32>,
    #[serde(skip, default)]
    pub embeddings_state: AdamState2,
    pub blocks: Vec<DecoderBlock>,
    pub final_norm: LayerNorm,
    pub lm_head: Linear,
    /// When true, `lm_head.weights` is kept equal to `embeddings` and only
    /// the LM-head bias is an independent parameter. Defaults to true; old
    /// checkpoints without the field are also treated as tied so their
    /// in-memory `lm_head.weights` becomes a mirror of `embeddings` on load.
    #[serde(default = "Model::default_tie_embeddings")]
    pub tie_embeddings: bool,
    /// Lazily grown cache of sinusoidal positional encodings. Never serialized.
    #[serde(skip, default)]
    pe_cache: std::sync::RwLock<Vec<Array1<f32>>>,
}

impl Clone for Model {
    fn clone(&self) -> Self {
        Self {
            version: self.version.clone(),
            embedding_dim: self.embedding_dim,
            depth: self.depth,
            n_heads: self.n_heads,
            tokens: self.tokens.clone(),
            embeddings: self.embeddings.clone(),
            embeddings_grad: self.embeddings_grad.clone(),
            embeddings_state: self.embeddings_state.clone(),
            blocks: self.blocks.clone(),
            final_norm: self.final_norm.clone(),
            lm_head: self.lm_head.clone(),
            tie_embeddings: self.tie_embeddings,
            pe_cache: std::sync::RwLock::new(
                self.pe_cache
                    .read()
                    .expect("pe cache read lock")
                    .clone(),
            ),
        }
    }
}

impl Model {
    fn is_supported_version(v: &str) -> bool {
        // Older checkpoints lack <user>/<ai> tokens; loading them works but
        // the REPL/tuning paths assume the new specials exist. We still allow
        // the prior version to load so people can inspect old models.
        matches!(v, "0.4.0" | "0.5.0" | "0.6.0" | MODEL_VERSION)
    }

    fn default_tie_embeddings() -> bool {
        true
    }

    pub fn new(embedding_dim: usize, depth: usize, n_heads: usize) -> Self {
        let blocks = (0..depth)
            .map(|_| DecoderBlock::new(embedding_dim, n_heads, 4 * embedding_dim))
            .collect();

        let mut model = Model {
            version: MODEL_VERSION.to_string(),
            embedding_dim,
            depth,
            n_heads,
            tokens: Vec::new(),
            embeddings: Array2::zeros((0, embedding_dim)),
            embeddings_grad: Array2::zeros((0, embedding_dim)),
            embeddings_state: AdamState2::default(),
            blocks,
            final_norm: LayerNorm::new(embedding_dim),
            lm_head: Linear::new(embedding_dim, 1),
            tie_embeddings: true,
            pe_cache: std::sync::RwLock::new(Vec::new()),
        };

        model.register_token(TOKEN_UNKNOWN);
        model.register_token(TOKEN_STOP);
        model.register_token(TOKEN_TOOL);
        model.register_token(TOKEN_BEGIN);
        model.register_token(TOKEN_USER);
        model.register_token(TOKEN_AI);
        for token in default_token_texts() {
            model.register_token(&token);
        }

        model
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Set dropout probability on every decoder block. Affects training only;
    /// inference (`forward`, `forward_logits`, `generate*`) is unchanged.
    pub fn set_dropout(&mut self, p: f32) {
        for b in &mut self.blocks {
            b.set_dropout(p);
        }
    }

    /// Loads from disk. JSON if path ends in `.json`, otherwise bincode.
    pub fn load(file_name: &str) -> Result<Self, String> {
        let bytes = std::fs::read(file_name).map_err(|e| e.to_string())?;
        let mut model: Model = if Path::new(file_name)
            .extension()
            .and_then(|s| s.to_str())
            == Some("json")
        {
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?
        } else {
            let cfg = bincode::config::standard();
            let (m, _): (Model, usize) =
                bincode::serde::decode_from_slice(&bytes, cfg).map_err(|e| e.to_string())?;
            m
        };
        if !Self::is_supported_version(&model.version) {
            return Err(format!(
                "Model version mismatch: expected {} (or compatible prior version), got {}",
                MODEL_VERSION, model.version
            ));
        }
        // Normalize compatible checkpoints to the current version when loaded.
        model.version = MODEL_VERSION.to_string();
        model.embeddings_grad = Array2::zeros(model.embeddings.raw_dim());
        // Re-tie the LM head from the embedding table. For old checkpoints
        // that trained with an independent head this discards the previously
        // learned head weights, but going forward the model is tied. The
        // bias term is preserved.
        if model.tie_embeddings {
            model.sync_lm_head_from_embeddings();
        }
        Ok(model)
    }

    pub fn save(&self, file_name: &str) -> Result<(), String> {
        if let Some(parent) = Path::new(file_name).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        if Path::new(file_name)
            .extension()
            .and_then(|s| s.to_str())
            == Some("json")
        {
            let file = std::fs::File::create(file_name).map_err(|e| e.to_string())?;
            serde_json::to_writer(file, self).map_err(|e| e.to_string())?;
        } else {
            let cfg = bincode::config::standard();
            let bytes = bincode::serde::encode_to_vec(self, cfg).map_err(|e| e.to_string())?;
            std::fs::write(file_name, bytes).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    // -- vocabulary -----------------------------------------------------------

    pub fn get_token_id(&self, token: &str) -> Option<usize> {
        self.tokens.iter().position(|e| e.text == token)
    }

    pub fn get_token_by_id(&self, id: usize) -> Option<&str> {
        self.tokens.get(id).map(|t| t.text.as_str())
    }

    pub fn get_unknown_token_id(&self) -> usize {
        self.get_token_id(TOKEN_UNKNOWN).unwrap()
    }
    pub fn get_stop_token_id(&self) -> usize {
        self.get_token_id(TOKEN_STOP).unwrap()
    }
    pub fn get_tool_token_id(&self) -> usize {
        self.get_token_id(TOKEN_TOOL).unwrap()
    }
    pub fn get_begin_token_id(&self) -> usize {
        self.get_token_id(TOKEN_BEGIN).unwrap()
    }
    pub fn get_user_token_id(&self) -> usize {
        self.get_token_id(TOKEN_USER).unwrap()
    }
    pub fn get_ai_token_id(&self) -> usize {
        self.get_token_id(TOKEN_AI).unwrap()
    }

    pub fn register_token(&mut self, id: &str) -> usize {
        if let Some(index) = self.get_token_id(id) {
            self.tokens[index].usage_count += 1;
            return index;
        }
        let new_index = self.tokens.len();
        self.tokens.push(Token {
            text: id.to_string(),
            usage_count: 1,
        });
        new_index
    }

    /// Drop tokens with `usage_count < min_count` from the vocabulary. Special
    /// tokens (`<unk>`, `<stop>`, `<tool>`, `<bos>`) are always retained.
    /// Must be called before `initialize_embeddings`. Returns the number of
    /// tokens removed.
    pub fn trim_vocab(&mut self, min_count: u32) -> usize {
        if min_count <= 1 {
            return 0;
        }
        let before = self.tokens.len();
        let specials = [TOKEN_UNKNOWN, TOKEN_STOP, TOKEN_TOOL, TOKEN_BEGIN, TOKEN_USER, TOKEN_AI];
        self.tokens.retain(|t| {
            specials.contains(&t.text.as_str()) || t.usage_count >= min_count
        });
        before - self.tokens.len()
    }

    /// Allocate the embedding matrix and LM head once the vocabulary is final.
    pub fn initialize_embeddings(&mut self) {
        let vocab = self.tokens.len();
        self.embeddings = crate::nn::xavier(vocab, self.embedding_dim);
        self.embeddings_grad = Array2::zeros((vocab, self.embedding_dim));
        self.embeddings_state = AdamState2::default();
        self.lm_head = Linear::new(self.embedding_dim, vocab);
        if self.tie_embeddings {
            self.sync_lm_head_from_embeddings();
        }
    }

    /// Copy `embeddings` into `lm_head.weights` so the head shares parameters
    /// with the input embedding table. The head bias is left untouched.
    pub fn sync_lm_head_from_embeddings(&mut self) {
        if self.lm_head.weights.shape() == self.embeddings.shape() {
            self.lm_head.weights.assign(&self.embeddings);
        }
    }

    fn ensure_embeddings_grad(&mut self) {
        if self.embeddings_grad.shape() != self.embeddings.shape() {
            self.embeddings_grad = Array2::zeros(self.embeddings.raw_dim());
        }
    }

    /// Extend `pe_cache` to cover at least `seq_len` positions. Positions
    /// already cached are never recomputed.
    fn ensure_pe(&self, seq_len: usize) {
        let current = self.pe_cache.read().expect("pe cache read lock").len();
        if seq_len > current {
            let mut cache = self.pe_cache.write().expect("pe cache write lock");
            // Compute only the new tail positions.
            for pos in current..seq_len {
                let mut v = Array1::<f32>::zeros(self.embedding_dim);
                let mut i = 0;
                while i < self.embedding_dim {
                    let angle =
                        pos as f32 / 10000_f32.powf(i as f32 / self.embedding_dim as f32);
                    v[i] = angle.sin();
                    if i + 1 < self.embedding_dim {
                        v[i + 1] = angle.cos();
                    }
                    i += 2;
                }
                cache.push(v);
            }
        }
    }

    fn embed_sequence(&self, ids: &[usize]) -> Array2<f32> {
        self.ensure_pe(ids.len());
        let pe = self.pe_cache.read().expect("pe cache read lock");
        let mut out = Array2::<f32>::zeros((ids.len(), self.embedding_dim));
        for (i, &id) in ids.iter().enumerate() {
            let mut row = self.embeddings.row(id).to_owned();
            row += &pe[i];
            out.row_mut(i).assign(&row);
        }
        out
    }

    fn embed_token_at(&self, id: usize, pos: usize) -> Array1<f32> {
        self.ensure_pe(pos + 1);
        let pe = self.pe_cache.read().expect("pe cache read lock");
        let mut v = self.embeddings.row(id).to_owned();
        v += &pe[pos];
        v
    }

    // -- inference forward ----------------------------------------------------

    /// KV cache for incremental generation. One attention cache per decoder
    /// block. Position is implied by cache lengths.
    pub fn new_kv_cache(&self) -> Vec<AttnKVCache> {
        (0..self.blocks.len()).map(|_| AttnKVCache::new()).collect()
    }

    /// Single-token forward step at absolute sequence position `pos`.
    /// The passed KV cache is mutated in place (appended keys/values).
    pub fn step_with_cache(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut [AttnKVCache],
    ) -> Array1<f32> {
        assert_eq!(cache.len(), self.blocks.len(), "cache depth mismatch");
        let mut x = self.embed_token_at(token_id, pos);
        for (block, layer_cache) in self.blocks.iter().zip(cache.iter_mut()) {
            x = block.step(&x, layer_cache);
        }
        let normed = self.final_norm.forward(&x);
        self.lm_head.forward(&normed)
    }

    /// Returns the logits at the last position only.
    pub fn forward_logits(&self, ids: &[usize]) -> Array1<f32> {
        let mut xs = self.embed_sequence(ids);
        for block in &self.blocks {
            xs = block.forward(&xs);
        }
        let last = xs.row(xs.nrows() - 1).to_owned();
        let normed = self.final_norm.forward(&last);
        self.lm_head.forward(&normed)
    }

    // -- training step --------------------------------------------------------

    /// One forward+backward over a single sequence using next-token prediction
    /// at every position. Accumulates gradients (does NOT step).
    pub fn train_sequence(&mut self, ids: &[usize]) -> f32 {
        assert!(ids.len() >= 2, "need at least 2 tokens to predict");

        let inputs = &ids[..ids.len() - 1];
        let targets = &ids[1..];

        let mut xs = self.embed_sequence(inputs);
        let mut block_caches: Vec<DecoderBlockCache> = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (out, cache) = block.forward_train(&xs);
            xs = out;
            block_caches.push(cache);
        }

        let n = xs.nrows();
        let mut ln_caches: Vec<LayerNormCache> = Vec::with_capacity(n);
        let mut normed = Array2::<f32>::zeros(xs.raw_dim());
        for i in 0..n {
            let (y, c) = self.final_norm.forward_train(&xs.row(i).to_owned());
            normed.row_mut(i).assign(&y);
            ln_caches.push(c);
        }

        let mut logits: Vec<Array1<f32>> = Vec::with_capacity(n);
        let mut head_caches: Vec<Array1<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let (y, c) = self.lm_head.forward_train(&normed.row(i).to_owned());
            logits.push(y);
            head_caches.push(c);
        }

        let mut total_loss = 0.0_f32;
        let mut d_logits: Vec<Array1<f32>> = Vec::with_capacity(n);
        for (i, target) in targets.iter().enumerate() {
            let (loss, d) = softmax_cross_entropy(&logits[i], *target);
            total_loss += loss;
            d_logits.push(d);
        }
        let avg_loss = total_loss / targets.len() as f32;

        let scale = 1.0 / targets.len() as f32;
        for d in &mut d_logits {
            *d *= scale;
        }

        let mut d_xs = Array2::<f32>::zeros(xs.raw_dim());
        for i in 0..d_logits.len() {
            let d_normed = self.lm_head.backward(&d_logits[i], &head_caches[i]);
            let d_x = self.final_norm.backward(&d_normed, &ln_caches[i]);
            d_xs.row_mut(i).assign(&d_x);
        }
        for (block, cache) in self.blocks.iter_mut().zip(block_caches.iter()).rev() {
            d_xs = block.backward(&d_xs, cache);
        }

        self.ensure_embeddings_grad();
        for (i, &id) in inputs.iter().enumerate() {
            let mut row = self.embeddings_grad.row_mut(id);
            row += &d_xs.row(i);
        }

        avg_loss
    }

    /// Evaluate average cross-entropy on a sequence (no gradients).
    pub fn eval_sequence(&self, ids: &[usize]) -> f32 {
        if ids.len() < 2 {
            return 0.0;
        }
        let inputs = &ids[..ids.len() - 1];
        let targets = &ids[1..];
        let mut xs = self.embed_sequence(inputs);
        for block in &self.blocks {
            xs = block.forward(&xs);
        }
        let mut total = 0.0_f32;
        for (i, &t) in targets.iter().enumerate() {
            let normed = self.final_norm.forward(&xs.row(i).to_owned());
            let logits = self.lm_head.forward(&normed);
            let p = softmax(&logits);
            total += -p[t].max(1e-12).ln();
        }
        total / targets.len() as f32
    }

    /// Total squared norm of all parameter gradients.
    pub fn grad_sq_norm(&self) -> f32 {
        let mut s = self
            .embeddings_grad
            .iter()
            .map(|v| v * v)
            .sum::<f32>();
        for b in &self.blocks {
            s += b.grad_sq_norm();
        }
        s += self.final_norm.grad_sq_norm();
        s += self.lm_head.grad_sq_norm();
        s
    }

    pub fn scale_all_grads(&mut self, factor: f32) {
        self.embeddings_grad *= factor;
        for b in &mut self.blocks {
            b.scale_grads(factor);
        }
        self.final_norm.scale_grads(factor);
        self.lm_head.scale_grads(factor);
    }

    /// Zero every accumulated gradient tensor in the model.
    pub fn zero_all_grads(&mut self) {
        self.ensure_embeddings_grad();
        self.embeddings_grad.fill(0.0);

        let zero_linear = |l: &mut Linear| {
            l.ensure_grads();
            l.w_grad.fill(0.0);
            l.b_grad.fill(0.0);
        };
        let zero_ln = |ln: &mut LayerNorm| {
            ln.ensure_grads();
            ln.g_grad.fill(0.0);
            ln.b_grad.fill(0.0);
        };

        for b in &mut self.blocks {
            zero_ln(&mut b.ln1);
            zero_linear(&mut b.attn.w_q);
            zero_linear(&mut b.attn.w_k);
            zero_linear(&mut b.attn.w_v);
            zero_linear(&mut b.attn.w_o);
            zero_ln(&mut b.ln2);
            zero_linear(&mut b.ffn.fc1);
            zero_linear(&mut b.ffn.fc2);
        }

        zero_ln(&mut self.final_norm);
        zero_linear(&mut self.lm_head);
    }

    /// Add gradients from another model with identical architecture.
    pub fn add_grads_from(&mut self, other: &Model) {
        self.ensure_embeddings_grad();
        self.embeddings_grad += &other.embeddings_grad;

        let add_linear = |dst: &mut Linear, src: &Linear| {
            dst.ensure_grads();
            dst.w_grad += &src.w_grad;
            dst.b_grad += &src.b_grad;
        };
        let add_ln = |dst: &mut LayerNorm, src: &LayerNorm| {
            dst.ensure_grads();
            dst.g_grad += &src.g_grad;
            dst.b_grad += &src.b_grad;
        };

        for (dst, src) in self.blocks.iter_mut().zip(other.blocks.iter()) {
            add_ln(&mut dst.ln1, &src.ln1);
            add_linear(&mut dst.attn.w_q, &src.attn.w_q);
            add_linear(&mut dst.attn.w_k, &src.attn.w_k);
            add_linear(&mut dst.attn.w_v, &src.attn.w_v);
            add_linear(&mut dst.attn.w_o, &src.attn.w_o);
            add_ln(&mut dst.ln2, &src.ln2);
            add_linear(&mut dst.ffn.fc1, &src.ffn.fc1);
            add_linear(&mut dst.ffn.fc2, &src.ffn.fc2);
        }

        add_ln(&mut self.final_norm, &other.final_norm);
        add_linear(&mut self.lm_head, &other.lm_head);
    }

    /// Apply one optimizer step. If `opt` is Adam with `grad_clip = Some(c)`,
    /// global L2 norm is clipped to `c` before stepping.
    pub fn apply_grads(&mut self, opt: Optimizer) {
        // With tied embeddings, the LM head's weight gradient is the gradient
        // w.r.t. the *shared* embedding matrix. Fold it into `embeddings_grad`
        // before clipping, then zero `lm_head.w_grad` so the head's own
        // optimizer step only updates the bias.
        if self.tie_embeddings {
            self.ensure_embeddings_grad();
            self.lm_head.ensure_grads();
            if self.lm_head.w_grad.shape() == self.embeddings_grad.shape() {
                self.embeddings_grad += &self.lm_head.w_grad;
                self.lm_head.w_grad.fill(0.0);
            }
        }

        if let Optimizer::Adam {
            grad_clip: Some(c), ..
        } = opt
        {
            let norm = self.grad_sq_norm().sqrt();
            if norm > c && norm > 0.0 {
                self.scale_all_grads(c / norm);
            }
        }

        // Embeddings step
        self.ensure_embeddings_grad();
        match opt {
            Optimizer::Sgd { lr } => {
                self.embeddings.scaled_add(-lr, &self.embeddings_grad);
            }
            Optimizer::Adam {
                lr, beta1, beta2, eps, ..
            } => {
                // Use the same Adam logic as in `nn`. Inline to avoid exposing
                // the helper publicly.
                let state = &mut self.embeddings_state;
                if state.m.shape() != self.embeddings.shape() {
                    state.m = Array2::zeros(self.embeddings.raw_dim());
                    state.v = Array2::zeros(self.embeddings.raw_dim());
                    state.t = 0;
                }
                state.t += 1;
                let bc1 = 1.0 - beta1.powi(state.t as i32);
                let bc2 = 1.0 - beta2.powi(state.t as i32);
                for ((p, g), (m, v)) in self
                    .embeddings
                    .iter_mut()
                    .zip(self.embeddings_grad.iter())
                    .zip(state.m.iter_mut().zip(state.v.iter_mut()))
                {
                    *m = beta1 * *m + (1.0 - beta1) * *g;
                    *v = beta2 * *v + (1.0 - beta2) * *g * *g;
                    let m_hat = *m / bc1;
                    let v_hat = *v / bc2;
                    *p -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
        }
        self.embeddings_grad.fill(0.0);

        for block in &mut self.blocks {
            block.apply_grads(opt);
        }
        self.final_norm.apply_grads(opt);
        self.lm_head.apply_grads(opt);

        // Re-tie: keep the LM head's weight matrix equal to the (now updated)
        // embeddings. The head's own weight gradient was zeroed above, so its
        // Adam state for the weight matrix never moves.
        if self.tie_embeddings {
            self.sync_lm_head_from_embeddings();
        }
    }

    // -- generation -----------------------------------------------------------

    pub fn complete(&self, tokens: &[usize]) -> Result<String, String> {
        self.complete_with_limit(tokens, 64)
    }

    pub fn complete_with_limit(
        &self,
        tokens: &[usize],
        max_new_tokens: usize,
    ) -> Result<String, String> {
        self.generate(tokens, &SamplingConfig::greedy(max_new_tokens), &mut rand::rng())
    }

    /// Generate continuation tokens. Returns a space-joined string of newly
    /// generated token texts (excluding the prompt).
    pub fn generate<R: Rng>(
        &self,
        prompt_tokens: &[usize],
        cfg: &SamplingConfig,
        rng: &mut R,
    ) -> Result<String, String> {
        let texts = self.generate_tokens(prompt_tokens, cfg, rng)?;
        Ok(texts.join(" "))
    }

    pub fn generate_tokens<R: Rng>(
        &self,
        prompt_tokens: &[usize],
        cfg: &SamplingConfig,
        rng: &mut R,
    ) -> Result<Vec<String>, String> {
        let ids = self.generate_token_ids(prompt_tokens, cfg, rng)?;
        Ok(ids
            .into_iter()
            .filter_map(|id| self.get_token_by_id(id).map(|s| s.to_string()))
            .collect())
    }

    /// Generate continuation token ids only (excluding the prompt). Stops on
    /// `<stop>` or after `cfg.max_new_tokens`.
    pub fn generate_token_ids<R: Rng>(
        &self,
        prompt_tokens: &[usize],
        cfg: &SamplingConfig,
        rng: &mut R,
    ) -> Result<Vec<usize>, String> {
        if prompt_tokens.is_empty() {
            return Ok(Vec::new());
        }
        let stop = self.get_stop_token_id();
        let mut seq: Vec<usize> = prompt_tokens.to_vec();
        let mut kv_cache = self.new_kv_cache();
        // Prime cache with the prompt and keep the logits of the last prompt
        // position as the first distribution to sample from.
        let mut logits_opt: Option<Array1<f32>> = None;
        for (pos, &id) in prompt_tokens.iter().enumerate() {
            logits_opt = Some(self.step_with_cache(id, pos, &mut kv_cache));
        }
        let mut logits = logits_opt.expect("prompt is non-empty");
        let mut out: Vec<usize> = Vec::new();
        for _ in 0..cfg.max_new_tokens {
            let mut next = sample_from_logits(&logits, cfg, rng);
            // Avoid an immediately empty completion when the very first choice
            // is `<stop>`. Prefer the best non-stop token once, then allow
            // normal stopping behavior on subsequent steps.
            if out.is_empty() && next == stop {
                if let Some(alt) = argmax_excluding(&logits, stop) {
                    next = alt;
                }
            }
            if next == stop {
                break;
            }
            out.push(next);
            seq.push(next);
            let pos = seq.len() - 1;
            logits = self.step_with_cache(next, pos, &mut kv_cache);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    #[test]
    fn forward_logits_has_vocab_size() {
        crate::nn::seed(0);
        let mut m = Model::new(8, 1, 2);
        m.register_token("hello");
        m.register_token("world");
        m.initialize_embeddings();
        let ids = vec![
            m.get_token_id("hello").unwrap(),
            m.get_token_id("world").unwrap(),
        ];
        let logits = m.forward_logits(&ids);
        assert_eq!(logits.len(), m.vocab_size());
    }

    #[test]
    fn train_sequence_overfits_with_adam() {
        crate::nn::seed(1);
        let mut m = Model::new(16, 2, 4);
        for tok in ["a", "b", "c"] {
            m.register_token(tok);
        }
        m.initialize_embeddings();
        let a = m.get_token_id("a").unwrap();
        let b = m.get_token_id("b").unwrap();
        let c = m.get_token_id("c").unwrap();
        let seq = vec![a, b, c, a, b, c, a, b, c];

        let opt = Optimizer::adam(0.01);
        let l0 = m.train_sequence(&seq);
        m.apply_grads(opt);
        let mut last = l0;
        for _ in 0..200 {
            last = m.train_sequence(&seq);
            m.apply_grads(opt);
        }
        assert!(
            last < l0 * 0.3,
            "loss did not decrease enough: l0={} last={}",
            l0,
            last
        );
    }

    #[test]
    fn save_load_roundtrip_bincode() {
        crate::nn::seed(2);
        let dir = std::env::temp_dir().join("blip_test_model.bin");
        let mut m = Model::new(8, 1, 2);
        m.register_token("x");
        m.initialize_embeddings();
        m.save(dir.to_str().unwrap()).unwrap();
        let m2 = Model::load(dir.to_str().unwrap()).unwrap();
        assert_eq!(m2.vocab_size(), m.vocab_size());
        assert_eq!(m2.embedding_dim(), m.embedding_dim());
        assert_eq!(m2.n_heads(), m.n_heads());
        let _ = std::fs::remove_file(dir);
    }

    #[test]
    fn save_load_roundtrip_json() {
        crate::nn::seed(3);
        let dir = std::env::temp_dir().join("blip_test_model.json");
        let mut m = Model::new(8, 1, 2);
        m.register_token("x");
        m.initialize_embeddings();
        m.save(dir.to_str().unwrap()).unwrap();
        let m2 = Model::load(dir.to_str().unwrap()).unwrap();
        assert_eq!(m2.vocab_size(), m.vocab_size());
        let _ = std::fs::remove_file(dir);
    }

    #[test]
    fn sampling_runs() {
        crate::nn::seed(4);
        let mut m = Model::new(8, 1, 2);
        for tok in ["a", "b", "c"] {
            m.register_token(tok);
        }
        m.initialize_embeddings();
        let a = m.get_token_id("a").unwrap();
        let cfg = SamplingConfig {
            temperature: 0.8,
            top_k: Some(3),
            top_p: Some(0.9),
            max_new_tokens: 5,
        };
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let out = m.generate(&[a], &cfg, &mut rng).unwrap();
        // Just make sure it returns without panicking; output may be empty if
        // <stop> is sampled immediately.
        let _ = out;
    }

    #[test]
    fn step_with_cache_matches_forward_logits_on_prompt_tail() {
        crate::nn::seed(5);
        let mut m = Model::new(16, 2, 4);
        for tok in ["i", "am", "blip"] {
            m.register_token(tok);
        }
        m.initialize_embeddings();

        let bos = m.get_begin_token_id();
        let i = m.get_token_id("i").unwrap();
        let am = m.get_token_id("am").unwrap();
        let blip = m.get_token_id("blip").unwrap();
        let prompt = vec![bos, i, am, blip];

        let full = m.forward_logits(&prompt);
        let mut cache = m.new_kv_cache();
        let mut stepped = None;
        for (pos, &id) in prompt.iter().enumerate() {
            stepped = Some(m.step_with_cache(id, pos, &mut cache));
        }
        let stepped = stepped.unwrap();

        assert_eq!(full.len(), stepped.len());
        for idx in 0..full.len() {
            let diff = (full[idx] - stepped[idx]).abs();
            assert!(diff < 1e-5, "logit {} mismatch: {} vs {}", idx, full[idx], stepped[idx]);
        }
    }

    #[test]
    fn generation_avoids_immediate_stop_on_first_token() {
        crate::nn::seed(6);
        let mut m = Model::new(8, 1, 2);
        m.register_token("a");
        m.initialize_embeddings();

        // Force deterministic logits where `<stop>` is highest and "a" is
        // second-highest regardless of prompt state.
        m.lm_head.weights.fill(0.0);
        m.lm_head.bias.fill(-10.0);
        let stop = m.get_stop_token_id();
        let a = m.get_token_id("a").unwrap();
        m.lm_head.bias[stop] = 10.0;
        m.lm_head.bias[a] = 5.0;

        let cfg = SamplingConfig::greedy(1);
        let bos = m.get_begin_token_id();
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        let ids = m.generate_token_ids(&[bos], &cfg, &mut rng).unwrap();
        assert_eq!(ids, vec![a]);
    }

    #[test]
    fn tied_embeddings_keep_lm_head_in_sync_after_step() {
        crate::nn::seed(7);
        let mut m = Model::new(8, 1, 2);
        for tok in ["i", "am", "blip"] {
            m.register_token(tok);
        }
        m.initialize_embeddings();
        assert!(m.tie_embeddings);

        // After init, lm_head.weights mirrors the embedding table.
        assert_eq!(m.lm_head.weights, m.embeddings);

        let bos = m.get_begin_token_id();
        let i = m.get_token_id("i").unwrap();
        let am = m.get_token_id("am").unwrap();
        let blip = m.get_token_id("blip").unwrap();
        let stop = m.get_stop_token_id();

        let _ = m.train_sequence(&[bos, i, am, blip, stop]);

        // Both gradient buffers are non-zero before apply_grads.
        let head_grad_norm: f32 = m.lm_head.w_grad.iter().map(|v| v * v).sum();
        let emb_grad_norm: f32 = m.embeddings_grad.iter().map(|v| v * v).sum();
        assert!(head_grad_norm > 0.0, "lm_head should have weight grads");
        assert!(emb_grad_norm > 0.0, "embeddings should have grads");

        m.apply_grads(crate::nn::Optimizer::adam(0.01));

        // After the step, lm_head.weights still equals embeddings (re-tied),
        // and the head weight grad has been folded in / zeroed.
        assert_eq!(m.lm_head.weights, m.embeddings);
        let head_grad_after: f32 = m.lm_head.w_grad.iter().map(|v| v * v).sum();
        assert_eq!(head_grad_after, 0.0);
    }
}
