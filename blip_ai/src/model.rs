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
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

// Special tokens
pub const TOKEN_UNKNOWN: &str = "<unk>";
pub const TOKEN_STOP: &str = "<stop>";
pub const TOKEN_TOOL: &str = "<tool>";
pub const TOKEN_BEGIN: &str = "<bos>";

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
    pub xs: Vec<Array1<f32>>,
    pub qs: Vec<Array1<f32>>,
    pub ks: Vec<Array1<f32>>,
    pub vs: Vec<Array1<f32>>,
    /// Per (head, query position) attention distribution over keys 0..=i.
    /// `attn[h][i][j]` = weight from query i to key j in head h.
    pub attn: Vec<Vec<Vec<f32>>>,
    /// Per-position concatenated context (input to W_o).
    pub context: Vec<Array1<f32>>,
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

    pub fn forward(&self, xs: &[Array1<f32>]) -> Vec<Array1<f32>> {
        self.forward_train(xs).0
    }

    pub fn forward_train(&self, xs: &[Array1<f32>]) -> (Vec<Array1<f32>>, AttentionCache) {
        let dim = self.dim();
        let head_dim = self.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = xs.len();

        let qs: Vec<Array1<f32>> = xs.iter().map(|x| self.w_q.forward(x)).collect();
        let ks: Vec<Array1<f32>> = xs.iter().map(|x| self.w_k.forward(x)).collect();
        let vs: Vec<Array1<f32>> = xs.iter().map(|x| self.w_v.forward(x)).collect();

        let mut attn: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); n]; self.n_heads];
        let mut context: Vec<Array1<f32>> = vec![Array1::zeros(dim); n];

        for h in 0..self.n_heads {
            let off = h * head_dim;
            for i in 0..n {
                // scores[j] = (q_i_h . k_j_h) * scale, for j in 0..=i (causal)
                let mut scores = vec![0.0_f32; i + 1];
                for j in 0..=i {
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += qs[i][off + d] * ks[j][off + d];
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

                // ctx[h][i] = sum_j scores[j] * v_j_h
                for j in 0..=i {
                    let w = scores[j];
                    for d in 0..head_dim {
                        context[i][off + d] += w * vs[j][off + d];
                    }
                }
                attn[h][i] = scores;
            }
        }

        let outputs: Vec<Array1<f32>> = context.iter().map(|c| self.w_o.forward(c)).collect();

        (
            outputs,
            AttentionCache {
                xs: xs.to_vec(),
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
        d_outs: &[Array1<f32>],
        cache: &AttentionCache,
    ) -> Vec<Array1<f32>> {
        let dim = self.dim();
        let head_dim = self.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = d_outs.len();

        // d_context per position via W_o backward
        let mut d_context: Vec<Array1<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            d_context.push(self.w_o.backward(&d_outs[i], &cache.context[i]));
        }

        let mut d_q: Vec<Array1<f32>> = (0..n).map(|_| Array1::zeros(dim)).collect();
        let mut d_k: Vec<Array1<f32>> = (0..n).map(|_| Array1::zeros(dim)).collect();
        let mut d_v: Vec<Array1<f32>> = (0..n).map(|_| Array1::zeros(dim)).collect();

        for h in 0..self.n_heads {
            let off = h * head_dim;
            for i in 0..n {
                // d_attn[j] = sum_d d_context[i][off+d] * v_j[off+d]
                // d_v[j][off+d] += attn[j] * d_context[i][off+d]
                let mut d_attn = vec![0.0_f32; i + 1];
                for j in 0..=i {
                    let mut s = 0.0_f32;
                    for d in 0..head_dim {
                        let dc = d_context[i][off + d];
                        s += dc * cache.vs[j][off + d];
                        d_v[j][off + d] += cache.attn[h][i][j] * dc;
                    }
                    d_attn[j] = s;
                }
                // softmax backward
                let mut dot = 0.0_f32;
                for j in 0..=i {
                    dot += cache.attn[h][i][j] * d_attn[j];
                }
                let mut d_score = vec![0.0_f32; i + 1];
                for k in 0..=i {
                    d_score[k] = cache.attn[h][i][k] * (d_attn[k] - dot);
                }
                // d_q[i][off+d] += scale * sum_k d_score[k] * k_k[off+d]
                // d_k[k][off+d] += scale * d_score[k] * q_i[off+d]
                for k in 0..=i {
                    let s_k = scale * d_score[k];
                    for d in 0..head_dim {
                        d_q[i][off + d] += s_k * cache.ks[k][off + d];
                        d_k[k][off + d] += s_k * cache.qs[i][off + d];
                    }
                }
            }
        }

        let mut d_xs: Vec<Array1<f32>> = (0..n).map(|_| Array1::zeros(dim)).collect();
        for i in 0..n {
            let dq = self.w_q.backward(&d_q[i], &cache.xs[i]);
            let dk = self.w_k.backward(&d_k[i], &cache.xs[i]);
            let dv = self.w_v.backward(&d_v[i], &cache.xs[i]);
            d_xs[i] = dq + dk + dv;
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
        for h in 0..self.n_heads {
            let off = h * head_dim;
            let mut scores = vec![0.0_f32; n];
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
    pub ln1_outs: Vec<Array1<f32>>,
    pub attn_cache: AttentionCache,
    pub attn_drop: Vec<crate::nn::DropoutCache>,
    pub residual1: Vec<Array1<f32>>,
    pub ln2_caches: Vec<LayerNormCache>,
    pub ln2_outs: Vec<Array1<f32>>,
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

    pub fn forward(&self, xs: &[Array1<f32>]) -> Vec<Array1<f32>> {
        let normed1: Vec<Array1<f32>> = xs.iter().map(|x| self.ln1.forward(x)).collect();
        let attn_out = self.attn.forward(&normed1);
        let res1: Vec<Array1<f32>> = xs.iter().zip(attn_out.iter()).map(|(x, a)| x + a).collect();
        let normed2: Vec<Array1<f32>> = res1.iter().map(|x| self.ln2.forward(x)).collect();
        let ffn_out: Vec<Array1<f32>> = normed2.iter().map(|x| self.ffn.forward(x)).collect();
        res1.iter().zip(ffn_out.iter()).map(|(r, f)| r + f).collect()
    }

    pub fn forward_train(&self, xs: &[Array1<f32>]) -> (Vec<Array1<f32>>, DecoderBlockCache) {
        let n = xs.len();
        let mut ln1_caches = Vec::with_capacity(n);
        let mut ln1_outs = Vec::with_capacity(n);
        for x in xs {
            let (y, c) = self.ln1.forward_train(x);
            ln1_outs.push(y);
            ln1_caches.push(c);
        }
        let (attn_outs_raw, attn_cache) = self.attn.forward_train(&ln1_outs);
        let mut attn_drop = Vec::with_capacity(n);
        let mut attn_outs = Vec::with_capacity(n);
        for a in &attn_outs_raw {
            let (y, c) = crate::nn::dropout_forward(a, self.dropout);
            attn_outs.push(y);
            attn_drop.push(c);
        }
        let residual1: Vec<Array1<f32>> =
            xs.iter().zip(attn_outs.iter()).map(|(x, a)| x + a).collect();

        let mut ln2_caches = Vec::with_capacity(n);
        let mut ln2_outs = Vec::with_capacity(n);
        for r in &residual1 {
            let (y, c) = self.ln2.forward_train(r);
            ln2_outs.push(y);
            ln2_caches.push(c);
        }

        let mut ffn_caches = Vec::with_capacity(n);
        let mut ffn_drop = Vec::with_capacity(n);
        let mut ffn_outs = Vec::with_capacity(n);
        for x in &ln2_outs {
            let (y_raw, c_ffn) = self.ffn.forward_train(x);
            let (y, c_drop) = crate::nn::dropout_forward(&y_raw, self.dropout);
            ffn_outs.push(y);
            ffn_caches.push(c_ffn);
            ffn_drop.push(c_drop);
        }
        let outs: Vec<Array1<f32>> = residual1
            .iter()
            .zip(ffn_outs.iter())
            .map(|(r, f)| r + f)
            .collect();

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
        d_outs: &[Array1<f32>],
        cache: &DecoderBlockCache,
    ) -> Vec<Array1<f32>> {
        let n = d_outs.len();
        // d_out flows into both branches of the FFN residual.
        let mut d_residual1 = d_outs.to_vec();

        let mut d_ln2_out: Vec<Array1<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            // backprop through FFN dropout, then FFN itself.
            let d_ffn_raw = crate::nn::dropout_backward(&d_outs[i], &cache.ffn_drop[i]);
            d_ln2_out.push(self.ffn.backward(&d_ffn_raw, &cache.ffn_caches[i]));
        }
        for i in 0..n {
            let d_r = self.ln2.backward(&d_ln2_out[i], &cache.ln2_caches[i]);
            d_residual1[i] = &d_residual1[i] + &d_r;
        }

        // d_residual1 flows into both branches of the attention residual.
        let mut d_x = d_residual1.clone();
        let d_attn_out_raw: Vec<Array1<f32>> = (0..n)
            .map(|i| crate::nn::dropout_backward(&d_residual1[i], &cache.attn_drop[i]))
            .collect();
        let d_ln1_out = self.attn.backward(&d_attn_out_raw, &cache.attn_cache);
        for i in 0..n {
            let d = self.ln1.backward(&d_ln1_out[i], &cache.ln1_caches[i]);
            d_x[i] = &d_x[i] + &d;
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
    /// Lazily grown cache of sinusoidal positional encodings. Never serialized.
    #[serde(skip, default)]
    pe_cache: std::cell::RefCell<Vec<Array1<f32>>>,
}

impl Model {
    fn is_supported_version(v: &str) -> bool {
        matches!(v, "0.4.0" | MODEL_VERSION)
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
            pe_cache: std::cell::RefCell::new(Vec::new()),
        };

        model.register_token(TOKEN_UNKNOWN);
        model.register_token(TOKEN_STOP);
        model.register_token(TOKEN_TOOL);
        model.register_token(TOKEN_BEGIN);
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
        let specials = [TOKEN_UNKNOWN, TOKEN_STOP, TOKEN_TOOL, TOKEN_BEGIN];
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
    }

    fn ensure_embeddings_grad(&mut self) {
        if self.embeddings_grad.shape() != self.embeddings.shape() {
            self.embeddings_grad = Array2::zeros(self.embeddings.raw_dim());
        }
    }

    /// Extend `pe_cache` to cover at least `seq_len` positions. Positions
    /// already cached are never recomputed.
    fn ensure_pe(&self, seq_len: usize) {
        let current = self.pe_cache.borrow().len();
        if seq_len > current {
            let mut cache = self.pe_cache.borrow_mut();
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

    fn embed_sequence(&self, ids: &[usize]) -> Vec<Array1<f32>> {
        self.ensure_pe(ids.len());
        let pe = self.pe_cache.borrow();
        ids.iter()
            .enumerate()
            .map(|(i, &id)| {
                let mut v = self.embeddings.row(id).to_owned();
                v += &pe[i];
                v
            })
            .collect()
    }

    fn embed_token_at(&self, id: usize, pos: usize) -> Array1<f32> {
        self.ensure_pe(pos + 1);
        let pe = self.pe_cache.borrow();
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
        let last = xs.last().expect("non-empty input");
        let normed = self.final_norm.forward(last);
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

        let mut ln_caches: Vec<LayerNormCache> = Vec::with_capacity(xs.len());
        let mut normed: Vec<Array1<f32>> = Vec::with_capacity(xs.len());
        for x in &xs {
            let (y, c) = self.final_norm.forward_train(x);
            normed.push(y);
            ln_caches.push(c);
        }

        let mut logits: Vec<Array1<f32>> = Vec::with_capacity(xs.len());
        let mut head_caches: Vec<Array1<f32>> = Vec::with_capacity(xs.len());
        for n in &normed {
            let (y, c) = self.lm_head.forward_train(n);
            logits.push(y);
            head_caches.push(c);
        }

        let mut total_loss = 0.0_f32;
        let mut d_logits: Vec<Array1<f32>> = Vec::with_capacity(xs.len());
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

        let mut d_xs: Vec<Array1<f32>> = Vec::with_capacity(xs.len());
        for i in 0..d_logits.len() {
            let d_normed = self.lm_head.backward(&d_logits[i], &head_caches[i]);
            let d_x = self.final_norm.backward(&d_normed, &ln_caches[i]);
            d_xs.push(d_x);
        }
        for (block, cache) in self.blocks.iter_mut().zip(block_caches.iter()).rev() {
            d_xs = block.backward(&d_xs, cache);
        }

        self.ensure_embeddings_grad();
        for (i, &id) in inputs.iter().enumerate() {
            let mut row = self.embeddings_grad.row_mut(id);
            row += &d_xs[i];
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
            let normed = self.final_norm.forward(&xs[i]);
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

    /// Apply one optimizer step. If `opt` is Adam with `grad_clip = Some(c)`,
    /// global L2 norm is clipped to `c` before stepping.
    pub fn apply_grads(&mut self, opt: Optimizer) {
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
}
