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
//! Checkpoints are JSON (`.json`) for easier inspection.

use crate::nn::{
    softmax, softmax_cross_entropy, AdamState2, FeedForward, FeedForwardBatchCache,
    LayerNorm, LayerNormBatchCache, Linear, Optimizer,
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

        let qs = self.w_q.forward_batch(xs);
        let ks = self.w_k.forward_batch(xs);
        let vs = self.w_v.forward_batch(xs);

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

        let outputs = self.w_o.forward_batch(&context);

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

        // d_context via batched W_o backward
        let d_context = self.w_o.backward_batch(d_outs, &cache.context);

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

        let dq_x = self.w_q.backward_batch(&d_q, &cache.xs);
        let dk_x = self.w_k.backward_batch(&d_k, &cache.xs);
        let dv_x = self.w_v.backward_batch(&d_v, &cache.xs);
        let d_xs = dq_x + dk_x + dv_x;
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
}

pub struct DecoderBlockCache {
    pub ln1_cache: LayerNormBatchCache,
    pub attn_cache: AttentionCache,
    pub attn_drop: Vec<crate::nn::DropoutCache>,
    pub residual1: Array2<f32>,
    pub ln2_cache: LayerNormBatchCache,
    pub ffn_cache: FeedForwardBatchCache,
    pub ffn_drop: Vec<crate::nn::DropoutCache>,
}

impl DecoderBlock {
    pub fn new(dim: usize, n_heads: usize, ffn_hidden: usize) -> Self {
        Self {
            ln1: LayerNorm::new(dim),
            attn: SelfAttention::new(dim, n_heads),
            ln2: LayerNorm::new(dim),
            ffn: FeedForward::new(dim, ffn_hidden),
        }
    }

    pub fn forward(&self, xs: &Array2<f32>) -> Array2<f32> {
        let normed1 = self.ln1.forward_batch(xs);
        let attn_out = self.attn.forward(&normed1);
        let mut res1 = xs.clone();
        res1 += &attn_out;
        let normed2 = self.ln2.forward_batch(&res1);
        let ffn_out = self.ffn.forward_batch(&normed2);
        res1 + ffn_out
    }

    pub fn forward_train(&self, xs: &Array2<f32>, dropout: f32) -> (Array2<f32>, DecoderBlockCache) {
        let n = xs.nrows();
        let dim = xs.ncols();
        let (ln1_outs, ln1_cache) = self.ln1.forward_batch_train(xs);
        let (attn_outs_raw, attn_cache) = self.attn.forward_train(&ln1_outs);
        let mut attn_drop = Vec::with_capacity(n);
        let mut attn_outs = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let (y, c) = crate::nn::dropout_forward(&attn_outs_raw.row(i).to_owned(), dropout);
            attn_outs.row_mut(i).assign(&y);
            attn_drop.push(c);
        }
        let mut residual1 = xs.clone();
        residual1 += &attn_outs;

        let (ln2_outs, ln2_cache) = self.ln2.forward_batch_train(&residual1);
        let (ffn_outs_raw, ffn_cache) = self.ffn.forward_batch_train(&ln2_outs);
        let mut ffn_drop = Vec::with_capacity(n);
        let mut ffn_outs = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let (y, c_drop) = crate::nn::dropout_forward(&ffn_outs_raw.row(i).to_owned(), dropout);
            ffn_outs.row_mut(i).assign(&y);
            ffn_drop.push(c_drop);
        }
        let outs = &residual1 + &ffn_outs;

        (
            outs,
            DecoderBlockCache {
                ln1_cache,
                attn_cache,
                attn_drop,
                residual1,
                ln2_cache,
                ffn_cache,
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
            d_ln2_out.row_mut(i).assign(&d_ffn_raw);
        }
        let d_ln2_in = self.ffn.backward_batch(&d_ln2_out, &cache.ffn_cache);
        let d_r = self.ln2.backward_batch(&d_ln2_in, &cache.ln2_cache);
        d_residual1 += &d_r;

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
        let d = self.ln1.backward_batch(&d_ln1_out, &cache.ln1_cache);
        d_x += &d;
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
            let sum_k: f32 = probs.iter().sum();
            if sum_k > 0.0 {
                for p in &mut probs {
                    *p /= sum_k;
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainingMetadata {
    pub embedding_dim: usize,
    pub depth: usize,
    pub n_heads: usize,
    pub pretrain_epochs: usize,
    pub pretrain_lr: f32,
    pub pretrain_batch_size: usize,
    pub pretrain_warmup: usize,
    pub pretrain_min_lr: f32,
    pub num_epochs: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub dropout: f32,
    pub val_split: f32,
    pub warmup_steps: usize,
    pub min_lr: f32,
    pub grad_clip: f32,
    pub deterministic: bool,
    pub seed: u64,
    pub checkpoint_every: usize,
    pub min_count: u32,
    pub seq_length: usize,
    pub pretraining_files: Vec<String>,
    pub tuning_files: Vec<String>,
    pub output_file: String,
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
    /// Runtime-only dropout probability used during training.
    #[serde(skip, default)]
    dropout: f32,
    pub blocks: Vec<DecoderBlock>,
    pub final_norm: LayerNorm,
    pub lm_head: Linear,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_metadata: Option<TrainingMetadata>,
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
            dropout: self.dropout,
            blocks: self.blocks.clone(),
            final_norm: self.final_norm.clone(),
            lm_head: self.lm_head.clone(),
            training_metadata: self.training_metadata.clone(),
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
    fn insert_token_if_missing(&mut self, id: &str, usage_count: u32) -> usize {
        if let Some(index) = self.get_token_id(id) {
            return index;
        }
        let new_index = self.tokens.len();
        self.tokens.push(Token {
            text: id.to_string(),
            usage_count,
        });
        new_index
    }

    fn is_supported_version(v: &str) -> bool {
        // Only the current version is supported. The `<bos>` token was
        // removed in 0.8.0, which shifted token ids — older checkpoints
        // would silently misinterpret special-token ids if loaded.
        v == MODEL_VERSION
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
            dropout: 0.0,
            blocks,
            final_norm: LayerNorm::new(embedding_dim),
            lm_head: Linear::new(embedding_dim, 1),
            training_metadata: None,
            pe_cache: std::sync::RwLock::new(Vec::new()),
        };

        model.insert_token_if_missing(TOKEN_UNKNOWN, 0);
        model.insert_token_if_missing(TOKEN_STOP, 0);
        model.insert_token_if_missing(TOKEN_TOOL, 0);
        model.insert_token_if_missing(TOKEN_USER, 0);
        model.insert_token_if_missing(TOKEN_AI, 0);
        for token in default_token_texts() {
            model.insert_token_if_missing(&token, 0);
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
        self.dropout = p.clamp(0.0, 1.0);
    }

    pub fn set_training_metadata(&mut self, metadata: TrainingMetadata) {
        self.training_metadata = Some(metadata);
    }

    /// Loads from disk. Only `.json` checkpoints are supported.
    pub fn load(file_name: &str) -> Result<Self, String> {
        if Path::new(file_name)
            .extension()
            .and_then(|s| s.to_str())
            != Some("json")
        {
            return Err(format!(
                "Unsupported checkpoint format for '{}'. Use a .json file.",
                file_name
            ));
        }
        let bytes = std::fs::read(file_name).map_err(|e| e.to_string())?;
        let mut model: Model = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if !Self::is_supported_version(&model.version) {
            return Err(format!(
                "Model version mismatch: expected {} (or compatible prior version), got {}",
                MODEL_VERSION, model.version
            ));
        }
        // Normalize compatible checkpoints to the current version when loaded.
        model.version = MODEL_VERSION.to_string();
        model.embeddings_grad = Array2::zeros(model.embeddings.raw_dim());
        model.dropout = 0.0;
        Ok(model)
    }

    pub fn save(&self, file_name: &str) -> Result<(), String> {
        if Path::new(file_name)
            .extension()
            .and_then(|s| s.to_str())
            != Some("json")
        {
            return Err(format!(
                "Unsupported checkpoint format for '{}'. Use a .json file.",
                file_name
            ));
        }
        if let Some(parent) = Path::new(file_name).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        let file = std::fs::File::create(file_name).map_err(|e| e.to_string())?;
        serde_json::to_writer(file, self).map_err(|e| e.to_string())?;
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
        self.insert_token_if_missing(id, 1)
    }

    /// Drop tokens with `usage_count < min_count` from the vocabulary. Special
    /// tokens (`<unk>`, `<stop>`, `<tool>`, `<user>`, `<ai>`) are always
    /// retained. Must be called before `initialize_embeddings`. Returns the
    /// number of tokens removed.
    pub fn trim_vocab(&mut self, min_count: u32) -> usize {
        if min_count <= 1 {
            return 0;
        }
        let before = self.tokens.len();
        let specials = [TOKEN_UNKNOWN, TOKEN_STOP, TOKEN_TOOL, TOKEN_USER, TOKEN_AI];
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
        let normed = self.final_norm.forward_batch(&xs);
        let logits = self.lm_head.forward_batch(&normed);
        logits.row(logits.nrows() - 1).to_owned()
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
            let (out, cache) = block.forward_train(&xs, self.dropout);
            xs = out;
            block_caches.push(cache);
        }

        let (normed, ln_cache) = self.final_norm.forward_batch_train(&xs);
        let (logits, head_cache) = self.lm_head.forward_batch_train(&normed);

        let mut total_loss = 0.0_f32;
        let mut d_logits = Array2::<f32>::zeros(logits.raw_dim());
        for (i, target) in targets.iter().enumerate() {
            let (loss, d) = softmax_cross_entropy(&logits.row(i).to_owned(), *target);
            total_loss += loss;
            d_logits.row_mut(i).assign(&d);
        }
        let avg_loss = total_loss / targets.len() as f32;

        let scale = 1.0 / targets.len() as f32;
        d_logits *= scale;

        let d_normed = self.lm_head.backward_batch(&d_logits, &head_cache);
        let mut d_xs = self.final_norm.backward_batch(&d_normed, &ln_cache);
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
        let normed = self.final_norm.forward_batch(&xs);
        let logits = self.lm_head.forward_batch(&normed);
        let mut total = 0.0_f32;
        for (i, &t) in targets.iter().enumerate() {
            let p = softmax(&logits.row(i).to_owned());
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
            Optimizer::Adam {
                lr, beta1, beta2, eps, ..
            } => {
                crate::nn::adam_step2(
                    &mut self.embeddings,
                    &self.embeddings_grad,
                    &mut self.embeddings_state,
                    lr,
                    beta1,
                    beta2,
                    eps,
                );
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
            let next = sample_from_logits(&logits, cfg, rng);
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

        let user = m.get_user_token_id();
        let i = m.get_token_id("i").unwrap();
        let am = m.get_token_id("am").unwrap();
        let blip = m.get_token_id("blip").unwrap();
        let prompt = vec![user, i, am, blip];

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

}
