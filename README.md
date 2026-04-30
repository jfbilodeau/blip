# Blip

A tiny generative-AI language model built **from the ground up** in pure Rust.
Decoder-only transformer (multi-head causal self-attention + feed-forward +
pre-norm residual blocks) with hand-written forward + backward passes on
`ndarray`. No `tch`, no `candle`, no autograd — just linear algebra.

The point is to *understand the math* by writing every piece by hand: Xavier
init, layer norm forward + backward, scaled dot-product attention with a
manually derived softmax backward, FFN, residuals, Adam with global L2
gradient clipping, cosine-with-warmup learning-rate schedules, and softmax
cross-entropy.

## Workspace layout

| Crate          | Purpose                                                  |
| -------------- | -------------------------------------------------------- |
| `blip_ai`      | Library: `nn`, `model` (Transformer), `tokenizer`, `trainer` |
| `blip_trainer` | CLI: trains a model from a text file                     |
| `blip`         | CLI: loads a model and generates / runs a REPL           |

## Quickstart

Train on the bundled sample (`training/basic.txt`):

```pwsh
cargo run --release -p blip_trainer -- `
  -e 64 -d 2 --n-heads 4 -n 400 --seed 42 `
  --dropout 0.1 --warmup-steps 100 --min-lr 0.0001 `
    -o models/basic.bin
```

Generate:

```pwsh
cargo run --release -p blip -- -p "I am" -m 32 -t 0.8 --seed 1
```

REPL:

```pwsh
cargo run --release -p blip -- --repl -t 0.8 --seed 1
```

## Trainer flags

| Flag              | Default               | Meaning                                     |
| ----------------- | --------------------- | ------------------------------------------- |
| `-e, --embedding-dim` | 128               | Embedding / model dimension                 |
| `-d, --depth`     | 2                     | Number of decoder blocks                    |
| `--n-heads`       | 4                     | Attention heads (must divide `embedding_dim`) |
| `-n, --epochs`    | 200                   | Training epochs                             |
| `-l, --learning-rate` | 0.001             | Adam learning rate                          |
| `-b, --batch-size` | 1                    | Sequences per optimizer step                |
| `--dropout`       | 0.0                   | Training-time dropout on attention/FFN outputs |
| `--val-split`     | 0.0                   | Validation split fraction                   |
| `--warmup-steps`  | 0                     | Warmup steps for cosine LR (`0` = constant LR) |
| `--min-lr`        | 0.0                   | Final LR floor for cosine decay             |
| `--min-count`     | 1                     | Drop tokens below this usage_count (specials always kept) |
| `--seed`          | 0 (= OS entropy)      | RNG seed for init / shuffling               |
| `--checkpoint-every` | 0 (= end only)     | Save every N epochs                         |
| `-i, --input-files`  | `training/basic.txt` | Training corpus(es)                       |
| `-o, --output-file`  | `models/basic.bin` | Output checkpoint (`.json` → JSON, else bincode) |

## Inference flags

| Flag              | Default               | Meaning                                     |
| ----------------- | --------------------- | ------------------------------------------- |
| `-f, --model-file` | `models/basic.bin`   | Checkpoint to load                          |
| `-p, --prompt`    | `Who are you?`        | One-shot prompt                             |
| `-m, --max-new-tokens` | 64               | Generation budget                           |
| `-t, --temperature` | 0.0 (= greedy)      | Sampling temperature                        |
| `--top-k`         | none                  | Top-k filter                                |
| `--top-p`         | none                  | Nucleus-sampling cutoff                     |
| `--seed`          | 0 (= OS entropy)      | Sampling RNG seed                           |
| `--repl`          | off                   | Interactive prompt loop                     |

## Architecture

```
token_id
  │
  ▼ embedding (+ sinusoidal positional encoding)
  ▼ N × DecoderBlock(
  │       LayerNorm → MultiHeadCausalSelfAttention → +residual
  │       LayerNorm → FeedForward (Linear → GELU → Linear) → +residual
  │   )
  ▼ final LayerNorm
  ▼ LM head Linear(embedding_dim → vocab_size)
  ▼ softmax + cross-entropy
```

Special tokens: `<unk>`, `<stop>`, `<tool>`, `<bos>`. `<bos>` is prepended at
training and inference time; `<stop>` is appended at training time and
terminates generation.

## Tests

```pwsh
cargo test --workspace
```

Includes unit tests for every nn primitive, attention/transformer math,
save/load roundtrips (bincode + JSON), an Adam overfit test, and an
end-to-end pipeline test (`blip_ai/tests/end_to_end.rs`).

Generation uses an incremental per-layer KV cache so token generation is
significantly faster than re-running full-prefix attention each step.

## Checkpoint format

Model version is stamped in `blip_ai/src/version.rs`. Loading accepts current
version and compatible v0.4 checkpoints, then normalizes to the current
version in-memory before saving.
Path extension picks the format: `.json` is human-readable, anything else is
bincode (smaller and faster).

## Limitations / roadmap

- Word-level tokenizer (with contractions), still no BPE.
- Tied embeddings are not implemented yet.
- No dropout in attention weights (only on attn/FFN outputs).

These are the natural next things to add.
