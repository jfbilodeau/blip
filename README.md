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

Train with the default corpus globs (`training/pretraining/*` and `training/tuning/*`):

```pwsh
cargo run --release -p blip_trainer -- `
  -o models/basic.bin
```

Generate a one-shot response:

```pwsh
cargo run --release -p blip -- --repl false -p "Who are you?" -m 32 -t 0.8 --seed 1
```

REPL:

```pwsh
cargo run --release -p blip -- --repl -t 0.8 --seed 1
```

## Trainer flags

| Flag              | Default               | Meaning                                     |
| ----------------- | --------------------- | ------------------------------------------- |
| `-e, --embedding-dim` | 128               | Embedding / model dimension                 |
| `-d, --depth`     | 4                     | Number of decoder blocks                    |
| `--n-heads`       | 4                     | Attention heads (must divide `embedding_dim`) |
| `-n, --epochs`    | 60                    | Training epochs                             |
| `-l, --learning-rate` | 0.001             | Adam learning rate                          |
| `-b, --batch-size` | 16                   | Sequences per optimizer step                |
| `--dropout`       | 0.10                  | Training-time dropout on attention/FFN outputs |
| `--val-split`     | 0.10                  | Validation split fraction                   |
| `--warmup-steps`  | 200                   | Warmup steps for cosine LR (`0` = constant LR) |
| `--min-lr`        | 0.0001                | Final LR floor for cosine decay             |
| `--min-count`     | 3                     | Drop tokens below this usage_count (specials always kept) |
| `--seed`          | 42                    | RNG seed for init / shuffling               |
| `--checkpoint-every` | 10                | Save every N epochs                         |
| `-p, --pretraining-files` | `training/pretraining/*` | Pretraining corpus file(s), trained without `<stop>` |
| `-t, --tuning-files`  | `training/tuning/*` | Chat-tuning file(s), trained with `<stop>` |
| `-o, --output-file`  | `models/basic.json` | Output checkpoint (`.json` → JSON, else bincode) |

## Inference flags

| Flag              | Default               | Meaning                                     |
| ----------------- | --------------------- | ------------------------------------------- |
| `-f, --model-file` | `models/basic.json`   | Checkpoint to load                          |
| `-p, --prompt`    | none                  | One-shot prompt; if `--repl false`, falls back to `Who are you?` |
| `-m, --max-new-tokens` | 64               | Generation budget                           |
| `-t, --temperature` | 0.0 (= greedy)      | Sampling temperature                        |
| `--top-k`         | none                  | Top-k filter                                |
| `--top-p`         | none                  | Nucleus-sampling cutoff                     |
| `--seed`          | 0 (= OS entropy)      | Sampling RNG seed                           |
| `--repl`          | true                  | Interactive prompt loop                     |

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
training and inference time. Tokens are always lowercase for simplicity to
reduce vocabulary size. The trainer runs in two phases by default:
pretraining files are trained without appending `<stop>`, then chat-tuning
files are trained with `<stop>` appended to each sequence. During inference,
generation stops when `<stop>` is sampled.

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
bincode (smaller and faster). The CLI defaults to `models/basic.json` and will
fallback to `models/basic.bin` if the JSON path is missing.

## Limitations / roadmap

- Word-level tokenizer (with contractions), still no BPE.
- Tied embeddings are not implemented yet.
- No dropout in attention weights (only on attn/FFN outputs).

These are the natural next things to add.
