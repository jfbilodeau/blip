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
  -o models/basic.json
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
| `--pretrain-epochs` | 4                   | Pretraining epochs over corpus data         |
| `--pretrain-lr`   | 0.0005                | Pretraining Adam learning rate              |
| `--pretrain-batch-size` | 128            | Pretraining sequences per optimizer step    |
| `--pretrain-warmup` | 200                 | Pretraining warmup steps (`0` = constant LR) |
| `--pretrain-min-lr` | 0.00001             | Pretraining cosine-decay LR floor           |
| `-n, --epochs`    | 60                    | Chat-tuning epochs                           |
| `-l, --learning-rate` | 0.001             | Chat-tuning Adam learning rate              |
| `-b, --batch-size` | 128                  | Chat-tuning sequences per optimizer step    |
| `--dropout`       | 0.10                  | Training-time dropout on attention/FFN outputs |
| `--val-split`     | 0.10                  | Chat-tuning validation split fraction       |
| `--warmup-steps`  | 50                    | Chat-tuning warmup steps (`0` = constant LR) |
| `--min-lr`        | 0.0001                | Chat-tuning cosine-decay LR floor           |
| `--seed`          | 42                    | Random seed (`0` = OS entropy)              |
| `--checkpoint-every` | 10                | Save checkpoint every N epochs (`0` = end only) |
| `--min-count`     | 1                     | Drop tokens below this usage_count (specials always kept) |
| `--seq-length`    | 256                   | Target sequence length for pretraining corpus loader |
| `-p, --pretraining-files` | `training/pretraining/*`, `training/pretraining/books/*` | Pretraining corpus file globs, trained without `<stop>` |
| `-t, --tuning-files`  | `training/tuning/*` | Chat-tuning file globs, trained with `<stop>` |
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

Special tokens: `<unk>`, `<stop>`, `<tool>`, `<user>`, `<ai>`. Tokens are
always lowercase for simplicity to reduce vocabulary size. The trainer runs
in two phases by default: pretraining files are trained as bare token
streams (no role/control wrapping), then chat-tuning files are trained with
`<stop>` appended to each sequence. During inference, generation stops when
`<stop>` is sampled.

Inference prompt framing uses role tokens: the runtime builds
`[<user>, ...prompt_tokens, <ai>]` before generation. A leading literal
`user:` prefix typed in REPL/CLI input is stripped before tokenization.

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

When generated ids detokenize to only whitespace/control output, the inference
CLI prints a diagnostic (`<no output>` or `<blank output; generated ...>`) so
empty-looking generations are easier to debug.

## Limitations / roadmap

- Word-level tokenizer (with contractions), still no BPE.
- No dropout in attention weights (only on attn/FFN outputs).

These are the natural next things to add.
