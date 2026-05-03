use crate::model::Model;
use crate::nn::Optimizer;
use crate::tokenizer::{tokenize_build_vocab_with_specials, tokenize_with_vocab_and_specials};
use log::info;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use std::io::{self, Write};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct TrainingPrompt {
    pub text: String,
    pub tokens: Vec<usize>,
}

pub struct TrainingData {
    model: Model,
    data: Vec<TrainingPrompt>,
}

#[derive(Clone, Copy, Debug)]
pub enum StopTokenMode {
    /// Append `<stop>` to each prompt. Used for chat tuning so the model
    /// learns to terminate replies.
    AppendStop,
    /// Use the raw token stream with no `<stop>` suffix. Used for
    /// pretraining over plain corpus so the `<stop>`, `<user>`, `<ai>`
    /// embedding rows aren't trained on negative-class signal from
    /// contexts they never appear in.
    Bare,
}

/// Learning-rate schedule applied per optimizer step.
#[derive(Clone, Copy, Debug)]
pub enum LrSchedule {
    /// Constant learning rate. `lr` from `TrainingConfig` is used as-is.
    Constant,
    /// Linear warmup for `warmup_steps` from 0 -> `lr`, then cosine decay
    /// down to `min_lr` over the remaining steps.
    CosineWithWarmup { warmup_steps: usize, min_lr: f32 },
}

impl LrSchedule {
    pub fn lr_at(&self, step: usize, total_steps: usize, base_lr: f32) -> f32 {
        match *self {
            LrSchedule::Constant => base_lr,
            LrSchedule::CosineWithWarmup { warmup_steps, min_lr } => {
                if step < warmup_steps {
                    base_lr * (step as f32 + 1.0) / (warmup_steps.max(1) as f32)
                } else if total_steps <= warmup_steps {
                    base_lr
                } else {
                    let progress = (step - warmup_steps) as f32
                        / (total_steps - warmup_steps).max(1) as f32;
                    let cos = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                    min_lr + (base_lr - min_lr) * cos
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub val_split: f32,
    pub seed: u64,
    /// Save a checkpoint every N epochs (0 = only at the end).
    pub checkpoint_every: usize,
    pub checkpoint_path: Option<String>,
    pub lr_schedule: LrSchedule,
    /// Dropout probability applied to attention and FFN outputs during training.
    pub dropout: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 100,
            learning_rate: 1e-3,
            batch_size: 1,
            val_split: 0.0,
            seed: 0,
            checkpoint_every: 0,
            checkpoint_path: None,
            lr_schedule: LrSchedule::Constant,
            dropout: 0.0,
        }
    }
}

impl TrainingData {
    pub fn new(model: Model) -> Self {
        TrainingData {
            model,
            data: Vec::new(),
        }
    }

    fn add_prompt_internal(&mut self, prompt: &str, register_new_tokens: bool) {
        let tokens = if register_new_tokens {
            tokenize_build_vocab_with_specials(prompt, &mut self.model)
        } else {
            tokenize_with_vocab_and_specials(prompt, &self.model)
        };
        self.data.push(TrainingPrompt {
            text: prompt.to_string(),
            tokens,
        });
    }

    pub fn add_prompt(&mut self, prompt: &str) {
        self.add_prompt_internal(prompt, true);
    }

    /// Add a prompt while keeping vocabulary fixed. Unknown tokens map to `<unk>`.
    pub fn add_prompt_with_existing_vocab(&mut self, prompt: &str) {
        self.add_prompt_internal(prompt, false);
    }

    /// Push a pre-tokenized prompt directly. The `text` field is used only
    /// for diagnostics and for `add_prompt_with_existing_vocab` re-tokenization
    /// of plain corpus prompts; tuning prompts that contain role tokens
    /// should set it to a human-readable rendition.
    pub fn add_prompt_tokens(&mut self, text: String, tokens: Vec<usize>) {
        self.data.push(TrainingPrompt { text, tokens });
    }

    /// Detect whether a line starts with a recognized chat role prefix and
    /// return `(role_token_id, message_text)` if so. The literal `role:`
    /// prefix is stripped — the role is represented by a special token id
    /// in the resulting sequence rather than as text.
    fn parse_role_line(&self, line: &str) -> Option<(usize, String)> {
        let trimmed = line.trim();
        let (role, rest) = trimmed.split_once(':')?;
        let role_id = if role.eq_ignore_ascii_case("user") {
            self.model.get_user_token_id()
        } else if role.eq_ignore_ascii_case("ai") || role.eq_ignore_ascii_case("assistant") {
            self.model.get_ai_token_id()
        } else {
            return None;
        };
        Some((role_id, rest.trim().to_string()))
    }

    fn load_conversations_from_str(&mut self, contents: &str, register_new_tokens: bool) {
        // A "conversation" is a contiguous run of role-tagged lines (user:/ai:),
        // turned into a single sequence of:
        //   [<user>, ...msg tokens, <ai>, ...msg tokens, <user>, ...]
        // The literal `user:` / `ai:` prefixes are NOT included in the token
        // stream — only the special role token ids are.
        //
        // A "corpus" line (no recognized role prefix) is added as its own
        // single-prompt sequence containing just the tokenized text.
        let mut current_conv_tokens: Vec<usize> = Vec::new();
        let mut current_conv_text: String = String::new();

        for raw_line in contents.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                if !current_conv_tokens.is_empty() {
                    self.add_prompt_tokens(
                        std::mem::take(&mut current_conv_text),
                        std::mem::take(&mut current_conv_tokens),
                    );
                }
                continue;
            }
            if line.starts_with('#') {
                continue;
            }

            if let Some((role_id, message)) = self.parse_role_line(line) {
                let msg_tokens = if register_new_tokens {
                    tokenize_build_vocab_with_specials(&message, &mut self.model)
                } else {
                    tokenize_with_vocab_and_specials(&message, &self.model)
                };
                current_conv_tokens.push(role_id);
                current_conv_tokens.extend(msg_tokens);

                if !current_conv_text.is_empty() {
                    current_conv_text.push(' ');
                }
                let role_text = self
                    .model
                    .get_token_by_id(role_id)
                    .unwrap_or("")
                    .to_string();
                current_conv_text.push_str(&role_text);
                current_conv_text.push(' ');
                current_conv_text.push_str(&message);
            } else {
                // Corpus line — flush any in-progress conversation first.
                if !current_conv_tokens.is_empty() {
                    self.add_prompt_tokens(
                        std::mem::take(&mut current_conv_text),
                        std::mem::take(&mut current_conv_tokens),
                    );
                }
                self.add_prompt_internal(line, register_new_tokens);
            }
        }

        if !current_conv_tokens.is_empty() {
            self.add_prompt_tokens(current_conv_text, current_conv_tokens);
        }
    }

    pub fn load(&mut self, file_name: &str) -> Result<(), String> {
        let contents = std::fs::read_to_string(file_name).map_err(|e| e.to_string())?;
        self.load_conversations_from_str(&contents, true);
        Ok(())
    }

    /// Load prompts without expanding vocabulary. Unknown tokens map to `<unk>`.
    pub fn load_with_existing_vocab(&mut self, file_name: &str) -> Result<(), String> {
        let contents = std::fs::read_to_string(file_name).map_err(|e| e.to_string())?;
        self.load_conversations_from_str(&contents, false);
        Ok(())
    }

    pub fn get_model(&self) -> &Model {
        &self.model
    }

    pub fn get_model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    pub fn num_prompts(&self) -> usize {
        self.data.len()
    }

    pub fn clear_prompts(&mut self) {
        self.data.clear();
    }

    /// Trim rare tokens from the vocabulary before training. Must be called
    /// before `initialize_embeddings`. Callers MUST reload prompts after
    /// trimming (typically via `clear_prompts` + `load_with_existing_vocab`)
    /// because already-loaded token streams will reference removed token ids.
    pub fn trim_vocab(&mut self, min_count: u32) -> usize {
        let removed = self.model.trim_vocab(min_count);
        if removed > 0 {
            self.data.clear();
        }
        removed
    }

    /// Train using next-token cross-entropy with Adam and global grad clipping.
    pub fn train(&mut self, cfg: &TrainingConfig) {
        self.train_with_stop_mode(cfg, StopTokenMode::AppendStop);
    }

    /// Train using next-token cross-entropy with Adam and global grad clipping.
    ///
    /// `StopTokenMode::AppendStop` appends `<stop>` to each prompt.
    /// `StopTokenMode::Bare` leaves prompts as-is (no wrapping).
    pub fn train_with_stop_mode(&mut self, cfg: &TrainingConfig, stop_mode: StopTokenMode) {
        let stop = self.model.get_stop_token_id();

        let mut sequences: Vec<Vec<usize>> = self
            .data
            .iter()
            .map(|p| {
                match stop_mode {
                    StopTokenMode::Bare => p.tokens.clone(),
                    StopTokenMode::AppendStop => {
                        let mut s = Vec::with_capacity(p.tokens.len() + 1);
                        s.extend_from_slice(&p.tokens);
                        s.push(stop);
                        s
                    }
                }
            })
            .filter(|s| s.len() >= 2)
            .collect();

        if sequences.is_empty() {
            eprintln!("No training sequences (need at least 2 tokens per prompt).");
            return;
        }

        // Deterministic shuffle of the dataset for the train/val split.
        let mut split_rng = ChaCha12Rng::seed_from_u64(cfg.seed);
        sequences.shuffle(&mut split_rng);

        let n_val = ((sequences.len() as f32) * cfg.val_split).floor() as usize;
        let val_seqs: Vec<Vec<usize>> = sequences.drain(..n_val).collect();
        let train_seqs = sequences;

        info!(
            "training on {} sequences, validating on {}",
            train_seqs.len(),
            val_seqs.len()
        );

        // Apply dropout to the model for the duration of training, then turn
        // it back off so generation/eval are deterministic w.r.t. the model.
        self.model.set_dropout(cfg.dropout);

        // Each `apply_grads` is one optimizer step. Total steps is bounded by
        // ceil(train_seqs.len() / batch_size) per epoch.
        let steps_per_epoch =
            (train_seqs.len() + cfg.batch_size.max(1) - 1) / cfg.batch_size.max(1);
        let total_steps = steps_per_epoch * cfg.num_epochs;
        let mut step: usize = 0;

        let mut epoch_rng = ChaCha12Rng::seed_from_u64(cfg.seed.wrapping_add(1));
        let mut indices: Vec<usize> = (0..train_seqs.len()).collect();
        let mut progress_line = ProgressLine::default();

        for epoch in 0..cfg.num_epochs {
            indices.shuffle(&mut epoch_rng);
            let mut total_loss = 0.0_f32;
            let mut count = 0usize;
            let epoch_start = Instant::now();
            let mut last_progress_update = epoch_start.checked_sub(Duration::from_secs(1)).unwrap_or(epoch_start);
            let batch_size = cfg.batch_size.max(1);

            for batch in indices.chunks(batch_size) {
                let template = &self.model;
                let (batch_grads, batch_loss, batch_count) = batch
                    .par_iter()
                    .fold(
                        || {
                            let mut local = template.clone();
                            local.zero_all_grads();
                            (local, 0.0_f32, 0usize)
                        },
                        |(mut local, loss_sum, count), &idx| {
                            let loss = local.train_sequence(&train_seqs[idx]);
                            (local, loss_sum + loss, count + 1)
                        },
                    )
                    .reduce(
                        || {
                            let mut local = template.clone();
                            local.zero_all_grads();
                            (local, 0.0_f32, 0usize)
                        },
                        |(mut lhs_model, lhs_loss, lhs_count), (rhs_model, rhs_loss, rhs_count)| {
                            lhs_model.add_grads_from(&rhs_model);
                            (lhs_model, lhs_loss + rhs_loss, lhs_count + rhs_count)
                        },
                    );

                self.model.zero_all_grads();
                self.model.add_grads_from(&batch_grads);

                total_loss += batch_loss;
                count += batch_count;

                let lr = cfg.lr_schedule.lr_at(step, total_steps, cfg.learning_rate);
                self.model.apply_grads(Optimizer::adam(lr));
                step += 1;

                let should_update_progress = count == train_seqs.len()
                    || last_progress_update.elapsed() >= Duration::from_millis(250);
                if should_update_progress {
                    let percent = (count as f32 / train_seqs.len().max(1) as f32) * 100.0;
                    let avg_loss = total_loss / count.max(1) as f32;
                    let current_lr = cfg.lr_schedule.lr_at(step, total_steps, cfg.learning_rate);
                    let elapsed = epoch_start.elapsed();
                    let elapsed_secs = elapsed.as_secs_f32();
                    let seqs_per_sec = (count as f32 / elapsed_secs.max(1e-6)).max(1e-6);
                    let remaining = train_seqs.len().saturating_sub(count);
                    let eta_secs = remaining as f32 / seqs_per_sec;
                    progress_line.print(&format!(
                        "Epoch {}/{} [{:>5.1}%] {}/{} seq - avg loss {:.4} - lr {:.6} - elapsed {:.1}s - eta {:.1}s",
                        epoch + 1,
                        cfg.num_epochs,
                        percent,
                        count,
                        train_seqs.len(),
                        avg_loss,
                        current_lr,
                        elapsed_secs,
                        eta_secs
                    ));
                    last_progress_update = Instant::now();
                }
            }
            let train_avg = total_loss / count.max(1) as f32;

            let val_avg = if val_seqs.is_empty() {
                f32::NAN
            } else {
                let mut s = 0.0_f32;
                let mut c = 0usize;
                for v in &val_seqs {
                    s += self.model.eval_sequence(v);
                    c += 1;
                }
                s / c.max(1) as f32
            };

            progress_line.clear();

            if val_avg.is_nan() {
                println!(
                    "Epoch {}/{} - train {:.4}",
                    epoch + 1,
                    cfg.num_epochs,
                    train_avg
                );
            } else {
                println!(
                    "Epoch {}/{} - train {:.4} - val {:.4}",
                    epoch + 1,
                    cfg.num_epochs,
                    train_avg,
                    val_avg
                );
            }

            if cfg.checkpoint_every > 0
                && (epoch + 1) % cfg.checkpoint_every == 0
                && cfg.checkpoint_path.is_some()
            {
                let path = cfg.checkpoint_path.as_ref().unwrap();
                if let Err(e) = self.model.save(path) {
                    eprintln!("Checkpoint save failed: {}", e);
                } else {
                    info!("checkpoint saved to {}", path);
                }
            }
        }

        // Restore inference mode (no-op for forward()/forward_logits, but
        // keeps `train_sequence` deterministic if called again).
        self.model.set_dropout(0.0);
    }
}

#[derive(Default)]
struct ProgressLine {
    last_len: usize,
}

impl ProgressLine {
    fn print(&mut self, message: &str) {
        let mut stdout = io::stdout();
        let padding = self.last_len.saturating_sub(message.len());
        let _ = write!(stdout, "\r{}{}", message, " ".repeat(padding));
        let _ = stdout.flush();
        self.last_len = message.len();
    }

    fn clear(&mut self) {
        if self.last_len == 0 {
            return;
        }

        let mut stdout = io::stdout();
        let _ = write!(stdout, "\r{}\r", " ".repeat(self.last_len));
        let _ = stdout.flush();
        self.last_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_groups_blank_line_separated_conversations() {
        let model = Model::new(8, 1, 2);
        let mut training = TrainingData::new(model);
        let contents = "# comment\nuser:Who are you?\nai:I am Blip.\n\nuser:What can you do?\nassistant:I can help.\n";

        training.load_conversations_from_str(contents, true);

        assert_eq!(training.num_prompts(), 2);
        let user_id = training.model.get_user_token_id();
        let ai_id = training.model.get_ai_token_id();
        // Each conversation must contain both <user> and <ai> token ids and
        // must NOT contain the literal "user" / "ai" / ":" tokens used as
        // role prefixes.
        for prompt in &training.data {
            assert!(prompt.tokens.contains(&user_id));
            assert!(prompt.tokens.contains(&ai_id));
        }
    }

    #[test]
    fn load_keeps_legacy_single_line_records() {
        let model = Model::new(8, 1, 2);
        let mut training = TrainingData::new(model);
        let contents = "hello world\nplain text\n";

        training.load_conversations_from_str(contents, true);

        assert_eq!(training.num_prompts(), 2);
        assert_eq!(training.data[0].text, "hello world");
        assert_eq!(training.data[1].text, "plain text");
    }

    #[test]
    fn role_lines_emit_role_tokens_not_literal_text() {
        let model = Model::new(8, 1, 2);
        let mut training = TrainingData::new(model);
        training.load_conversations_from_str("user:hi\nai:hello\n", true);
        assert_eq!(training.num_prompts(), 1);

        let user_id = training.model.get_user_token_id();
        let ai_id = training.model.get_ai_token_id();
        let toks = &training.data[0].tokens;

        // First token is <user>, then "hi" tokens, then <ai>, then "hello" tokens.
        assert_eq!(toks.first().copied(), Some(user_id));
        assert!(toks.contains(&ai_id));

        // Vocabulary must not have grown to include the literal role words
        // as standalone uppercase tokens — split_tokens lowercases, so just
        // verify the colon character isn't being injected at conversation
        // boundaries between role and message.
        let colon_id = training.model.get_token_id(":");
        if let Some(colon_id) = colon_id {
            // ':' may exist as a default symbol token, but it must not appear
            // immediately after the <user> token (which would mean we left
            // the role prefix literal in place).
            for window in toks.windows(2) {
                assert!(
                    !(window[0] == user_id && window[1] == colon_id),
                    "role prefix ':' leaked into token stream"
                );
                assert!(
                    !(window[0] == ai_id && window[1] == colon_id),
                    "role prefix ':' leaked into token stream"
                );
            }
        }
    }
}
