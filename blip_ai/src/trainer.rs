use crate::model::Model;
use crate::nn::Optimizer;
use crate::tokenizer::tokenize;
use log::info;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;

#[derive(Clone)]
pub struct TrainingPrompt {
    pub text: String,
    pub tokens: Vec<usize>,
}

pub struct TrainingData {
    model: Model,
    data: Vec<TrainingPrompt>,
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

    pub fn add_prompt(&mut self, prompt: &str) {
        let tokens = tokenize(prompt, &mut self.model);
        self.data.push(TrainingPrompt {
            text: prompt.to_string(),
            tokens,
        });
    }

    fn normalize_training_line(line: &str) -> (String, bool) {
        let trimmed = line.trim();
        let Some((role, rest)) = trimmed.split_once(':') else {
            return (trimmed.to_string(), false);
        };

        let normalized_role = if role.eq_ignore_ascii_case("assistant") {
            "ai"
        } else if role.eq_ignore_ascii_case("user") || role.eq_ignore_ascii_case("ai") {
            role
        } else {
            return (trimmed.to_string(), false);
        };

        (format!("{}:{}", normalized_role, rest.trim()), true)
    }

    fn load_conversations_from_str(&mut self, contents: &str) {
        let mut current_conversation: Vec<String> = Vec::new();
        let mut current_corpus: Vec<String> = Vec::new();

        let flush_conversation = |training_data: &mut TrainingData, conversation: &mut Vec<String>| {
            if conversation.is_empty() {
                return;
            }

            training_data.add_prompt(&conversation.join(" "));
            conversation.clear();
        };

        let flush_corpus = |training_data: &mut TrainingData, corpus: &mut Vec<String>| {
            if corpus.is_empty() {
                return;
            }

            training_data.add_prompt(&corpus.join(" "));
            corpus.clear();
        };

        for raw_line in contents.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                flush_conversation(self, &mut current_conversation);
                flush_corpus(self, &mut current_corpus);
                continue;
            }
            if line.starts_with('#') {
                continue;
            }

            let (normalized, is_role_line) = Self::normalize_training_line(line);
            if is_role_line {
                flush_corpus(self, &mut current_corpus);
                current_conversation.push(normalized);
            } else {
                flush_conversation(self, &mut current_conversation);
                current_corpus.push(normalized);
                flush_corpus(self, &mut current_corpus);
            }
        }

        flush_conversation(self, &mut current_conversation);
        flush_corpus(self, &mut current_corpus);
    }

    pub fn load(&mut self, file_name: &str) -> Result<(), String> {
        let contents = std::fs::read_to_string(file_name).map_err(|e| e.to_string())?;
        self.load_conversations_from_str(&contents);
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

    /// Trim rare tokens from the vocabulary before training. Re-tokenizes any
    /// already-loaded prompts so their ids point to the new vocab (rare tokens
    /// collapse to `<unk>`). Must be called before `initialize_embeddings`.
    pub fn trim_vocab(&mut self, min_count: u32) -> usize {
        let removed = self.model.trim_vocab(min_count);
        if removed > 0 {
            let unk = self.model.get_unknown_token_id();
            for prompt in &mut self.data {
                prompt.tokens = crate::tokenizer::split_tokens(&prompt.text)
                    .into_iter()
                    .map(|t| self.model.get_token_id(&t).unwrap_or(unk))
                    .collect();
            }
        }
        removed
    }

    /// Train using next-token cross-entropy with Adam and global grad clipping.
    pub fn train(&mut self, cfg: &TrainingConfig) {
        let bos = self.model.get_begin_token_id();
        let stop = self.model.get_stop_token_id();

        let mut sequences: Vec<Vec<usize>> = self
            .data
            .iter()
            .map(|p| {
                let mut s = Vec::with_capacity(p.tokens.len() + 2);
                s.push(bos);
                s.extend_from_slice(&p.tokens);
                s.push(stop);
                s
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

        for epoch in 0..cfg.num_epochs {
            indices.shuffle(&mut epoch_rng);
            let mut total_loss = 0.0_f32;
            let mut count = 0usize;
            let mut batch_in_progress = 0usize;

            for &idx in &indices {
                let loss = self.model.train_sequence(&train_seqs[idx]);
                total_loss += loss;
                count += 1;
                batch_in_progress += 1;
                if batch_in_progress >= cfg.batch_size {
                    let lr = cfg.lr_schedule.lr_at(step, total_steps, cfg.learning_rate);
                    self.model.apply_grads(Optimizer::adam(lr));
                    batch_in_progress = 0;
                    step += 1;
                }
            }
            if batch_in_progress > 0 {
                let lr = cfg.lr_schedule.lr_at(step, total_steps, cfg.learning_rate);
                self.model.apply_grads(Optimizer::adam(lr));
                step += 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_groups_blank_line_separated_conversations() {
        let model = Model::new(8, 1, 2);
        let mut training = TrainingData::new(model);
        let contents = "# comment\nuser:Who are you?\nai:I am Blip.\n\nuser:What can you do?\nassistant:I can help.\n";

        training.load_conversations_from_str(contents);

        assert_eq!(training.num_prompts(), 2);
        assert_eq!(training.data[0].text, "user:Who are you? ai:I am Blip.");
        assert_eq!(training.data[1].text, "user:What can you do? ai:I can help.");
    }

    #[test]
    fn load_keeps_legacy_single_line_records() {
        let model = Model::new(8, 1, 2);
        let mut training = TrainingData::new(model);
        let contents = "hello world\nplain text\n";

        training.load_conversations_from_str(contents);

        assert_eq!(training.num_prompts(), 2);
        assert_eq!(training.data[0].text, "hello world");
        assert_eq!(training.data[1].text, "plain text");
    }
}
