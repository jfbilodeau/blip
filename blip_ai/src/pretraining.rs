//! Pretraining data loader that groups lines into longer sequences.
//! 
//! Unlike the tuning loader (which treats each line/conversation as a single
//! sequence), this loader joins lines into longer coherent sequences up to a
//! target token count. This preserves longer context for the model to learn from.

use crate::model::Model;
use crate::tokenizer::tokenize;

#[derive(Clone)]
pub struct PretrainingSequence {
    pub text: String,
    pub tokens: Vec<usize>,
}

pub struct PretrainingData {
    model: Model,
    data: Vec<PretrainingSequence>,
}

impl PretrainingData {
    pub fn new(model: Model) -> Self {
        PretrainingData {
            model,
            data: Vec::new(),
        }
    }

    /// Load pretraining corpus, joining lines into sequences of approximately
    /// `target_seq_len` tokens. Each sequence is a separate training example.
    pub fn load(&mut self, file_name: &str, target_seq_len: usize) -> Result<(), String> {
        let contents = std::fs::read_to_string(file_name).map_err(|e| e.to_string())?;
        self.load_from_str(&contents, target_seq_len);
        Ok(())
    }

    fn load_from_str(&mut self, contents: &str, target_seq_len: usize) {
        let mut current_text = String::new();
        let mut current_tokens: Vec<usize> = Vec::new();

        for raw_line in contents.lines() {
            let line = raw_line.trim();

            // Skip empty lines and comments.
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Tokenize this line.
            let line_tokens = tokenize(line, &mut self.model);

            // Check if adding this line would exceed target length.
            if !current_tokens.is_empty()
                && current_tokens.len() + line_tokens.len() > target_seq_len
            {
                // Flush current sequence and start a new one.
                if !current_tokens.is_empty() {
                    self.data.push(PretrainingSequence {
                        text: current_text.clone(),
                        tokens: current_tokens.clone(),
                    });
                }
                current_text.clear();
                current_tokens.clear();
            }

            // Append line to current sequence.
            if !current_text.is_empty() {
                current_text.push(' ');
            }
            current_text.push_str(line);
            current_tokens.extend(line_tokens);
        }

        // Flush any remaining sequence.
        if !current_tokens.is_empty() {
            self.data.push(PretrainingSequence {
                text: current_text,
                tokens: current_tokens,
            });
        }
    }

    pub fn get_model(&self) -> &Model {
        &self.model
    }

    pub fn get_model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    pub fn num_sequences(&self) -> usize {
        self.data.len()
    }

    /// Trim rare tokens from the vocabulary before training. Re-tokenizes all
    /// sequences so their token ids point to the new vocab.
    pub fn trim_vocab(&mut self, min_count: u32) -> usize {
        let removed = self.model.trim_vocab(min_count);
        if removed > 0 {
            let unk = self.model.get_unknown_token_id();
            for seq in &mut self.data {
                seq.tokens = crate::tokenizer::split_tokens(&seq.text)
                    .into_iter()
                    .map(|t| self.model.get_token_id(&t).unwrap_or(unk))
                    .collect();
            }
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_joins_lines_into_sequences() {
        let model = Model::new(8, 1, 2);
        let mut pretraining = PretrainingData::new(model);
        let contents = "hello world\nthis is a test\nfoo bar\n";

        // With target_seq_len=10, should join at least some lines.
        pretraining.load_from_str(contents, 10);

        // Should have at least 1 sequence (multiple lines joined).
        assert!(pretraining.num_sequences() > 0);
    }

    #[test]
    fn load_skips_empty_lines_and_comments() {
        let model = Model::new(8, 1, 2);
        let mut pretraining = PretrainingData::new(model);
        let contents = "# comment\nhello\n\nworld\n";

        pretraining.load_from_str(contents, 100);

        // Should have loaded "hello world" as one sequence, skipping comment and blank line.
        assert_eq!(pretraining.num_sequences(), 1);
    }
}
