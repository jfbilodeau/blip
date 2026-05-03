use crate::model::Model;
use crate::tokenizer::tokenize_with_vocab;

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
    ///
    /// Tokenization is read-only against the model's frozen vocabulary, and
    /// any `<unk>` ids produced by tokenization are filtered out so the
    /// pretraining stream never contains them.
    pub fn load(&mut self, file_name: &str, target_seq_len: usize) -> Result<(), String> {
        let contents = std::fs::read_to_string(file_name).map_err(|e| e.to_string())?;
        self.load_from_str(&contents, target_seq_len);
        Ok(())
    }

    fn load_from_str(&mut self, contents: &str, target_seq_len: usize) {
        let unk = self.model.get_unknown_token_id();
        let mut current_text = String::new();
        let mut current_tokens: Vec<usize> = Vec::new();

        for raw_line in contents.lines() {
            let line = raw_line.trim();

            // Skip empty lines and comments.
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Tokenize this line read-only, then drop <unk> ids entirely.
            let line_tokens: Vec<usize> = tokenize_with_vocab(line, &self.model)
                .into_iter()
                .filter(|&id| id != unk)
                .collect();

            if line_tokens.is_empty() {
                continue;
            }

            // Check if adding this line would exceed target length.
            if !current_tokens.is_empty()
                && current_tokens.len() + line_tokens.len() > target_seq_len
            {
                // Flush current sequence and start a new one.
                self.data.push(PretrainingSequence {
                    text: std::mem::take(&mut current_text),
                    tokens: std::mem::take(&mut current_tokens),
                });
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

    pub fn sequences(&self) -> &[PretrainingSequence] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize_build_vocab;

    fn model_with_words(words: &[&str]) -> Model {
        let mut m = Model::new(8, 1, 2);
        for w in words {
            tokenize_build_vocab(w, &mut m);
        }
        m.initialize_embeddings();
        m
    }

    #[test]
    fn load_joins_lines_into_sequences() {
        let model = model_with_words(&["hello world this is a test foo bar"]);
        let mut pretraining = PretrainingData::new(model);
        let contents = "hello world\nthis is a test\nfoo bar\n";

        pretraining.load_from_str(contents, 10);

        assert!(pretraining.num_sequences() > 0);
    }

    #[test]
    fn load_skips_empty_lines_and_comments() {
        let model = model_with_words(&["hello world"]);
        let mut pretraining = PretrainingData::new(model);
        let contents = "# comment\nhello\n\nworld\n";

        pretraining.load_from_str(contents, 100);

        assert_eq!(pretraining.num_sequences(), 1);
    }

    #[test]
    fn pretraining_sequences_never_contain_unk() {
        // Vocab only knows "hello"; "world" should produce <unk> which is
        // then filtered out, leaving just the "hello" tokens.
        let model = model_with_words(&["hello"]);
        let unk = model.get_unknown_token_id();
        let mut pretraining = PretrainingData::new(model);
        pretraining.load_from_str("hello world\n", 100);
        for seq in pretraining.sequences() {
            assert!(
                seq.tokens.iter().all(|&id| id != unk),
                "pretraining sequence contains <unk>"
            );
        }
    }
}
