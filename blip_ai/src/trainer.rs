use crate::model::Model;
use crate::tokenizer::tokenize;
use rand::prelude::SliceRandom;

#[derive(Clone)]
pub struct TrainingPrompt {
    prompt: String,
    tokens: Vec<usize>,
}

pub struct TrainingData {
    model: Model,
    data: Vec<TrainingPrompt>,
}

impl TrainingData {
    pub fn new(model: Model) -> Self {
        TrainingData {
            model,
            data: Vec::new(),
        }
    }

    pub fn add_prompt(&mut self, prompt: &str) {
        let tokens = tokenize(prompt, &mut self.model); // Assuming a default embedding dimension

        self.data.push(TrainingPrompt {
            prompt: prompt.to_string(),
            tokens,
        });
    }

    pub fn load(&mut self, file_name: &str) -> Result<(), String> {
        let lines = std::fs::read_to_string(file_name)
            .map_err(|e| e.to_string())?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#')) // Ignore empty lines and comments
            .collect::<Vec<String>>();

        for line in lines {
            let tokens = tokenize(&line, &mut self.model);

            self.data.push(TrainingPrompt {
                prompt: line,
                tokens,
            });
        }

        Ok(())
    }

    pub fn get_model(&self) -> &Model {
        &self.model
    }

    pub fn get_model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    pub fn train_multi_head_attention(&mut self, num_epochs: usize, learning_rate: f32) {
        // Zip all token ids in a single vector
        let token_stop = self.model.get_stop_token_id();
        let all_tokens: Vec<usize> = self
            .data
            .iter_mut()
            .map(|p| p.tokens.clone())
            .map(|mut p| {
                p.push(token_stop);
                p
            })
            .flat_map(|p| p)
            .collect();

        for epoch in 0..num_epochs {
            println!("Training epoch {}/{}", epoch + 1, num_epochs);
            self.model
                .train_multi_head_attention(&all_tokens, learning_rate);
        }
    }

    pub fn train_neural_network(&mut self, num_epochs: usize, learning_rate: f32, batch_size: usize) {
        let mut rng = rand::rng();

        for epoch in 0..num_epochs {
            println!("NN training epoch {}/{}", epoch + 1, num_epochs);
            let mut prompts = self.data.clone();
            prompts.shuffle(&mut rng);

            let mut batch = Vec::new();
            for prompt in &prompts {
                for window in prompt.tokens.windows(2) {
                    let input_id = window[0];
                    let target_id = window[1];
                    batch.push((input_id, target_id));
                    if batch.len() == batch_size {
                        for (input_id, target_id) in &batch {
                            self.model
                                .network_train_step(*input_id, *target_id, learning_rate);
                        }
                        batch.clear();
                    }
                }
            }

            // Traiter le dernier batch s'il n'est pas vide
            if !batch.is_empty() {
                for (input_id, target_id) in &batch {
                    self.model
                        .network_train_step(*input_id, *target_id, learning_rate);
                }
            }
        }

        // let mut rng = rand::rng();
        // for epoch in 0..num_epochs {
        //     println!("NN training epoch {}/{}", epoch + 1, num_epochs);
        //     let mut prompts = self.data.clone();
        //     prompts.shuffle(&mut rng);
        //     for prompt in &prompts {
        //         for window in prompt.tokens.windows(2) {
        //             let input_id = window[0];
        //             let target_id = window[1];
        //             self.model.network_train_step(input_id, target_id, learning_rate);
        //         }
        //     }
        // }

        // for epoch in 0..num_epochs {
        //     println!("NN training epoch {}/{}", epoch + 1, num_epochs);
        //     for prompt in &self.data {
        //         // On entraîne sur chaque paire de tokens consécutifs
        //         for window in prompt.tokens.windows(2) {
        //             let input_id = window[0];
        //             let target_id = window[1];
        //             self.model.network_train_step(input_id, target_id, learning_rate);
        //         }
        //     }
        // }
    }
}
