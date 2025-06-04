use crate::version::MODEL_VERSION;
use rand::Rng;
use rand::distr::Uniform;
use serde::{Deserialize, Serialize};
use crate::nn::NeuralNetwork;

#[derive(Serialize, Deserialize)]
struct Token {
    text: String,
    usage_count: u32,
    embedding: Vec<f32>,
    keys: Vec<f32>,
    queries: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct Model {
    version: String,
    embedding_dim: usize,
    tokens: Vec<Token>,
    feedforward_layers: Vec<Vec<f32>>,
    network: NeuralNetwork,
    weights1: Vec<Vec<f32>>,
    bias1: Vec<f32>,
    weights2: Vec<Vec<f32>>,
    bias2: Vec<f32>,
}

// Special token IDs
const TOKEN_UNKNOWN: &str = "<unk>";
const TOKEN_STOP: &str = "<stop>";
const TOKEN_TOOL: &str = "<tool>";

// Compute attention score between query and key vectors
fn attention_score(q: &[f32], k: &[f32]) -> f32 {
    let dot_product: f32 = q.iter().zip(k.iter()).map(|(x, y)| x * y).sum();
    dot_product / (q.len() as f32).sqrt() // Scaling factor
}

// Update Q/K/V vectors using gradient descent
fn update_qkv_vectors(qkv: &[f32], gradient: &[f32], learning_rate: f32) -> Vec<f32> {
    let mut result = qkv.to_vec();

    for (i, &grad) in gradient.iter().enumerate() {
        if i < result.len() {
            result[i] -= learning_rate * grad; // Adjust values based on gradient
        }
    }

    result
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let exp_scores: Vec<f32> = scores.iter().map(|s| s.exp()).collect();
    let sum_exp: f32 = exp_scores.iter().sum();

    exp_scores.iter().map(|s| s / sum_exp).collect()
}

pub fn positional_encoding(seq_len: usize, embedding_dim: usize) -> Vec<Vec<f32>> {
    let mut encoding = vec![vec![0.0; embedding_dim]; seq_len];

    for pos in 0..seq_len {
        for i in (0..embedding_dim).step_by(2) {
            let angle = pos as f32 / f32::powf(10000.0, (i as f32 / embedding_dim as f32));

            encoding[pos][i] = angle.sin(); // Sine for even indices
            if i + 1 < embedding_dim {
                encoding[pos][i + 1] = angle.cos(); // Cosine for odd indices
            }
        }
    }

    encoding
}

fn feedforward_layer(
    input: &[f32],
    weights1: &[Vec<f32>],
    bias1: &[f32],
    weights2: &[Vec<f32>],
    bias2: &[f32],
) -> Vec<f32> {
    let mut hidden_layer: Vec<f32> = weights1
        .iter()
        .map(|weight_vector| {
            weight_vector
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>()
        })
        .zip(bias1.iter())
        .map(|(sum, b)| sum + b)
        .collect();

    // Apply activation function (ReLU)
    hidden_layer.iter_mut().for_each(|x| *x = x.max(0.0)); // ReLU activation

    // Apply second linear transformation
    let output_layer: Vec<f32> = hidden_layer
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            weights2[i]
                .iter()
                .zip(bias2.iter())
                .map(|(w, b)| x * w + b)
                .sum()
        })
        .collect();

    output_layer
}

impl Model {
    pub fn new(embedding_dim: usize, depth: usize) -> Self {
        let mut model = Model {
            version: MODEL_VERSION.to_string(),
            embedding_dim,
            tokens: Vec::new(),
            feedforward_layers: Vec::new(),
            network: NeuralNetwork::new(&vec![embedding_dim; depth]),
            weights1: vec![vec![0.0; embedding_dim]; embedding_dim],
            bias1: vec![0.0; embedding_dim],
            weights2: vec![vec![0.0; embedding_dim]; embedding_dim],
            bias2: vec![0.0; embedding_dim],
        };

        // Register special tokens
        model.register_token(TOKEN_UNKNOWN);
        model.register_token(TOKEN_STOP);
        model.register_token(TOKEN_TOOL);

        model
    }

    pub fn load(file_name: &str) -> Result<Self, String> {
        let file = std::fs::File::open(file_name).map_err(|e| e.to_string())?;
        let model: Model = serde_json::from_reader(file).map_err(|e| e.to_string())?;

        if model.version != MODEL_VERSION {
            println!(
                "Model version mismatch: expected {}, got {}",
                MODEL_VERSION, model.version
            );
        }

        Ok(model)
    }

    pub fn save(&self, file_name: &str) -> Result<(), String> {
        let file = std::fs::File::create(file_name).map_err(|e| e.to_string())?;
        serde_json::to_writer(file, self).map_err(|e| e.to_string())?;

        Ok(())
    }

    pub fn get_token_id(&self, token: &str) -> Option<usize> {
        self.tokens.iter().position(|e| e.text == token.to_string())
    }

    pub fn get_token_by_id(&self, id: usize) -> Option<&str> {
        self.tokens.get(id).map(|token| token.text.as_str())
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

    pub fn register_token(&mut self, id: &str) -> usize {
        if let Some(index) = self.get_token_id(&id) {
            self.tokens[index].usage_count += 1;

            return index;
        }

        let new_index = self.tokens.len();

        self.tokens.push(Token {
            text: id.to_string(),
            usage_count: 1,
            embedding: vec![0.0; self.embedding_dim],
            keys: vec![0.0; self.embedding_dim],
            queries: vec![0.0; self.embedding_dim],
            values: vec![0.0; self.embedding_dim],
        });

        new_index
    }

    pub fn initialize_embeddings(&mut self) {
        let token_dim = self.tokens.len();
        let token_dim_f = token_dim as f32;
        let mut rng = rand::rng();

        let xavier = Uniform::new(-1.0 / token_dim_f.sqrt(), 1.0 / token_dim_f.sqrt())
            .expect("Failed to create Xavier distribution");

        let pos_encodings = positional_encoding(token_dim, self.embedding_dim);

        for (idx, token) in self.tokens.iter_mut().enumerate() {
            token.embedding = (0..self.embedding_dim)
                .map(|_| rng.sample(&xavier))
                .collect();

            token
                .embedding
                .iter_mut()
                .zip(pos_encodings[idx].iter())
                .for_each(|(emb, pos_enc)| *emb += pos_enc); // Add positional encoding

            // Initialize queries, keys, values with slight variations
            token.queries = token.embedding.iter().map(|&v| v * 0.8).collect();
            token.keys = token.embedding.iter().map(|&v| v * 0.9).collect();
            token.values = token.embedding.iter().map(|&v| v * 1.1).collect();
        }

        // Initialize feedforward layers
        self.weights1 = (0..self.embedding_dim)
            .map(|_| {
                (0..self.embedding_dim)
                    .map(|_| rng.sample(&xavier))
                    .collect()
            })
            .collect();

        self.bias1 = vec![0.0; self.embedding_dim];

        self.weights2 = (0..self.embedding_dim)
            .map(|_| {
                (0..self.embedding_dim)
                    .map(|_| rng.sample(&xavier))
                    .collect()
            })
            .collect();

        self.bias2 = vec![0.0; self.embedding_dim];
    }

    pub fn train_multi_head_attention(&mut self, token_ids: &[usize], learning_rate: f32) {
        let mut gradients = vec![vec![0.0; self.embedding_dim]; token_ids.len()];

        for (i, &id) in token_ids.iter().enumerate() {
            let q = &self.tokens[id].queries;
            // let k = &self.tokens[id].keys;
            let v = &self.tokens[id].values;

            // Compute attention scores
            let scores: Vec<f32> = token_ids
                .iter()
                .map(|&other_id| attention_score(q, &self.tokens[other_id].keys))
                .collect();

            // Apply softmax
            let weights = softmax(&scores);

            // Compute weighted sum of values
            let mut weighted_sum = vec![0.0; v.len()];
            for (j, &other_id) in token_ids.iter().enumerate() {
                let other_v = &self.tokens[other_id].values;

                for (sum, &val) in weighted_sum.iter_mut().zip(other_v.iter()) {
                    *sum += weights[j] * val;
                }
            }

            for (grad, (&predicted, &actual)) in gradients[i]
                .iter_mut()
                .zip(weighted_sum.iter().zip(v.iter()))
            {
                *grad = (predicted - actual).powi(2) / token_ids.len() as f32; // Scale by token count
            }
        }

        for &id in token_ids {
            let weights1 = &self.weights1;
            let bias1 = &self.bias1;
            let weights2 = &self.weights2;
            let bias2 = &self.bias2;

            let values = &self.tokens[id].values;

            let transformed_values = feedforward_layer(values, weights1, bias1, weights2, bias2);
            self.tokens[id].values =
                update_qkv_vectors(&values, &transformed_values, learning_rate);

            let queries = &self.tokens[id].queries;
            let transformed_queries = feedforward_layer(queries, weights1, bias1, weights2, bias2);
            self.tokens[id].queries =
                update_qkv_vectors(&queries, &transformed_queries, learning_rate);

            let keys = &self.tokens[id].keys;
            let transformed_keys = feedforward_layer(keys, weights1, bias1, weights2, bias2);
            self.tokens[id].keys = update_qkv_vectors(&keys, &transformed_keys, learning_rate);
        }
    }
    // Train step: Computes loss gradient and updates model weights
    pub fn train_step(
        &mut self,
        token_ids: &[usize],
        target_values: &[Vec<f32>],
        learning_rate: f32,
    ) {
        for (&id, target) in token_ids.iter().zip(target_values.iter()) {
            let values = &self.tokens[id].values;
            let values = values.to_vec();
            let predicted_values = feedforward_layer(
                &values,
                &self.weights1,
                &self.bias1,
                &self.weights2,
                &self.bias2,
            );

            let loss_gradient: Vec<f32> = predicted_values
                .iter()
                .zip(target.iter())
                .map(|(pred, actual)| 2.0 * (pred - actual)) // Derivative of squared error loss
                .collect();

            self.backpropagate(&values, &loss_gradient, learning_rate);
        }
    }

    // Backpropagation: Updates weights using loss gradient
    fn backpropagate(&mut self, input: &[f32], loss_gradient: &[f32], learning_rate: f32) {
        // Adjust weights2 and bias2
        for (i, grad) in loss_gradient.iter().enumerate() {
            for (j, w) in self.weights2[i].iter_mut().enumerate() {
                *w -= learning_rate * grad * input[j];
            }
            self.bias2[i] -= learning_rate * grad;
        }

        // Compute gradients for weights1
        let mut hidden_layer_gradient = vec![0.0; self.bias1.len()];
        for (i, grad) in loss_gradient.iter().enumerate() {
            for (j, w) in self.weights2[i].iter().enumerate() {
                hidden_layer_gradient[j] += grad * w;
            }
        }

        // Update weights1 and bias1
        for (i, h_grad) in hidden_layer_gradient.iter().enumerate() {
            for (j, w) in self.weights1[i].iter_mut().enumerate() {
                *w -= learning_rate * h_grad * input[j];
            }
            self.bias1[i] -= learning_rate * h_grad;
        }
    }

    pub fn network_forward(&self, input: Vec<f32>) -> Vec<f32> {
        self.network.forward(input)
    }

    pub fn network_train_step(&mut self, input_id: usize, target_id: usize, learning_rate: f32) {
        let input = &self.tokens[input_id].embedding;
        let target = &self.tokens[target_id].embedding;

        self.network.train_step(input, target, learning_rate);
    }

    pub fn complete(&self, tokens: &[usize]) -> Result<String, String> {
        let mut context_embeddings: Vec<f32> = vec![0.0; self.embedding_dim];

        for token_id in tokens {
            let embedding = &self.tokens[*token_id].embedding;
            for (i, &val) in embedding.iter().enumerate() {
                context_embeddings[i] += val;
            }
        }

        let mut output_sequence: Vec<usize> = Vec::new();

        loop {
            // 1. Calcul de la query du contexte
            let mut context_query = vec![0.0; self.embedding_dim];
            let mut count = 0;
            for token_id in tokens.iter().chain(output_sequence.iter()) {
                let query = &self.tokens[*token_id].queries;
                for (i, &v) in query.iter().enumerate() {
                    context_query[i] += v;
                }
                count += 1;
            }
            if count > 0 {
                for v in &mut context_query {
                    *v /= count as f32;
                }
            }

            // 2. Attention
            let scores: Vec<f32> = self.tokens
                .iter()
                .map(|token| attention_score(&context_query, &token.keys))
                .collect();

            let weights = softmax(&scores);

            let mut attended = vec![0.0; self.embedding_dim];
            for (token, &w) in self.tokens.iter().zip(weights.iter()) {
                for (i, &v) in token.values.iter().enumerate() {
                    attended[i] += w * v;
                }
            }

            // 3. Réseau de neurones
            let output_embedding = self.network.forward(attended);

            // 4. Trouver le token le plus proche (on récupère l'index)
            let mut best_match = None;
            let mut best_score = f32::INFINITY;
            for (idx, token) in self.tokens.iter().enumerate() {
                let distance = output_embedding.iter()
                    .zip(token.embedding.iter())
                    .map(|(o, e)| (*o - *e).powi(2))
                    .sum::<f32>();
                if distance < best_score {
                    best_score = distance;
                    best_match = Some(idx);
                }
            }

            // 5. Arrêter si <STOP>
            match best_match {
                Some(token_idx) if token_idx == self.get_stop_token_id() => break,
                Some(token_idx) => output_sequence.push(token_idx),
                None => return Err("No suitable token found".to_string()),
            }
        }

        // Convertir les indices en texte
        let result: Vec<&str> = output_sequence
            .iter()
            .filter_map(|&idx| self.get_token_by_id(idx))
            .collect();

        Ok(result.join(" "))
    }

    // pub fn complete(&self, tokens: &[usize]) -> Result<String, String> {
    //     let mut context_embeddings: Vec<f32> = vec![0.0; self.embedding_dim];
    //
    //     for token_id in tokens {
    //         let embedding = &self.tokens[*token_id].embedding;
    //         for (i, &val) in embedding.iter().enumerate() {
    //             context_embeddings[i] += val;
    //         }
    //     }
    //
    //     let mut output_sequence = Vec::new();
    //
    //     loop {
    //         // Utilise le réseau de neurones pour générer l'embedding du prochain token
    //         let output_embedding = self.network.forward(context_embeddings.clone());
    //
    //         // Trouve le token le plus proche de l'embedding généré
    //         let mut best_match = None;
    //         let mut best_score = f32::INFINITY;
    //
    //         for token in &self.tokens {
    //             let distance = output_embedding.iter()
    //                 .zip(token.embedding.iter())
    //                 .map(|(o, e)| (*o - *e).powi(2))
    //                 .sum::<f32>();
    //
    //             if distance < best_score {
    //                 best_score = distance;
    //                 best_match = Some(token.id.clone());
    //             }
    //         }
    //
    //         // Arrête si <STOP>
    //         match best_match {
    //             Some(ref token) if token == "<STOP>" => break,
    //             Some(token) => output_sequence.push(token),
    //             None => return Err("No suitable token found".to_string()),
    //         }
    //
    //         // Met à jour le contexte avec le nouvel embedding
    //         if let Some(idx) = self.get_token_id(output_sequence.last().unwrap()) {
    //             for (i, &val) in self.tokens[idx].embedding.iter().enumerate() {
    //                 context_embeddings[i] += val;
    //             }
    //         }
    //     }
    //
    //     Ok(output_sequence.join(" "))
    // }

    // pub fn complete(&self, tokens: &[usize]) -> Result<String, String> {
    //     let mut context_embeddings: Vec<f32> = vec![0.0; self.embedding_dim];
    //
    //     for token_id in tokens {
    //         let embedding = &self.tokens[*token_id].embedding;
    //         for (i, &val) in embedding.iter().enumerate() {
    //             context_embeddings[i] += val;
    //         }
    //     }
    //
    //     // Apply feedforward network
    //     let output = feedforward_layer(
    //         &context_embeddings,
    //         &self.weights1,
    //         &self.bias1,
    //         &self.weights2,
    //         &self.bias2,
    //     );
    //
    //     let mut output_sequence = Vec::new();
    //
    //     loop {
    //         // Pass embeddings through the feedforward network to generate next token
    //         let output_embedding = feedforward_layer(
    //             &context_embeddings,
    //             &self.weights1,
    //             &self.bias1,
    //             &self.weights2,
    //             &self.bias2,
    //         );
    //
    //         // Find the closest matching token by embedding distance
    //         let mut best_match = None;
    //         let mut best_score = f32::INFINITY;
    //
    //         for token in &self.tokens {
    //             let distance = output_embedding.iter()
    //                 .zip(token.embedding.iter())
    //                 .map(|(o, e)| (*o - *e).powi(2))
    //                 .sum::<f32>();
    //
    //             if distance < best_score {
    //                 best_score = distance;
    //                 best_match = Some(token.id.clone());
    //             }
    //         }
    //
    //         // Stop generation if <STOP> token is reached
    //         match best_match {
    //             Some(ref token) if token == "<STOP>" => break,
    //             Some(token) => output_sequence.push(token),
    //             None => return Err("No suitable token found".to_string()),
    //         }
    //
    //         // Update context embeddings with the newly generated token
    //         if let Some(idx) = self.get_token_id(output_sequence.last().unwrap()) {
    //             for (i, &val) in self.tokens[idx].embedding.iter().enumerate() {
    //                 context_embeddings[i] += val;
    //             }
    //         }
    //     }
    //
    //     Ok(output_sequence.join(" "))
    // }
}
